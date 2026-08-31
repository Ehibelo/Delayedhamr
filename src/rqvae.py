"""
RQ-VAE 基础模型 — MLP Encoder/Decoder + 官方 RQBottleneck

基于 kakaobrain/rq-vae-transformer 的 RQBottleneck，替换 Encoder/Decoder 为 MLP
参考文献：
- RQ-VAE (Lee et al., CVPR 2022): https://github.com/kakaobrain/rq-vae-transformer
- QuaSID (arxiv 2603.00632)
"""

import torch
from torch import nn
from torch.nn import functional as F

from quantization import RQBottleneck


class MLPEncoder(nn.Module):
    """MLP Encoder: 768 → 512 → 256 → 128 → 32 (TIGER paper config)"""

    def __init__(self, input_dim=768, latent_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)  # [B, latent_dim]


class MLPDecoder(nn.Module):
    """MLP Decoder: 32 → 128 → 256 → 512 → 768 (symmetric to encoder)"""

    def __init__(self, latent_dim=32, output_dim=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        # x: [B, latent_dim]
        return self.net(x)  # [B, output_dim]


class RQVAE(nn.Module):
    """
    RQ-VAE 基础模型

    架构: Item Feature → MLP Encoder → RQ-VAE Quantizer → MLP Decoder → Reconstructed Feature
    损失: L_recon (MSE) + β * L_commit

    参数:
        input_dim: 输入特征维度（文本嵌入维度）
        hidden_dim: MLP 隐层维度
        latent_dim: 潜在空间维度（每个 codebook entry 的维度）
        n_embed: 每个 codebook 的条目数 K
        n_codebook: RQ 层数 L
        decay: EMA 衰减率
        beta: commitment loss 权重
        dropout: Dropout 比率
    """

    def __init__(self,
                 input_dim=768,
                 latent_dim=32,
                 n_embed=256,
                 n_codebook=3,
                 decay=0.99,
                 beta=0.25,
                 dropout=0.1,
                 restart_unused_codes=True,
                 normalize_codebook=False,
                 hidden_dim=None):  # deprecated, kept for compat
        super().__init__()

        self.latent_dim = latent_dim
        self.n_codebook = n_codebook
        self.n_embed = n_embed
        self.beta = beta

        self.encoder = MLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # RQBottleneck: 将1D向量视为 (1, 1, d) 的空间特征
        # latent_shape=(1,1,d), code_shape=(1,1,L)
        self.quantizer = RQBottleneck(
            latent_shape=(1, 1, latent_dim),
            code_shape=(1, 1, n_codebook),
            n_embed=n_embed,
            decay=decay,
            shared_codebook=False,
            restart_unused_codes=restart_unused_codes,
            normalize_codebook=normalize_codebook,
        )

        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            dropout=dropout,
        )

    def encode(self, x):
        """编码输入特征为潜在向量 z"""
        z = self.encoder(x)  # [B, d]
        z = z.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, d] — 适配 RQBottleneck
        return z

    def decode(self, z_q):
        """解码量化特征"""
        z_q = z_q.squeeze(1).squeeze(1)  # [B, d] — 从 [B, 1, 1, d] 压平
        out = self.decoder(z_q)  # [B, input_dim]
        return out

    def forward(self, x):
        """
        Args:
            x: [B, input_dim] item 特征
        Returns:
            out: [B, input_dim] 重建特征
            quant_loss: commitment loss
            codes: [B, 1, 1, L] SID tokens
            z: [B, d] 编码器输出 (用于对比学习)
            z_q: [B, d] 量化后特征
        """
        z = self.encode(x)  # [B, 1, 1, d]
        z_q, quant_loss, codes = self.quantizer(z)  # z_q: [B, 1, 1, d], codes: [B, 1, 1, L]
        out = self.decode(z_q)

        z_flat = z.squeeze(1).squeeze(1)  # [B, d]
        z_q_flat = z_q.squeeze(1).squeeze(1)  # [B, d]

        return out, quant_loss, codes, z_flat, z_q_flat

    def compute_loss(self, out, quant_loss, target):
        """计算 RQ-VAE 重建损失"""
        loss_recon = F.mse_loss(out, target, reduction='mean')
        loss_total = loss_recon + self.beta * quant_loss
        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_latent': quant_loss,
        }

    @torch.no_grad()
    def get_codes(self, x):
        """获取 SID tokens（评估用）"""
        z = self.encode(x)
        _, _, codes = self.quantizer(z)
        return codes  # [B, 1, 1, L]

    @torch.no_grad()
    def get_encoder_emb(self, x):
        """获取编码器嵌入（用于余弦相似度评估）"""
        z = self.encoder(x)  # [B, d]
        return z

    @torch.no_grad()
    def decode_code(self, codes):
        """从 SID tokens 解码"""
        z_q = self.quantizer.embed_code(codes)
        out = self.decode(z_q)
        return out
