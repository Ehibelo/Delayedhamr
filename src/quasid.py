"""
QuaSID 完整模型 + Baseline 方法

- QuaSID: RQ-VAE + HaMR 损失 + 对比学习 + CVPM
- Improved VQGAN: RQ-VAE + LayerNorm + 感知损失
- SimRQ: RQ-VAE + 相似度保持损失
- RQ-VAE-Rotation: RQ-VAE + 正交旋转

参考：
- QuaSID 论文 (arxiv 2603.00632)
- kakaobrain/rq-vae-transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rqvae import RQVAE, MLPEncoder, MLPDecoder
from quantization import RQBottleneck


# ==============================================================================
# 辅助模块
# ==============================================================================

class L2Norm(nn.Module):
    """L2-normalization module (for use in nn.Sequential-like contexts)."""
    def forward(self, x):
        return F.normalize(x, dim=-1)


# ==============================================================================
# HaMR 损失实现
# ==============================================================================

def compute_hamming_distance(codes_i, codes_j):
    """
    计算两个 SID 码之间的 Hamming 距离

    Args:
        codes_i: [B, L] 或 [B, 1, 1, L] — SID tokens
        codes_j: [B, L] 或 [B, 1, 1, L] — SID tokens

    Returns:
        H: [B, B] Hamming 距离矩阵
    """
    # 压平到 [B, L]
    if codes_i.dim() == 4:
        codes_i = codes_i.squeeze(1).squeeze(1)
    if codes_j.dim() == 4:
        codes_j = codes_j.squeeze(1).squeeze(1)

    # H(i,j) = Σ_l I[s_i^(l) != s_j^(l)]
    # [B, 1, L] != [1, B, L] → [B, B, L] → sum → [B, B]
    H = (codes_i.unsqueeze(1) != codes_j.unsqueeze(0)).float().sum(dim=-1)
    return H


def compute_hamr_loss(codes, embeddings, item_ids,
                      m_full=0.5, m_partial=0.2, R=2,
                      lambda_full=1.0, lambda_partial=0.5,
                      use_cvpm=True, trigger_item_ids=None, target_item_ids=None):
    """
    HaMR 损失 (Hamming distance-based Margin Ranking Loss)

    Args:
        codes: [2B, 1, 1, L] — trigger + target 的 SID tokens
        embeddings: [2B, d] — trigger + target 的编码器嵌入
        item_ids: [2B] — trigger + target 的 item ID
        m_full: 全冲突 margin (H=0)
        m_partial: 部分冲突 margin (0 < H ≤ R)
        R: Hamming radius
        lambda_full: 全冲突损失权重
        lambda_partial: 部分冲突损失权重
        use_cvpm: 是否使用 CVPM (Collision-View Positive Masking)
        trigger_item_ids: [B] trigger 的 item ID (用于 CVPM)
        target_item_ids: [B] target 的 item ID (用于 CVPM)

    Returns:
        loss_hamr: scalar
    """
    B = codes.shape[0] // 2  # batch size

    # 展平 SID
    codes_flat = codes.squeeze(1).squeeze(1)  # [2B, L]

    # Hamming 距离矩阵 [2B, 2B]
    H = compute_hamming_distance(codes_flat, codes_flat)

    # 归一化嵌入用于余弦距离
    emb_norm = F.normalize(embeddings, dim=-1)
    cos_sim = emb_norm @ emb_norm.T  # [2B, 2B] 余弦相似度
    cos_dist = 1.0 - cos_sim  # [2B, 2B] 余弦距离

    # M_item: 排除相同 item ID 的对
    M_item = (item_ids.unsqueeze(1) != item_ids.unsqueeze(0)).float()

    # M_i2i (CVPM): 排除共现正样本对
    if use_cvpm and trigger_item_ids is not None and target_item_ids is not None:
        M_i2i = torch.ones(2 * B, 2 * B, device=codes.device)
        # 向量化: (trigger[i], target[i]) 和 (target[i], trigger[i])
        idx = torch.arange(B, device=codes.device)
        M_i2i[idx, idx + B] = 0
        M_i2i[idx + B, idx] = 0
        M_i2i.fill_diagonal_(0)
    else:
        # 不使用 CVPM：仅排除对角线
        M_i2i = torch.ones(2 * B, 2 * B, device=codes.device)
        M_i2i.fill_diagonal_(0)

    # 最终掩码
    M = M_i2i * M_item  # [2B, 2B]

    # 全冲突掩码: H=0 且 M=1
    M_full = ((H == 0) & (M == 1)).float()

    # 部分冲突掩码: 0 < H ≤ R 且 M=1
    M_partial = ((H > 0) & (H <= R) & (M == 1)).float()

    # 全冲突损失
    loss_full = F.relu(m_full - cos_dist)
    loss_full = (loss_full * M_full).sum() / (M_full.sum() + 1e-8)

    # 部分冲突损失
    loss_partial = F.relu(m_partial - cos_dist)
    loss_partial = (loss_partial * M_partial).sum() / (M_partial.sum() + 1e-8)

    loss_hamr = lambda_full * loss_full + lambda_partial * loss_partial

    return loss_hamr


# ==============================================================================
# QuaSID 完整模型
# ==============================================================================

class QuaSID(RQVAE):
    """
    QuaSID: Qualification-Aware Semantic ID Learning

    在 RQ-VAE 基础上添加:
    - 对比学习损失 (InfoNCE)
    - HaMR 损失 (全冲突 + 部分冲突 margin ranking)
    - CVPM (Collision-View Positive Masking)

    超参数:
        tau: 对比学习温度
        lambda_cl: 对比学习损失权重
        m_full: 全冲突 margin
        m_partial: 部分冲突 margin
        R: Hamming radius
        lambda_full: 全冲突 HaMR 权重
        lambda_partial: 部分冲突 HaMR 权重
        use_cvpm: 是否使用 CVPM
    """

    def __init__(self,
                 input_dim=768,
                 hidden_dim=512,
                 latent_dim=32,
                 n_embed=256,
                 n_codebook=3,
                 decay=0.99,
                 beta=0.25,
                 dropout=0.1,
                 restart_unused_codes=True,
                 tau=0.07,
                 lambda_cl=0.1,
                 m_full=0.8,        # Paper: 0.8 (full collision margin)
                 m_partial=0.5,     # Paper: 0.5 (partial collision margin)
                 R=1,               # Paper: R=1 for offline experiments
                 lambda_full=0.2,   # Paper: best at 0.2 (Figure 5b)
                 lambda_partial=0.1, # Paper: best at 0.1 (Figure 5c)
                 use_cvpm=True,
                 pre_quant_mode='identity',  # 'identity' | 'layernorm' | 'l2_norm'
                 normalize_codebook=False,   # True for ImpVQGAN+HaMR variants
                 **kwargs):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_embed=n_embed,
            n_codebook=n_codebook,
            decay=decay,
            beta=beta,
            dropout=dropout,
            restart_unused_codes=restart_unused_codes,
            normalize_codebook=normalize_codebook,
        )

        self.tau = tau
        self.lambda_cl = lambda_cl
        self.m_full = m_full
        self.m_partial = m_partial
        self.R = R
        self.lambda_full = lambda_full
        self.lambda_partial = lambda_partial
        self.use_cvpm = use_cvpm
        self.pre_quant_mode = pre_quant_mode

        # Pre-quantization normalization
        if pre_quant_mode == 'layernorm':
            self.pre_quant_norm = nn.LayerNorm(latent_dim)
        elif pre_quant_mode == 'l2_norm':
            self.pre_quant_norm = L2Norm()
        else:
            self.pre_quant_norm = nn.Identity()

    def forward(self, x):
        """带可选 LayerNorm 的 forward"""
        z_flat = self.encoder(x)  # [B, d]
        z_flat = self.pre_quant_norm(z_flat)  # LayerNorm (ImpVQGAN+HaMR) 或 Identity
        z = z_flat.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, d]
        z_q, quant_loss, codes = self.quantizer(z)
        out = self.decode(z_q)
        z_q_flat = z_q.squeeze(1).squeeze(1)
        return out, quant_loss, codes, z_flat, z_q_flat

    @torch.no_grad()
    def get_encoder_emb(self, x):
        """获取编码器嵌入（应用 pre_quant_norm，与 forward 一致）"""
        z = self.encoder(x)  # [B, d]
        z = self.pre_quant_norm(z)
        return z

    def forward_contrastive_pair(self, trigger_feat, target_feat):
        """
        处理对比学习对：trigger 和 target 分别通过 encoder + quantizer

        Args:
            trigger_feat: [B, input_dim] trigger 特征
            target_feat: [B, input_dim] target 特征

        Returns:
            out: [2B, input_dim] 重建 (trigger + target)
            quant_loss: scalar
            codes: [2B, 1, 1, L]
            z: [2B, d] 编码器嵌入
            z_q: [2B, d] 量化后特征
        """
        # 拼接 trigger 和 target
        feats = torch.cat([trigger_feat, target_feat], dim=0)  # [2B, input_dim]
        return self.forward(feats)

    def compute_contrastive_loss(self, z, trigger_item_ids, target_item_ids):
        """
        InfoNCE 对比学习损失

        Args:
            z: [2B, d] — 前半是 trigger，后半是 target
            trigger_item_ids: [B]
            target_item_ids: [B]

        Returns:
            loss_cl: scalar
        """
        B = z.shape[0] // 2
        z_trigger = z[:B]   # [B, d]
        z_target = z[B:]    # [B, d]

        # L2 归一化
        z_trigger_norm = F.normalize(z_trigger, dim=-1)
        z_target_norm = F.normalize(z_target, dim=-1)

        # 相似度矩阵: [B, B]
        sim_matrix = z_trigger_norm @ z_target_norm.T / self.tau

        # 标签: 对角线是正样本对 (trigger[i], target[i])
        labels = torch.arange(B, device=z.device)

        # 双向 InfoNCE
        loss_forward = F.cross_entropy(sim_matrix, labels)
        loss_backward = F.cross_entropy(sim_matrix.T, labels)

        loss_cl = (loss_forward + loss_backward) / 2.0
        return loss_cl

    def compute_loss(self, out, quant_loss, codes, z,
                     trigger_feat, target_feat,
                     trigger_item_ids, target_item_ids,
                     target=None, valid=False):
        """
        计算 QuaSID 完整损失

        Returns:
            dict with loss_total, loss_recon, loss_latent, loss_cl, loss_hamr
        """
        B = out.shape[0] // 2

        # 重建损失（对 trigger+target 拼接后重建）
        xs = torch.cat([trigger_feat, target_feat], dim=0)
        loss_recon = F.mse_loss(out, xs, reduction='mean')
        loss_total = loss_recon + self.beta * quant_loss

        # 对比学习损失
        loss_cl = self.compute_contrastive_loss(z, trigger_item_ids, target_item_ids)
        loss_total = loss_total + self.lambda_cl * loss_cl

        # HaMR 损失
        item_ids = torch.cat([trigger_item_ids, target_item_ids], dim=0)
        loss_hamr = compute_hamr_loss(
            codes, z,
            item_ids=item_ids,
            m_full=self.m_full,
            m_partial=self.m_partial,
            R=self.R,
            lambda_full=self.lambda_full,
            lambda_partial=self.lambda_partial,
            use_cvpm=self.use_cvpm,
            trigger_item_ids=trigger_item_ids,
            target_item_ids=target_item_ids,
        )
        loss_total = loss_total + loss_hamr

        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_latent': quant_loss,
            'loss_cl': loss_cl,
            'loss_hamr': loss_hamr,
        }


# ==============================================================================
# Baseline: Improved VQGAN
# ==============================================================================

class ImprovedVQGAN(RQVAE):
    """
    Improved VQGAN (Yu et al., ICLR 2022):
    - L2-normalize encoder output + codebook vectors
    - Cosine-similarity-based code assignment (via normalized codebook)
    - Perceptual loss between normalized encoder output and quantized output

    Reference: "Vector-quantized Image Modeling with Improved VQGAN"
    """

    def __init__(self,
                 input_dim=768,
                 hidden_dim=512,
                 latent_dim=32,
                 n_embed=256,
                 n_codebook=3,
                 decay=0.99,
                 beta=0.25,
                 dropout=0.1,
                 restart_unused_codes=True,
                 perceptual_weight=0.1,
                 **kwargs):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_embed=n_embed,
            n_codebook=n_codebook,
            decay=decay,
            beta=beta,
            dropout=dropout,
            restart_unused_codes=restart_unused_codes,
            normalize_codebook=True,  # L2-normalize codebook for cosine assignment
        )

        self.perceptual_weight = perceptual_weight

    def forward(self, x):
        """L2-normalize encoder output before quantization"""
        z_flat = self.encoder(x)  # [B, d]
        # L2-normalize for cosine-similarity-based assignment (ImpVQGAN)
        z_flat = F.normalize(z_flat, dim=-1)
        z = z_flat.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, d]
        z_q, quant_loss, codes = self.quantizer(z)
        out = self.decode(z_q)
        z_q_flat = z_q.squeeze(1).squeeze(1)
        return out, quant_loss, codes, z_flat, z_q_flat

    @torch.no_grad()
    def get_encoder_emb(self, x):
        """获取编码器嵌入（L2-normalize，与 forward 一致）"""
        z = self.encoder(x)  # [B, d]
        z = F.normalize(z, dim=-1)
        return z

    def compute_loss(self, out, quant_loss, target, z, z_q):
        """MSE + β*commitment + perceptual_weight * perceptual_loss"""
        loss_recon = F.mse_loss(out, target, reduction='mean')

        # Perceptual loss: MSE between L2-normalized encoder output
        # and quantized output (both on unit sphere)
        perceptual_loss = F.mse_loss(z_q, z, reduction='mean')

        loss_total = loss_recon + self.beta * quant_loss + self.perceptual_weight * perceptual_loss

        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_latent': quant_loss,
            'loss_perceptual': perceptual_loss,
        }


# ==============================================================================
# Baseline: RQ-VAE-Rotation
# ==============================================================================

class RQVAERotation(RQVAE):
    """
    RQ-VAE-Rotation:
    - 量化前施加正交旋转矩阵（QR 分解初始化，固定不变）
    - 重建时施加逆旋转

    旋转矩阵在初始化时通过 QR 分解生成并固定
    """

    def __init__(self,
                 input_dim=768,
                 hidden_dim=512,
                 latent_dim=32,
                 n_embed=256,
                 n_codebook=3,
                 decay=0.99,
                 beta=0.25,
                 dropout=0.1,
                 restart_unused_codes=True,
                 **kwargs):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_embed=n_embed,
            n_codebook=n_codebook,
            decay=decay,
            beta=beta,
            dropout=dropout,
            restart_unused_codes=restart_unused_codes,
        )

        # 生成随机正交旋转矩阵
        with torch.no_grad():
            R = torch.randn(latent_dim, latent_dim)
            Q, _ = torch.linalg.qr(R)
            self.register_buffer('rotation', Q)  # [d, d]
            self.register_buffer('rotation_inv', Q.T)  # [d, d]

    def forward(self, x):
        z_flat = self.encoder(x)  # [B, d]

        # 正交旋转
        z_rotated = z_flat @ self.rotation  # [B, d]

        z = z_rotated.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, d]
        z_q, quant_loss, codes = self.quantizer(z)
        z_q_flat = z_q.squeeze(1).squeeze(1)  # [B, d]

        # 逆旋转
        z_q_inv = z_q_flat @ self.rotation_inv  # [B, d]

        out = self.decoder(z_q_inv)  # [B, input_dim]
        return out, quant_loss, codes, z_flat, z_q_inv


# ==============================================================================
# Baseline: SimRQ
# ==============================================================================

class SimRQ(RQVAE):
    """
    SimRQ (Zhu et al., ICCV 2023):
    - RQ-VAE + 相似度保持损失
    - Freezes codebooks after initialization
    - Uses frozen codebook as implicit linear projection for discrete codes

    Collaborative signal: Swing co-occurrence similarity matrix
    Loss: MSE to align encoder embedding similarity with Swing similarity
    """

    def __init__(self,
                 input_dim=768,
                 hidden_dim=512,
                 latent_dim=32,
                 n_embed=256,
                 n_codebook=3,
                 decay=0.99,
                 beta=0.25,
                 dropout=0.1,
                 restart_unused_codes=True,
                 sim_loss_weight=0.1,
                 **kwargs):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_embed=n_embed,
            n_codebook=n_codebook,
            decay=decay,
            beta=beta,
            dropout=dropout,
            restart_unused_codes=restart_unused_codes,
        )

        self.sim_loss_weight = sim_loss_weight

        # Freeze codebooks (SimRQ freezes codebooks after init)
        self._freeze_codebooks()

    def _freeze_codebooks(self):
        """Freeze all codebook parameters and disable EMA updates."""
        for codebook in self.quantizer.codebooks:
            codebook.ema = False
            for p in codebook.parameters():
                p.requires_grad_(False)

    def _quantize_frozen(self, z):
        """Quantize with frozen codebook (no EMA update; STE gradient passes through).

        NOTE: We do NOT use @torch.no_grad() here because the Straight-Through
        Estimator in RQBottleneck relies on autograd to propagate the
        reconstruction gradient back to the encoder. The codebook itself is
        frozen via requires_grad=False and ema=False, but the STE path
        (x + (quant - x).detach()) must execute inside an autograd-enabled
        context for the encoder to receive gradient from reconstruction loss.
        """
        return self.quantizer(z)

    def forward(self, x):
        z_flat = self.encoder(x)  # [B, d]
        z = z_flat.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, d]
        # Quantize with frozen codebook
        z_q, quant_loss, codes = self._quantize_frozen(z)
        out = self.decode(z_q)
        z_q_flat = z_q.squeeze(1).squeeze(1)
        return out, quant_loss, codes, z_flat, z_q_flat

    def compute_loss(self, out, quant_loss, target, z, swing_sim_batch=None):
        """MSE + β*commitment + sim_loss_weight * sim_preserving_loss"""
        loss_recon = F.mse_loss(out, target, reduction='mean')
        loss_total = loss_recon + self.beta * quant_loss

        loss_sim = torch.tensor(0.0, device=out.device)
        if swing_sim_batch is not None:
            # Encoder embedding similarity matrix
            z_norm = F.normalize(z, dim=-1)
            sim_pred = z_norm @ z_norm.T  # [B, B]

            # MSE against Swing similarity matrix
            loss_sim = F.mse_loss(sim_pred, swing_sim_batch, reduction='mean')
            loss_total = loss_total + self.sim_loss_weight * loss_sim

        return {
            'loss_total': loss_total,
            'loss_recon': loss_recon,
            'loss_latent': quant_loss,
            'loss_sim': loss_sim,
        }


# ==============================================================================
# Baseline: RQ-KMeans
# ==============================================================================

class RQKMeans(RQVAE):
    """
    RQ-KMeans: RQ-VAE with KMeans++ initialization for codebooks.

    Instead of random initialization, codebook vectors are initialized via
    KMeans clustering on a sample of encoder outputs. The residual
    quantization structure is preserved — each codebook is initialized
    sequentially on the residuals from previous codebooks.

    Reference: standard KMeans initialization for VQ (used as baseline).
    """

    def __init__(self,
                 input_dim=768,
                 hidden_dim=512,
                 latent_dim=32,
                 n_embed=256,
                 n_codebook=3,
                 decay=0.99,
                 beta=0.25,
                 dropout=0.1,
                 restart_unused_codes=True,
                 kmeans_init_samples=1024,
                 **kwargs):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_embed=n_embed,
            n_codebook=n_codebook,
            decay=decay,
            beta=beta,
            dropout=dropout,
            restart_unused_codes=restart_unused_codes,
        )
        self.kmeans_init_samples = kmeans_init_samples
        self._kmeans_initialized = False

    def kmeans_init_codebooks(self, dataloader, device):
        """
        Initialize codebooks via sequential KMeans on residuals.

        For each codebook in order:
          1. Sample encoder outputs (or residuals from previous codebooks)
          2. Run KMeans clustering (scikit-learn or faiss)
          3. Set codebook vectors to cluster centroids
        """
        from sklearn.cluster import MiniBatchKMeans

        self.eval()
        all_embeddings = []

        # Collect encoder outputs
        with torch.no_grad():
            for feats, _ in dataloader:
                feats = feats.to(device)
                z = self.encoder(feats)
                all_embeddings.append(z.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        # Sequential KMeans per codebook
        residual = all_embeddings.copy()
        for cb_idx, codebook in enumerate(self.quantizer.codebooks):
            # Subsample if needed
            n_samples = min(len(residual), self.kmeans_init_samples)
            indices = np.random.choice(len(residual), n_samples, replace=False)
            sample = residual[indices]

            k = min(codebook.n_embed, n_samples)
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
            kmeans.fit(sample)

            # Set codebook vectors (VQEmbedding inherits nn.Embedding, has padding_idx)
            centroids = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
            with torch.no_grad():
                codebook.weight.data[:k] = centroids  # avoid overwriting padding row
                # Normalize if needed
                if hasattr(codebook, 'normalize_embed'):
                    codebook.normalize_embed()

            # Compute residual for next codebook
            labels = kmeans.predict(residual)
            residual = residual - kmeans.cluster_centers_[labels]

        self._kmeans_initialized = True
        return self

    def forward(self, x):
        return super().forward(x)


# ==============================================================================
# Baseline: GRVQ (Gumbel-softmax RQ-VAE)
# ==============================================================================

class GRVQ(RQVAE):
    """
    GRVQ: Gumbel-softmax Residual Vector Quantization.

    Replaces the hard nearest-neighbor code assignment with Gumbel-softmax
    relaxation during training. At inference, uses argmax (hard assignment).

    The Gumbel-softmax temperature τ controls the softness:
    - τ → 0: hard assignment (standard RQ-VAE)
    - τ → ∞: uniform distribution

    During training, temperature is annealed from τ_start to τ_min.
    """

    def __init__(self,
                 input_dim=768,
                 hidden_dim=512,
                 latent_dim=32,
                 n_embed=256,
                 n_codebook=3,
                 decay=0.99,
                 beta=0.25,
                 dropout=0.1,
                 restart_unused_codes=True,
                 gumbel_temperature=0.5,
                 gumbel_tau_min=0.1,
                 gumbel_anneal_rate=0.9999,
                 **kwargs):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_embed=n_embed,
            n_codebook=n_codebook,
            decay=decay,
            beta=beta,
            dropout=dropout,
            restart_unused_codes=restart_unused_codes,
        )
        self.gumbel_temperature = gumbel_temperature
        self.gumbel_tau_min = gumbel_tau_min
        self.gumbel_anneal_rate = gumbel_anneal_rate
        self.training_steps = 0

    def forward(self, x):
        """
        Forward with Gumbel-softmax relaxation.

        During training: soft assignment via Gumbel-softmax
        During eval: hard assignment via argmax
        """
        z_flat = self.encoder(x)  # [B, d]
        z = z_flat.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, d]

        if self.training:
            # Anneal temperature
            tau = max(self.gumbel_tau_min,
                      self.gumbel_temperature * (self.gumbel_anneal_rate ** self.training_steps))
            self.training_steps += 1

            z_q, quant_loss, codes = self._gumbel_quantize(z, tau)
        else:
            z_q, quant_loss, codes = self.quantizer(z)

        out = self.decode(z_q)
        return out, quant_loss, codes, z_flat, z_q.squeeze(1).squeeze(1)

    def _gumbel_quantize(self, z, tau):
        """
        Gumbel-softmax quantization for a single codebook layer.

        For each codebook, computes soft assignment weights via:
          w_i = softmax((-dist(z, e_j) + g_i) / τ)
          z_q = Σ w_j * e_j

        Returns quantized vectors, commitment loss, and hard codes (for logging).
        """
        B = z.size(0)
        d = z.size(-1)
        z_flat = z.squeeze(1).squeeze(1)  # [B, d]
        residual = z_flat
        z_q_list = []
        codes_list = []
        total_loss = 0.0

        for codebook in self.quantizer.codebooks:
            # Get codebook vectors
            if hasattr(codebook, 'get_embeddings'):
                emb = codebook.get_embeddings()  # [K, d]
            else:
                emb = codebook.weight[:codebook.n_embed]  # [K, d] (exclude padding)

            # Compute distances
            # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z@e.T
            dist = (residual ** 2).sum(dim=1, keepdim=True) + \
                   (emb ** 2).sum(dim=1) - \
                   2 * residual @ emb.T  # [B, K]

            # Gumbel-softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(dist) + 1e-8) + 1e-8)
            logits = -dist / tau + gumbel_noise
            soft_assign = F.softmax(logits, dim=-1)  # [B, K]

            # Soft quantize: weighted sum of codebook vectors
            z_q_soft = soft_assign @ emb  # [B, d]

            # Hard codes for logging/analysis
            hard_codes = dist.argmin(dim=-1)  # [B]

            # Commitment loss
            commitment_loss = F.mse_loss(z_q_soft, residual.detach()) + \
                             0.25 * F.mse_loss(residual, z_q_soft.detach())

            total_loss = total_loss + commitment_loss
            z_q_list.append(z_q_soft)
            codes_list.append(hard_codes)
            residual = residual - z_q_soft

        z_q = torch.stack(z_q_list, dim=0).sum(dim=0)  # [B, d]
        codes = torch.stack(codes_list, dim=-1)  # [B, L]

        return z_q.unsqueeze(1).unsqueeze(1), total_loss, codes.unsqueeze(1).unsqueeze(1)


# ==============================================================================
# Baseline: RQ-OPQ (Residual Optimized Product Quantization)
# ==============================================================================

class RQOPQ(RQVAE):
    """
    RQ-OPQ: Residual Quantization with Optimized Product Quantization.

    OPQ learns an orthogonal rotation matrix to minimize quantization error.
    Unlike RQ-VAE-Rotation (fixed QR rotation), OPQ learns the rotation
    matrix jointly with the codebooks.

    The rotation matrix R is parameterized as an orthogonal matrix
    and updated via gradient descent with orthogonal constraint.

    Reference: "Optimized Product Quantization" (Ge et al., TPAMI 2014)
    """

    def __init__(self,
                 input_dim=768,
                 hidden_dim=512,
                 latent_dim=32,
                 n_embed=256,
                 n_codebook=3,
                 decay=0.99,
                 beta=0.25,
                 dropout=0.1,
                 restart_unused_codes=True,
                 opq_rotation_lr=0.001,
                 **kwargs):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_embed=n_embed,
            n_codebook=n_codebook,
            decay=decay,
            beta=beta,
            dropout=dropout,
            restart_unused_codes=restart_unused_codes,
        )

        # Learnable rotation matrix (initialized as identity)
        self.rotation_raw = nn.Parameter(torch.eye(latent_dim))
        self.opq_rotation_lr = opq_rotation_lr

    def _get_rotation(self):
        """Get orthogonal rotation matrix via Cayley transform or QR."""
        # Use QR decomposition to maintain orthogonality
        Q, _ = torch.linalg.qr(self.rotation_raw)
        return Q

    def forward(self, x):
        z_flat = self.encoder(x)  # [B, d]

        # Apply learned OPQ rotation
        R = self._get_rotation()
        z_rotated = z_flat @ R  # [B, d]

        z = z_rotated.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, d]
        z_q, quant_loss, codes = self.quantizer(z)
        z_q_flat = z_q.squeeze(1).squeeze(1)  # [B, d]

        # Inverse rotation
        z_q_inv = z_q_flat @ R.T  # [B, d]

        out = self.decoder(z_q_inv)
        return out, quant_loss, codes, z_flat, z_q_inv
