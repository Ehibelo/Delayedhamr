"""
VQEmbedding + RQBottleneck — 基于 kakaobrain/rq-vae-transformer 官方实现
修改：适配1D特征向量（非图像），保持核心量化逻辑不变

参考：https://github.com/kakaobrain/rq-vae-transformer
"""

from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class VQEmbedding(nn.Embedding):
    """VQ embedding module with EMA update.

    与官方实现完全一致，仅移除分布式相关的 all_reduce 调用中的 guard。

    Args:
        normalize_codebook: If True, L2-normalize codebook entries after EMA
            update, enabling cosine-similarity-based assignment.
            Used by Improved VQGAN (Yu et al., ICLR 2022).
    """

    def __init__(self, n_embed, embed_dim, ema=True, decay=0.99,
                 restart_unused_codes=True, eps=1e-5, normalize_codebook=False):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed
        self.normalize_codebook = normalize_codebook

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            self.register_buffer('cluster_size_ema', torch.zeros(n_embed))
            self.register_buffer('embed_ema', self.weight[:-1, :].detach().clone())

        # Initialize normalized codebook if needed
        if self.normalize_codebook:
            with torch.no_grad():
                self.weight[:-1, :] = F.normalize(self.weight[:-1, :], dim=-1)

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()  # [d, K]

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        if self.normalize_codebook:
            # Cosine-similarity-based assignment (Improved VQGAN)
            # L2-normalize both inputs and codebook; distance = 1 - cos(x,y)
            inputs_flat = F.normalize(inputs_flat, dim=-1)
            codebook_t = F.normalize(codebook_t, dim=0)
            # cos_sim = inputs_flat @ codebook_t → [N, K]
            # distance = 1 - cos_sim ∈ [0, 2], lower = more similar
            distances = 1.0 - torch.matmul(inputs_flat, codebook_t)
        else:
            # Standard Euclidean distance: ∥x - y∥² = ∥x∥² + ∥y∥² - 2⟨x,y⟩
            inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
            codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
            distances = torch.addmm(
                inputs_norm_sq + codebook_t_norm_sq,
                inputs_flat,
                codebook_t,
                alpha=-2.0,
            )
        distances = distances.reshape(*inputs_shape[:-1], -1)
        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)
        embed_idxs = distances.argmin(dim=-1)
        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):
        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(dim=0,
                              index=idxs.unsqueeze(0),
                              src=vectors.new_ones(1, n_vectors))

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(vectors_sum_per_cluster, alpha=1 - self.decay)

        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][:n_embed]

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1))

    @torch.no_grad()
    def _update_embedding(self):
        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)
        # L2-normalize codebook entries for cosine-similarity assignment
        if self.normalize_codebook:
            self.weight[:-1, :] = F.normalize(self.weight[:-1, :], dim=-1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds


class RQBottleneck(nn.Module):
    """
    Residual Quantization bottleneck.
    适配1D特征向量：latent_shape=(1, 1, d), code_shape=(1, 1, L)

    基于 kakaobrain/rq-vae-transformer 官方实现
    """

    def __init__(self,
                 latent_shape,
                 code_shape,
                 n_embed,
                 decay=0.99,
                 shared_codebook=False,
                 restart_unused_codes=True,
                 commitment_loss='cumsum',
                 normalize_codebook=False):
        super().__init__()

        if not len(code_shape) == len(latent_shape) == 3:
            raise ValueError("incompatible code shape or latent shape")
        if any([y % x != 0 for x, y in zip(code_shape[:2], latent_shape[:2])]):
            raise ValueError("incompatible code shape or latent shape")

        embed_dim = np.prod(latent_shape[:2]) // np.prod(code_shape[:2]) * latent_shape[2]

        self.latent_shape = torch.Size(latent_shape)
        self.code_shape = torch.Size(code_shape)
        self.shape_divisor = torch.Size([latent_shape[i] // code_shape[i] for i in range(len(latent_shape))])
        self.normalize_codebook = normalize_codebook

        self.shared_codebook = shared_codebook
        if self.shared_codebook:
            if isinstance(n_embed, Iterable) or isinstance(decay, Iterable):
                raise ValueError("Shared codebooks are incompatible with list types")

        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed if isinstance(n_embed, Iterable) else [n_embed for _ in range(self.code_shape[-1])]
        self.decay = decay if isinstance(decay, Iterable) else [decay for _ in range(self.code_shape[-1])]
        assert len(self.n_embed) == self.code_shape[-1]
        assert len(self.decay) == self.code_shape[-1]

        if self.shared_codebook:
            codebook0 = VQEmbedding(self.n_embed[0],
                                    embed_dim,
                                    decay=self.decay[0],
                                    restart_unused_codes=restart_unused_codes,
                                    normalize_codebook=normalize_codebook)
            self.codebooks = nn.ModuleList([codebook0 for _ in range(self.code_shape[-1])])
        else:
            codebooks = [VQEmbedding(self.n_embed[idx],
                                     embed_dim,
                                     decay=self.decay[idx],
                                     restart_unused_codes=restart_unused_codes,
                                     normalize_codebook=normalize_codebook)
                         for idx in range(self.code_shape[-1])]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss

    def to_code_shape(self, x):
        (B, H, W, D) = x.shape
        (rH, rW, _) = self.shape_divisor

        x = x.reshape(B, H // rH, rH, W // rW, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H // rH, W // rW, -1)
        return x

    def to_latent_shape(self, x):
        (B, h, w, _) = x.shape
        (_, _, D) = self.latent_shape
        (rH, rW, _) = self.shape_divisor

        x = x.reshape(B, h, w, rH, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, h * rH, w * rW, D)
        return x

    def quantize(self, x):
        """
        残差量化：逐层量化残差

        Returns:
            quant_list: 各层累积量化特征
            codes: [B, h, w, L] SID token 索引
        """
        B, h, w, embed_dim = x.shape

        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(self.code_shape[-1]):
            quant, code = self.codebooks[i](residual_feature)

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def forward(self, x):
        x_reshaped = self.to_code_shape(x)
        quant_list, codes = self.quantize(x_reshaped)

        commitment_loss = self.compute_commitment_loss(x_reshaped, quant_list)
        quants_trunc = self.to_latent_shape(quant_list[-1])
        # Straight-through estimator
        quants_trunc = x + (quants_trunc - x).detach()

        return quants_trunc, commitment_loss, codes

    def compute_commitment_loss(self, x, quant_list):
        """计算 commitment loss"""
        loss_list = []
        for idx, quant in enumerate(quant_list):
            partial_loss = (x - quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)
        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss

    @torch.no_grad()
    def embed_code(self, code):
        """将 SID token 解码为量化特征"""
        assert code.shape[1:] == self.code_shape

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [self.codebooks[0].embed(code_slice) for i, code_slice in enumerate(code_slices)]
        else:
            embeds = [self.codebooks[i].embed(code_slice) for i, code_slice in enumerate(code_slices)]

        embeds = torch.cat(embeds, dim=-2).sum(-2)
        embeds = self.to_latent_shape(embeds)
        return embeds

    @torch.no_grad()
    def get_soft_codes(self, x, temp=1.0, stochastic=False):
        """获取 soft codes（用于评估）"""
        x = self.to_code_shape(x)

        residual_feature = x.detach().clone()
        soft_code_list = []
        code_list = []

        n_codebooks = self.code_shape[-1]
        for i in range(n_codebooks):
            codebook = self.codebooks[i]
            distances = codebook.compute_distances(residual_feature)
            soft_code = F.softmax(-distances / temp, dim=-1)

            if stochastic:
                soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
                code = torch.multinomial(soft_code_flat, 1)
                code = code.reshape(*soft_code.shape[:-1])
            else:
                code = distances.argmin(dim=-1)
            quants = codebook.embed(code)
            residual_feature -= quants

            code_list.append(code.unsqueeze(-1))
            soft_code_list.append(soft_code.unsqueeze(-2))

        code = torch.cat(code_list, dim=-1)
        soft_code = torch.cat(soft_code_list, dim=-2)
        return soft_code, code
