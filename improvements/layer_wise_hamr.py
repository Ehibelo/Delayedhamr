"""
Layer-wise HaMR — 分层加权 Hamming 距离
========================================

动机:
  标准 HaMR 将所有层的码本冲突等权对待。
  但实际上不同层承载了不同粒度的语义:
  - c1 (第一层): 粗粒度 (大类目), 冲突意味着顶层语义不同 → 应被更强烈地推开
  - c2 (第二层): 中粒度 (子类目)
  - c3 (第三层): 细粒度 (具体属性), 冲突可能是噪声

方法:
  在计算 Hamming 距离时加权:
    H_weighted = Σ w_l * I[s_i^l != s_j^l]

  粗粒度层 (c1) 权重更大 → 仅 c1 冲突被视为更严重的冲突

三种权重设定:
  - 'front_heavy':  w = [0.6, 0.3, 0.1]  粗粒度主导
  - 'balanced':     w = [0.5, 0.3, 0.2]  渐进递减
  - 'uniform':      w = [1/3, 1/3, 1/3]  等价于标准 HaMR (对照)

用法:
  from improvements.layer_wise_hamr import QuaSIDLayerWise, train_layerwise
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from quasid import QuaSID
from curriculum_hamr import (evaluate_cosine, compute_entropy,
                             train_quasid_standard)


# ═══════════════════════════════════════════════════════════════════════════════
# 分层 Hamming 距离 + Layer-wise HaMR 损失
# ═══════════════════════════════════════════════════════════════════════════════

def compute_weighted_hamming(codes_i, codes_j, layer_weights):
    """计算加权 Hamming 距离

    Args:
        codes_i, codes_j: [B, L] SID tokens
        layer_weights: [L] 各层权重
    Returns:
        H_weighted: [B, B] 加权 Hamming 距离 (非整数, 连续值)
    """
    if codes_i.dim() == 4:
        codes_i = codes_i.squeeze(1).squeeze(1)
    if codes_j.dim() == 4:
        codes_j = codes_j.squeeze(1).squeeze(1)

    L = codes_i.shape[-1]
    w = torch.tensor(layer_weights, device=codes_i.device, dtype=torch.float)

    # [B, 1, L] != [1, B, L] → [B, B, L] → weighted sum → [B, B]
    diff = (codes_i.unsqueeze(1) != codes_j.unsqueeze(0)).float()  # [B, B, L]
    H_weighted = (diff * w).sum(dim=-1)  # [B, B]
    return H_weighted


def compute_layer_hamr_loss(codes, embeddings, item_ids,
                            layer_weights,
                            m_full=0.8, m_partial=0.5, R=1,
                            lambda_full=0.2, lambda_partial=0.1,
                            use_cvpm=True,
                            trigger_item_ids=None, target_item_ids=None):
    """Layer-wise HaMR: 用加权 Hamming 距离替代标准 Hamming 距离

    参数与 compute_hamr_loss 完全一致，额外增加 layer_weights [L]。
    """
    B = codes.shape[0] // 2
    codes_flat = codes.squeeze(1).squeeze(1)  # [2B, L]

    # ★ 核心变化: 加权 Hamming 距离
    H = compute_weighted_hamming(codes_flat, codes_flat, layer_weights)

    emb_norm = F.normalize(embeddings, dim=-1)
    cos_dist = 1.0 - emb_norm @ emb_norm.T

    M_item = (item_ids.unsqueeze(1) != item_ids.unsqueeze(0)).float()

    if use_cvpm and trigger_item_ids is not None and target_item_ids is not None:
        M_i2i = torch.ones(2 * B, 2 * B, device=codes.device)
        idx = torch.arange(B, device=codes.device)
        M_i2i[idx, idx + B] = 0
        M_i2i[idx + B, idx] = 0
        M_i2i.fill_diagonal_(0)
    else:
        M_i2i = torch.ones(2 * B, 2 * B, device=codes.device)
        M_i2i.fill_diagonal_(0)

    M = M_i2i * M_item

    # 全冲突: H_weighted == 0 (所有层都匹配)
    M_full = ((H == 0) & (M == 1)).float()

    # 部分冲突: 0 < H_weighted <= R (注意 R 需要根据 layer_weights 调整)
    # 因为加权后 H 的最大值是 sum(layer_weights) ≈ 1.0 (而非整数)
    # R 应被调整为 R_effective = R * mean(layer_weight)
    R_eff = R * np.mean(layer_weights)
    M_partial = ((H > 0) & (H <= R_eff) & (M == 1)).float()

    loss_full = F.relu(m_full - cos_dist)
    loss_full = (loss_full * M_full).sum() / (M_full.sum() + 1e-8)

    loss_partial = F.relu(m_partial - cos_dist)
    loss_partial = (loss_partial * M_partial).sum() / (M_partial.sum() + 1e-8)

    return lambda_full * loss_full + lambda_partial * loss_partial


# ═══════════════════════════════════════════════════════════════════════════════
# QuaSIDLayerWise — 分层加权 HaMR
# ═══════════════════════════════════════════════════════════════════════════════

class QuaSIDLayerWise(QuaSID):
    """QuaSID + Layer-wise HaMR: 不同码本层的冲突施加不同权重

    额外参数:
        layer_weights: list of float, 长度 = n_codebook (默认 [0.6, 0.3, 0.1])
                       控制各层在 Hamming 距离计算中的权重
    """

    def __init__(self, layer_weights=None, **kwargs):
        super().__init__(**kwargs)

        if layer_weights is None:
            # 默认: 粗粒度层权重大
            layer_weights = [0.6, 0.3, 0.1]
        self.layer_weights = layer_weights
        assert len(self.layer_weights) == self.n_codebook, \
            f"layer_weights length ({len(self.layer_weights)}) != n_codebook ({self.n_codebook})"
        assert abs(sum(self.layer_weights) - 1.0) < 0.01, \
            f"layer_weights must sum to 1.0, got {sum(self.layer_weights)}"

    def compute_loss(self, out, quant_loss, codes, z,
                     trigger_feat, target_feat,
                     trigger_item_ids, target_item_ids,
                     target=None, valid=False):
        """与 QuaSID.compute_loss 相同，仅 HaMR 部分替换为分层版本"""
        B = out.shape[0] // 2

        xs = torch.cat([trigger_feat, target_feat], dim=0)
        loss_recon = F.mse_loss(out, xs, reduction='mean')
        loss_total = loss_recon + self.beta * quant_loss

        loss_cl = self.compute_contrastive_loss(z, trigger_item_ids, target_item_ids)
        loss_total = loss_total + self.lambda_cl * loss_cl

        # ★ 替换为 Layer-wise HaMR
        item_ids = torch.cat([trigger_item_ids, target_item_ids], dim=0)
        loss_hamr = compute_layer_hamr_loss(
            codes, z, item_ids,
            layer_weights=self.layer_weights,
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


# ═══════════════════════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════════════════════

def train_layerwise(model, pair_loader, device, epochs=200, patience=10,
                    lr=3e-4, weight_decay=1e-5, log_interval=10):
    """训练 QuaSIDLayerWise"""
    return train_quasid_standard(model, pair_loader, device,
                                 epochs=epochs, patience=patience,
                                 lr=lr, weight_decay=weight_decay,
                                 log_interval=log_interval)


# ═══════════════════════════════════════════════════════════════════════════════
# 预设配置
# ═══════════════════════════════════════════════════════════════════════════════

LAYERWISE_CONFIGS = {
    'standard': {
        'desc': '标准 QuaSID (等权 [1/3, 1/3, 1/3])',
        'factory': lambda cfg: QuaSID(**cfg),
    },
    'front_heavy': {
        'desc': 'LayerWise-FrontHeavy: w=[0.6,0.3,0.1] c1突变=严重冲突',
        'factory': lambda cfg: QuaSIDLayerWise(
            layer_weights=[0.6, 0.3, 0.1], **cfg),
    },
    'balanced': {
        'desc': 'LayerWise-Balanced: w=[0.5,0.3,0.2] 渐进递减',
        'factory': lambda cfg: QuaSIDLayerWise(
            layer_weights=[0.5, 0.3, 0.2], **cfg),
    },
    'back_heavy': {
        'desc': 'LayerWise-BackHeavy: w=[0.1,0.3,0.6] c3突变=严重冲突 (反直觉对照)',
        'factory': lambda cfg: QuaSIDLayerWise(
            layer_weights=[0.1, 0.3, 0.6], **cfg),
    },
}
