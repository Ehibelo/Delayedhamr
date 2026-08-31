"""
Hard Negative Mining 增强对比学习
==================================

动机:
  标准 InfoNCE 对所有 batch 内负样本等权处理。
  但真正有信息量的是与 anchor 最相似的"难"负样本——
  那些已被轻松区分的负样本梯度接近零，浪费了对比学习的容量。

方法:
  在 sim_matrix 中识别每个 anchor 的 top-K hardest negatives，
  在 softmax 分母中放大其 logit，等价于对 hard negatives 施加更大的梯度。

  同时对正样本（对角线）不做任何修改。

用法:
  from improvements.hard_neg_mining import QuaSIDHardNeg, train_hardneg
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from quasid import QuaSID, compute_hamr_loss
from curriculum_hamr import (evaluate_cosine, compute_entropy,
                             train_quasid_standard)


# ═══════════════════════════════════════════════════════════════════════════════
# QuaSIDHardNeg — Hard Negative Mining 增强版
# ═══════════════════════════════════════════════════════════════════════════════

class QuaSIDHardNeg(QuaSID):
    """QuaSID + Hard Negative Mining 增强对比学习

    额外参数:
        hard_neg_ratio: 选取 hardest 多少比例的负样本 (默认 0.5, 即 top 50%)
        hard_neg_weight: 额外权重倍数 (默认 2.0)
                         1.0 = 不做任何加权 (退化为标准 QuaSID)
                        >1.0 = 放大 hard neg 的梯度
    """

    def __init__(self, hard_neg_ratio=0.5, hard_neg_weight=2.0, **kwargs):
        super().__init__(**kwargs)
        self.hard_neg_ratio = hard_neg_ratio
        self.hard_neg_weight = hard_neg_weight

    def compute_contrastive_loss(self, z, trigger_item_ids, target_item_ids):
        """InfoNCE + hard negative re-weighting"""
        B = z.shape[0] // 2
        z_trigger = z[:B]
        z_target = z[B:]

        z_trigger_norm = F.normalize(z_trigger, dim=-1)
        z_target_norm = F.normalize(z_target, dim=-1)

        sim_matrix = z_trigger_norm @ z_target_norm.T / self.tau  # [B, B]
        labels = torch.arange(B, device=z.device)

        # ── Hard Negative Re-weighting ──
        if self.training and self.hard_neg_ratio > 0 and self.hard_neg_weight != 1.0:
            # 标记正样本位置 (对角线)，排除
            diag_mask = torch.eye(B, device=z.device).bool()

            # 对每一行 (trigger → targets) 找 hardest negatives
            neg_sim = sim_matrix.clone()
            neg_sim[diag_mask] = float('-inf')
            k = max(1, int(B * self.hard_neg_ratio))
            _, hard_idx = neg_sim.topk(k, dim=1)  # [B, k]

            # 放大 hard negative 的 logit
            # 等价于: 在 softmax 分母中让这些位置贡献更大 → 梯度更强
            row_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(-1, k)
            sim_matrix[row_idx, hard_idx] *= self.hard_neg_weight

        loss_forward = F.cross_entropy(sim_matrix, labels)
        loss_backward = F.cross_entropy(sim_matrix.T, labels)
        return (loss_forward + loss_backward) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════════════════════

def train_hardneg(model, pair_loader, device, epochs=200, patience=10,
                  lr=3e-4, weight_decay=1e-5, log_interval=10):
    """训练 QuaSIDHardNeg（与标准 QuaSID 训练完全相同，区别在模型内部）"""
    return train_quasid_standard(model, pair_loader, device,
                                 epochs=epochs, patience=patience,
                                 lr=lr, weight_decay=weight_decay,
                                 log_interval=log_interval)


# ═══════════════════════════════════════════════════════════════════════════════
# 超参数搜索辅助
# ═══════════════════════════════════════════════════════════════════════════════

HARDNEG_CONFIGS = {
    'standard': {
        'desc': '标准 QuaSID (无 hard negative mining, ratio=0)',
        'factory': lambda cfg: QuaSID(**cfg),
    },
    'hard_top30': {
        'desc': 'HardNeg-Top30%: 最难 30% 负样本 ×2 加权',
        'factory': lambda cfg: QuaSIDHardNeg(hard_neg_ratio=0.3, hard_neg_weight=2.0, **cfg),
    },
    'hard_top50': {
        'desc': 'HardNeg-Top50%: 最难 50% 负样本 ×2 加权',
        'factory': lambda cfg: QuaSIDHardNeg(hard_neg_ratio=0.5, hard_neg_weight=2.0, **cfg),
    },
    'hard_top70': {
        'desc': 'HardNeg-Top70%: 最难 70% 负样本 ×2 加权',
        'factory': lambda cfg: QuaSIDHardNeg(hard_neg_ratio=0.7, hard_neg_weight=2.0, **cfg),
    },
    'hard_top50_x3': {
        'desc': 'HardNeg-Top50%-x3: 最难 50% 负样本 ×3 强加权',
        'factory': lambda cfg: QuaSIDHardNeg(hard_neg_ratio=0.5, hard_neg_weight=3.0, **cfg),
    },
}
