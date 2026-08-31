"""
Curriculum HaMR — 课程式渐进 margin 改进
==========================================

动机:
  标准 QuaSID 从第 1 个 epoch 就用完整 margin (m_full=0.8, m_partial=0.5)。
  但训练初期编码器输出还是随机的，强行用大 margin 分离冲突对会导致:
    1. 早期梯度震荡大
    2. 模型在学到有意义的语义前就陷入局部最优
    3. 对随机种子敏感

改进:
  训练早期用小 margin，让模型先学好重建和基本语义结构，
  随 epoch 逐步增加 margin 到目标值。

三种 schedule:
  - linear:   m(t) = m_target * (start_ratio + (1-start_ratio) * t/warmup)
  - cosine:   m(t) = m_target * (start_ratio + (1-start_ratio) * (1-cos(π*t/warmup))/2)
  - r_anneal: R 从 2 退火到 1，margin 用 linear (先宽松，后精细)

用法:
  from improvements.curriculum_hamr import QuaSIDCurriculum, train_curriculum
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from quasid import QuaSID, compute_hamr_loss


# ═══════════════════════════════════════════════════════════════════════════════
# QuaSIDCurriculum — 支持渐进式 margin schedule
# ═══════════════════════════════════════════════════════════════════════════════

class QuaSIDCurriculum(QuaSID):
    """QuaSID + 课程式渐进 HaMR margin

    额外参数:
        warmup_epochs:        warmup 持续 epoch 数 (默认 20)
        schedule_mode:        'linear' | 'cosine' | 'r_anneal'
        margin_start_ratio:   起始 margin 比例 (默认 0.1, 即从 10% 开始)
    """

    def __init__(self, warmup_epochs=20, schedule_mode='linear',
                 margin_start_ratio=0.1, **kwargs):
        super().__init__(**kwargs)

        self.warmup_epochs = warmup_epochs
        self.schedule_mode = schedule_mode
        self.margin_start_ratio = margin_start_ratio

        # 保存目标值
        self.m_full_target = self.m_full
        self.m_partial_target = self.m_partial
        self.R_target = self.R
        self.lambda_full_target = self.lambda_full
        self.lambda_partial_target = self.lambda_partial

        self.current_epoch = 0

    def set_epoch(self, epoch):
        """每个 epoch 开始前调用，更新当前 margin / lambda / R"""
        self.current_epoch = epoch
        t = min(1.0, epoch / max(1, self.warmup_epochs))

        if self.schedule_mode == 'linear':
            s = self.margin_start_ratio + (1.0 - self.margin_start_ratio) * t
        elif self.schedule_mode == 'cosine':
            s = self.margin_start_ratio + (1.0 - self.margin_start_ratio) * \
                (1.0 - np.cos(np.pi * t)) / 2.0
        elif self.schedule_mode == 'r_anneal':
            s = self.margin_start_ratio + (1.0 - self.margin_start_ratio) * t
            self.R = max(self.R_target, int(np.ceil(self.R_target + 1 - t * 1.0)))
        else:
            s = 1.0

        self.m_full = self.m_full_target * s
        self.m_partial = self.m_partial_target * s
        self.lambda_full = self.lambda_full_target * s
        self.lambda_partial = self.lambda_partial_target * s


# ═══════════════════════════════════════════════════════════════════════════════
# 训练函数
# ═══════════════════════════════════════════════════════════════════════════════

def train_curriculum(model, pair_loader, device, epochs=200, patience=10,
                     lr=3e-4, weight_decay=1e-5, log_interval=10):
    """训练 QuaSIDCurriculum 模型

    与标准 QuaSID 训练的唯一区别: 每个 epoch 前调用 model.set_epoch(epoch)
    来更新当前的 margin / lambda / R 值。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = defaultdict(list)

    for epoch in range(epochs):
        # ★ 关键: 更新课程进度
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)

        epoch_losses = defaultdict(list)

        for trigger_feat, target_feat, trigger_id, target_id in pair_loader:
            trigger_feat = trigger_feat.to(device)
            target_feat = target_feat.to(device)
            trigger_id = trigger_id.to(device)
            target_id = target_id.to(device)

            feats = torch.cat([trigger_feat, target_feat], dim=0)
            out, quant_loss, codes, z, z_q = model(feats)

            loss_dict = model.compute_loss(
                out, quant_loss, codes, z,
                trigger_feat, target_feat, trigger_id, target_id,
            )

            optimizer.zero_grad()
            loss_dict['loss_total'].backward()
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k].append(v.item())

        avg_loss = np.mean(epoch_losses['loss_total'])
        for k in epoch_losses:
            history[k].append(np.mean(epoch_losses[k]))

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % log_interval == 0:
            mf = getattr(model, 'm_full', '?')
            mp = getattr(model, 'm_partial', '?')
            Rv = getattr(model, 'R', '?')
            print(f'  Epoch {epoch+1:3d}: loss={avg_loss:.6f}  '
                  f'm_full={mf:.3f}  m_partial={mp:.3f}  R={Rv}  '
                  f'patience={patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'  早停 @ epoch {epoch+1}')
            break

    model.load_state_dict(best_state)
    return model, dict(history)


def train_quasid_standard(model, pair_loader, device, epochs=200, patience=10,
                          lr=3e-4, weight_decay=1e-5, log_interval=10):
    """训练标准 QuaSID（无 curriculum，作为对照）"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = defaultdict(list)

    for epoch in range(epochs):
        epoch_losses = defaultdict(list)

        for trigger_feat, target_feat, trigger_id, target_id in pair_loader:
            trigger_feat = trigger_feat.to(device)
            target_feat = target_feat.to(device)
            trigger_id = trigger_id.to(device)
            target_id = target_id.to(device)

            feats = torch.cat([trigger_feat, target_feat], dim=0)
            out, quant_loss, codes, z, z_q = model(feats)

            loss_dict = model.compute_loss(
                out, quant_loss, codes, z,
                trigger_feat, target_feat, trigger_id, target_id,
            )

            optimizer.zero_grad()
            loss_dict['loss_total'].backward()
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k].append(v.item())

        avg_loss = np.mean(epoch_losses['loss_total'])
        for k in epoch_losses:
            history[k].append(np.mean(epoch_losses[k]))

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % log_interval == 0:
            print(f'  Epoch {epoch+1:3d}: loss={avg_loss:.6f}  '
                  f'patience={patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'  早停 @ epoch {epoch+1}')
            break

    model.load_state_dict(best_state)
    return model, dict(history)


# ═══════════════════════════════════════════════════════════════════════════════
# 快速评估 (余弦相似度, 不需 TIGER)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_cosine(model, embeddings, test_users, all_items, device,
                    k_list=(1, 5, 10, 20)):
    """余弦相似度快速评估"""
    item_to_idx = {item: i for i, item in enumerate(all_items)}

    # 计算所有物品编码器嵌入
    all_emb = []
    bs = 256
    for start in range(0, len(all_items), bs):
        batch_items = all_items[start:start + bs]
        feats = torch.stack([torch.from_numpy(embeddings[i]).float()
                            for i in batch_items]).to(device)
        z = model.get_encoder_emb(feats)
        all_emb.append(z.cpu())
    all_emb = F.normalize(torch.cat(all_emb, dim=0), dim=-1)

    metrics = {k: {'HR': [], 'NDCG': []} for k in k_list}

    for user_data in test_users:
        target = user_data['target_item']
        train_items = user_data['train_items']
        if target not in item_to_idx:
            continue

        target_idx = item_to_idx[target]
        train_embs = [all_emb[item_to_idx[ti]] for ti in train_items
                      if ti in item_to_idx]
        if not train_embs:
            continue

        user_emb = F.normalize(torch.stack(train_embs).mean(dim=0, keepdim=True), dim=-1)
        scores = (user_emb @ all_emb.T).squeeze(0)

        interacted = {item_to_idx[ti] for ti in train_items if ti in item_to_idx}
        scores[list(interacted)] = -float('inf')

        _, top_indices = scores.topk(max(k_list))
        top_indices = top_indices.cpu().numpy()

        for k in k_list:
            top_k = top_indices[:k]
            hit = int(target_idx in top_k)
            metrics[k]['HR'].append(hit)
            if hit:
                rank = np.where(top_k == target_idx)[0][0] + 1
                metrics[k]['NDCG'].append(1.0 / np.log2(rank + 1))
            else:
                metrics[k]['NDCG'].append(0.0)

    return {f'HR@{k}': np.mean(metrics[k]['HR']) for k in k_list} | \
           {f'NDCG@{k}': np.mean(metrics[k]['NDCG']) for k in k_list}


@torch.no_grad()
def compute_entropy(model, embeddings, all_items, device, batch_size=256):
    """计算 SID 组合熵"""
    model.eval()
    all_codes = []
    for start in range(0, len(all_items), batch_size):
        batch_items = all_items[start:start + batch_size]
        feats = torch.stack([torch.from_numpy(embeddings[i]).float()
                            for i in batch_items]).to(device)
        codes = model.get_codes(feats).squeeze(1).squeeze(1).cpu()
        all_codes.append(codes)

    all_codes = torch.cat(all_codes, dim=0)
    L = all_codes.shape[1]
    K = model.n_embed
    multipliers = torch.tensor([K ** (L - 1 - l) for l in range(L)])
    sid_int = (all_codes * multipliers).sum(dim=1)

    unique, counts = sid_int.unique(return_counts=True)
    probs = counts.float() / len(sid_int)
    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
    usage_rate = len(unique) / (K ** L)
    return entropy, usage_rate, len(unique)
