"""
数据加载工具
- ItemDataset: 单 item 训练 (Baseline RQ-VAE)
- ContrastivePairDataset: 对比学习对训练 (QuaSID)
- 评估数据构造
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset


class ItemDataset(Dataset):
    """
    单 item 数据集 — 用于 RQ-VAE / ImpVQGAN / SimRQ / Rotation baseline

    每个样本: (item_feat, item_idx)
    其中 item_feat 是文本编码器的输出
    """

    def __init__(self, item_indices, embeddings, item2idx=None):
        """
        Args:
            item_indices: list of item indices (出现过的 item)
            embeddings: dict {item_idx: np.ndarray}
            item2idx: 原始 item id → 连续索引映射
        """
        self.item_indices = []
        self.features = []

        for idx in item_indices:
            if idx in embeddings:
                self.item_indices.append(idx)
                self.features.append(torch.from_numpy(embeddings[idx]).float())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.item_indices[idx]


class ContrastivePairDataset(Dataset):
    """
    对比学习对数据集 — 用于 QuaSID 训练

    对于训练交互 (user, pos_item):
      50%: trigger ← 从 Swing 共现对中随机采样
      50%: trigger ← pos_item 自身
      target ← pos_item

    返回: (trigger_feat, target_feat, trigger_id, target_id)
    """

    def __init__(self, train_df, embeddings, swing_dict, item2idx, user2idx=None):
        """
        Args:
            train_df: 训练交互 DataFrame (user, item, rating, timestamp)
            embeddings: dict {item_idx: np.ndarray}
            swing_dict: dict {item_idx: [(neighbor_idx, score), ...]}
            item2idx: dict {原始item_id → idx}
            user2idx: dict {原始user_id → idx}
        """
        self.embeddings = embeddings
        self.swing_dict = swing_dict
        self.item2idx = item2idx

        # 构建交互列表: [(user_idx, item_idx)]
        self.interactions = []
        for _, row in train_df.iterrows():
            user = row['user']
            item = row['item']
            if item in item2idx and item2idx[item] in embeddings:
                ui = user2idx[user] if user2idx else 0
                self.interactions.append((ui, item2idx[item]))

        print(f'ContrastivePairDataset: {len(self.interactions)} 个训练交互')

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_idx, target_idx = self.interactions[idx]

        # target 特征
        target_feat = torch.from_numpy(self.embeddings[target_idx]).float()

        # trigger: 50% Swing 共现, 50% 自身
        if random.random() < 0.5 and target_idx in self.swing_dict and len(self.swing_dict[target_idx]) > 0:
            # 从 Swing 共现对中随机采样
            neighbors = self.swing_dict[target_idx]
            trigger_idx = random.choice(neighbors)[0]
            # 确保 trigger 有嵌入
            if trigger_idx in self.embeddings:
                trigger_feat = torch.from_numpy(self.embeddings[trigger_idx]).float()
                return trigger_feat, target_feat, torch.tensor(trigger_idx), torch.tensor(target_idx)
            else:
                trigger_feat = target_feat.clone()
                return trigger_feat, target_feat, torch.tensor(target_idx), torch.tensor(target_idx)
        else:
            # trigger = target 自身
            trigger_feat = target_feat.clone()
            return trigger_feat, target_feat, torch.tensor(target_idx), torch.tensor(target_idx)


def collate_fn_single(batch):
    """单 item batch 整理"""
    feats = torch.stack([x[0] for x in batch])
    indices = torch.tensor([x[1] for x in batch])
    return feats, indices


def collate_fn_pair(batch):
    """对比学习对 batch 整理"""
    trigger_feats = torch.stack([x[0] for x in batch])
    target_feats = torch.stack([x[1] for x in batch])
    trigger_ids = torch.stack([x[2] for x in batch])
    target_ids = torch.stack([x[3] for x in batch])
    return trigger_feats, target_feats, trigger_ids, target_ids


def build_eval_data(train_df, test_df, valid_df, embeddings, item2idx, user2idx):
    """
    构建评估数据

    Returns:
        test_users: [{user_idx, train_items, target_item}]
        all_item_emb: [N, d] 所有物品的编码器嵌入（需评估时计算）
    """
    # 构建用户训练历史
    user_train_items = {}
    for _, row in train_df.iterrows():
        u = user2idx[row['user']]
        i = item2idx[row['item']]
        if u not in user_train_items:
            user_train_items[u] = []
        user_train_items[u].append(i)

    # 构建测试用户（有效交互 + 可嵌入的物品）
    test_users = []
    for _, row in test_df.iterrows():
        u_orig = row['user']
        i_orig = row['item']
        if u_orig not in user2idx or i_orig not in item2idx:
            continue
        u = user2idx[u_orig]
        i = item2idx[i_orig]
        if i not in embeddings:
            continue
        if u not in user_train_items:
            continue
        test_users.append({
            'user_idx': u,
            'target_item': i,
            'train_items': user_train_items[u],
        })

    # 所有可嵌入物品的索引列表
    all_items = sorted(embeddings.keys())

    return test_users, all_items
