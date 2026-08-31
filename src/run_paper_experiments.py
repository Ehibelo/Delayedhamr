"""
实验运行器 — 复现 Table 2, Table 3, Table 5

使用方法:
  python run_paper_experiments.py --data_dir ../data --category Beauty --tables 2,3,5
"""

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rqvae import RQVAE
from quasid import QuaSID, ImprovedVQGAN, SimRQ, RQVAERotation
from data_utils import (
    ItemDataset, ContrastivePairDataset,
    collate_fn_single, collate_fn_pair,
    build_eval_data,
)
from tiger import run_tiger_evaluation

warnings.filterwarnings('ignore')

# ==============================================================================
# 评估指标
# ==============================================================================


@torch.no_grad()
def compute_encoder_embeddings(model, embeddings, all_items, device, batch_size=256):
    """计算所有物品的编码器嵌入"""
    model.eval()
    all_emb = []
    for start in tqdm(range(0, len(all_items), batch_size), desc='计算编码器嵌入', leave=False):
        batch_items = all_items[start:start + batch_size]
        feats = torch.stack([torch.from_numpy(embeddings[i]).float() for i in batch_items]).to(device)
        z = model.get_encoder_emb(feats)  # [B, d]
        all_emb.append(z.cpu())
    all_emb = torch.cat(all_emb, dim=0)  # [N, d]
    return F.normalize(all_emb, dim=-1)


@torch.no_grad()
def evaluate_cosine(model, embeddings, test_users, all_items, device, k_list=[1, 5, 10, 20]):
    """
    余弦相似度评估

    1. 所有 item → encoder → all_item_emb
    2. 每个用户: user_emb = mean(训练历史 item 的编码器嵌入)
    3. 与所有物品计算余弦相似度
    4. 排除已交互 item → 排序 → HR@K, NDCG@K
    """
    item_to_idx = {item: i for i, item in enumerate(all_items)}
    all_item_emb = compute_encoder_embeddings(model, embeddings, all_items, device)

    metrics = {k: defaultdict(list) for k in k_list}

    for user_data in tqdm(test_users, desc='评估', leave=False):
        target = user_data['target_item']
        train_items = user_data['train_items']

        if target not in item_to_idx:
            continue

        target_idx = item_to_idx[target]

        # 用户嵌入 = 训练历史物品编码器嵌入的均值
        train_embs = []
        for ti in train_items:
            if ti in item_to_idx:
                train_embs.append(all_item_emb[item_to_idx[ti]])
        if not train_embs:
            continue
        user_emb = torch.stack(train_embs).mean(dim=0, keepdim=True)
        user_emb = F.normalize(user_emb, dim=-1)

        # 余弦相似度
        scores = (user_emb @ all_item_emb.T).squeeze(0)  # [N]

        # 排除已交互 item
        interacted = set(item_to_idx[ti] for ti in train_items if ti in item_to_idx)
        scores[list(interacted)] = -float('inf')

        # 排序
        _, top_indices = scores.topk(max(k_list))

        for k in k_list:
            top_k = top_indices[:k].cpu().numpy()
            hit = int(target_idx in top_k)
            metrics[k]['HR'].append(hit)

            if hit:
                rank = np.where(top_k == target_idx)[0][0] + 1
                ndcg = 1.0 / np.log2(rank + 1)
            else:
                ndcg = 0.0
            metrics[k]['NDCG'].append(ndcg)

    results = {}
    for k in k_list:
        results[f'HR@{k}'] = np.mean(metrics[k]['HR'])
        results[f'NDCG@{k}'] = np.mean(metrics[k]['NDCG'])

    return results


@torch.no_grad()
def compute_entropy(model, embeddings, all_items, device, batch_size=256):
    """
    计算 SID 组合熵

    Entropy = -Σ p(s) log p(s)
    其中 p(s) 是 SID 组合 s 在数据集中出现的频率

    理论最大值: log(K^L)
    """
    model.eval()
    all_codes = []

    for start in tqdm(range(0, len(all_items), batch_size), desc='计算 SID', leave=False):
        batch_items = all_items[start:start + batch_size]
        feats = torch.stack([torch.from_numpy(embeddings[i]).float() for i in batch_items]).to(device)
        codes = model.get_codes(feats)  # [B, 1, 1, L]
        codes = codes.squeeze(1).squeeze(1).cpu()  # [B, L]
        all_codes.append(codes)

    all_codes = torch.cat(all_codes, dim=0)  # [N, L]

    # 将每个 SID 组合编码为唯一整数
    L = all_codes.shape[1]
    K = model.n_embed
    multipliers = torch.tensor([K ** (L - 1 - l) for l in range(L)])
    sid_int = (all_codes * multipliers).sum(dim=1)  # [N]

    # 统计频率
    unique, counts = sid_int.unique(return_counts=True)
    probs = counts.float() / len(sid_int)
    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()

    # 使用率
    n_possible = K ** L
    n_used = len(unique)
    usage_rate = n_used / n_possible

    return entropy, usage_rate, n_used


# ==============================================================================
# 训练函数
# ==============================================================================


def train_rqvae_baseline(model, dataloader, optimizer, device, model_type='rqvae',
                         epochs=200, patience=10, embeddings=None, swing_dict=None):
    """训练 RQ-VAE baseline（仅重建损失）"""
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'loss_total': [], 'loss_recon': [], 'loss_latent': []}

    for epoch in range(epochs):
        epoch_losses = defaultdict(list)

        for feats, indices in dataloader:
            feats = feats.to(device)
            out, quant_loss, codes, z, z_q = model(feats)

            # 根据模型类型调用不同的 compute_loss
            if model_type == 'impvqgan':
                loss_dict = model.compute_loss(out, quant_loss, feats, z, z_q)
            elif model_type == 'simrq':
                # 构建 batch 内的 Swing 相似度矩阵
                swing_sim_batch = None
                if swing_dict is not None:
                    B = len(indices)
                    swing_sim_batch = torch.zeros(B, B, device=device)
                    # Pre-compute neighbor dicts outside the inner loop
                    neighbor_dicts = {}
                    for i in range(B):
                        item_i = indices[i].item()
                        neighbor_dicts[i] = dict(swing_dict.get(item_i, []))
                    for i in range(B):
                        ndict_i = neighbor_dicts[i]
                        for j in range(B):
                            if i != j:
                                item_j = indices[j].item()
                                if item_j in ndict_i:
                                    swing_sim_batch[i, j] = ndict_i[item_j]
                loss_dict = model.compute_loss(out, quant_loss, feats, z, swing_sim_batch=swing_sim_batch)
            else:
                loss_dict = model.compute_loss(out, quant_loss, feats)

            optimizer.zero_grad()
            loss_dict['loss_total'].backward()
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k].append(v.item())

        avg_loss = np.mean(epoch_losses['loss_total'])
        for k in history:
            history[k].append(np.mean(epoch_losses.get(k, [0])))

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model, history


def train_quasid(model, dataloader, optimizer, device, epochs=200, patience=10):
    """训练 QuaSID 完整模型"""
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'loss_total': [], 'loss_recon': [], 'loss_latent': [], 'loss_cl': [], 'loss_hamr': []}

    for epoch in range(epochs):
        epoch_losses = defaultdict(list)

        for trigger_feat, target_feat, trigger_id, target_id in dataloader:
            trigger_feat = trigger_feat.to(device)
            target_feat = target_feat.to(device)
            trigger_id = trigger_id.to(device)
            target_id = target_id.to(device)

            # 拼接通过 encoder + quantizer
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
        for k in history:
            history[k].append(np.mean(epoch_losses.get(k, [0])))

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model, history


def train_quasid_hamr_only(model, dataloader, optimizer, device, epochs=200, patience=10):
    """
    训练 RQ-VAE + HaMR（无对比学习，λ_cl=0）
    用于 Table 3: 预训练 RQ-VAE → 加 HaMR 微调
    """
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {'loss_total': [], 'loss_recon': [], 'loss_latent': [], 'loss_hamr': []}

    for epoch in range(epochs):
        epoch_losses = defaultdict(list)

        for trigger_feat, target_feat, trigger_id, target_id in dataloader:
            trigger_feat = trigger_feat.to(device)
            target_feat = target_feat.to(device)
            trigger_id = trigger_id.to(device)
            target_id = target_id.to(device)

            feats = torch.cat([trigger_feat, target_feat], dim=0)
            out, quant_loss, codes, z, z_q = model(feats)

            # 仅用重建 + HaMR（λ_cl=0）
            loss_recon = F.mse_loss(
                out,
                torch.cat([trigger_feat, target_feat], dim=0),
            )
            loss_total = loss_recon + model.beta * quant_loss

            # HaMR
            from quasid import compute_hamr_loss
            item_ids = torch.cat([trigger_id, target_id], dim=0)
            loss_hamr = compute_hamr_loss(
                codes, z, item_ids,
                m_full=model.m_full,
                m_partial=model.m_partial,
                R=model.R,
                lambda_full=model.lambda_full,
                lambda_partial=model.lambda_partial,
                use_cvpm=model.use_cvpm,
                trigger_item_ids=trigger_id,
                target_item_ids=target_id,
            )
            loss_total = loss_total + loss_hamr

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            epoch_losses['loss_total'].append(loss_total.item())
            epoch_losses['loss_recon'].append(loss_recon.item())
            epoch_losses['loss_latent'].append(quant_loss.item())
            epoch_losses['loss_hamr'].append(loss_hamr.item())

        avg_loss = np.mean(epoch_losses['loss_total'])
        for k in history:
            history[k].append(np.mean(epoch_losses.get(k, [0])))

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model, history


# ==============================================================================
# 实验运行
# ==============================================================================


def get_model_config(embed_dim):
    """获取模型配置 — 所有超参数与论文严格一致"""
    return {
        'input_dim': embed_dim,       # 文本特征维度 (768, 论文用T5-XXL)
        'hidden_dim': 512,             # 隐藏层维度 [512] 单层
        'latent_dim': 32,              # 潜在空间维度 d (TIGER paper)
        'n_embed': 256,               # codebook 大小 K
        'n_codebook': 3,              # RQ 层数 L
        'decay': 0.99,                # EMA 衰减率
        'beta': 0.25,                 # commitment loss 权重
        'dropout': 0.1,               # Dropout 比率
        'restart_unused_codes': True,  # 自动死码重启
    }


def run_single_experiment(model_type, model, dataloader, optimizer, device, config,
                          embeddings=None, swing_dict=None):
    """运行单次训练实验"""
    if model_type == 'quasid':
        return train_quasid(model, dataloader, optimizer, device,
                           epochs=config.get('max_epochs', 200),
                           patience=config.get('patience', 10))
    elif model_type == 'quasid_hamr_only':
        return train_quasid_hamr_only(model, dataloader, optimizer, device,
                                       epochs=config.get('max_epochs', 200),
                                       patience=config.get('patience', 10))
    else:
        return train_rqvae_baseline(model, dataloader, optimizer, device,
                                     model_type=model_type,
                                     epochs=config.get('max_epochs', 200),
                                     patience=config.get('patience', 10),
                                     embeddings=embeddings,
                                     swing_dict=swing_dict)


def run_table2(data_dir, device, seeds, config, use_tiger=False):
    """
    Table 2 — Baseline 对比

    5 个模型 × 5 seeds:
    - RQ-VAE (baseline)
    - Improved VQGAN
    - SimRQ
    - RQ-VAE-Rotation
    - QuaSID (full)
    """
    eval_method = 'TIGER' if use_tiger else 'Cosine'
    print('\n' + '='*80)
    print(f'Table 2: Baseline 对比 (评估方法: {eval_method})')
    print('='*80)

    data = torch.load(os.path.join(data_dir, f'amazon_Beauty_processed_5core.pt'), map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    embed_dim = data['embed_dim']
    all_items = sorted(embeddings.keys())
    model_config = get_model_config(embed_dim)

    # 单 item 数据集
    single_dataset = ItemDataset(all_items, embeddings)
    single_loader = DataLoader(single_dataset, batch_size=config['batch_size'], shuffle=True,
                               collate_fn=collate_fn_single)

    # 对比学习对数据集
    pair_dataset = ContrastivePairDataset(
        data['train_df'], embeddings, data['swing'],
        data['item2idx'], data['user2idx'],
    )
    pair_loader = DataLoader(pair_dataset, batch_size=config['batch_size'], shuffle=True,
                             collate_fn=collate_fn_pair)

    # 评估数据
    test_users, _ = build_eval_data(
        data['train_df'], data['test_df'], data['valid_df'],
        embeddings, data['item2idx'], data['user2idx'],
    )
    print(f'  测试用户数: {len(test_users)}')

    models_config = [
        ('RQ-VAE', 'rqvae', single_loader),
        ('Improved VQGAN', 'impvqgan', single_loader),
        ('SimRQ', 'simrq', single_loader),
        ('RQ-VAE-Rotation', 'rotation', single_loader),
        ('QuaSID', 'quasid', pair_loader),
    ]

    all_results = {}

    for model_name, model_type, loader in models_config:
        print(f'\n--- {model_name} ---')
        model_results = {'HR@5': [], 'HR@10': [], 'NDCG@5': [], 'NDCG@10': [], 'Entropy': []}

        for seed in seeds:
            print(f'  Seed={seed}')
            set_seed(seed)

            # 创建模型
            if model_type == 'rqvae':
                model = RQVAE(**model_config)
            elif model_type == 'impvqgan':
                model = ImprovedVQGAN(**model_config)
            elif model_type == 'simrq':
                model = SimRQ(**model_config)
            elif model_type == 'rotation':
                model = RQVAERotation(**model_config)
            elif model_type == 'quasid':
                model = QuaSID(**model_config)

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                         weight_decay=config['weight_decay'])

            model, history = run_single_experiment(model_type, model, loader, optimizer, device, config,
                                                      embeddings=embeddings, swing_dict=data['swing'])

            # 评估
            if use_tiger:
                # TIGER 生成式检索评估
                tiger_results, _ = run_tiger_evaluation(
                    model, embeddings, data['train_df'], data['test_df'], data['valid_df'],
                    data['item2idx'], data['user2idx'], device,
                    batch_size=config['batch_size'],
                    lr=config['lr'],
                    weight_decay=config['weight_decay'],
                    epochs=config.get('tiger_epochs', 200),
                    patience=config.get('patience', 10),
                    beam_size=config.get('beam_size', 100),
                )
                eval_results = {
                    'HR@5': tiger_results['HR@5'],
                    'HR@10': tiger_results['HR@10'],
                    'NDCG@5': tiger_results['NDCG@5'],
                    'NDCG@10': tiger_results['NDCG@10'],
                }
            else:
                eval_results = evaluate_cosine(model, embeddings, test_users, all_items, device)
            entropy, usage, n_used = compute_entropy(model, embeddings, all_items, device)

            model_results['HR@5'].append(eval_results['HR@5'])
            model_results['HR@10'].append(eval_results['HR@10'])
            model_results['NDCG@5'].append(eval_results['NDCG@5'])
            model_results['NDCG@10'].append(eval_results['NDCG@10'])
            model_results['Entropy'].append(entropy)

            print(f'    HR@5={eval_results["HR@5"]:.4f}, HR@10={eval_results["HR@10"]:.4f}, '
                  f'NDCG@5={eval_results["NDCG@5"]:.4f}, Entropy={entropy:.4f}')

            # 释放 GPU 显存
            del model
            torch.cuda.empty_cache()

        avg_results = {k: (np.mean(v), np.std(v)) for k, v in model_results.items()}
        all_results[model_name] = avg_results

        print(f'  {model_name} 均值: HR@5={avg_results["HR@5"][0]:.4f}±{avg_results["HR@5"][1]:.4f}, '
              f'NDCG@5={avg_results["NDCG@5"][0]:.4f}±{avg_results["NDCG@5"][1]:.4f}')

    return all_results


def run_table3(data_dir, device, seeds, config, use_tiger=False):
    """
    Table 3 — HaMR 即插即用

    2 个模型 × 5 seeds:
    - RQ-VAE + HaMR (预训练 RQ-VAE → 加载 → 加 HaMR 微调)
    - ImpVQGAN + HaMR
    """
    eval_method = 'TIGER' if use_tiger else 'Cosine'
    print('\n' + '='*80)
    print(f'Table 3: HaMR 即插即用 (评估方法: {eval_method})')
    print('='*80)

    data = torch.load(os.path.join(data_dir, f'amazon_Beauty_processed_5core.pt'), map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    embed_dim = data['embed_dim']
    all_items = sorted(embeddings.keys())
    model_config = get_model_config(embed_dim)

    pair_dataset = ContrastivePairDataset(
        data['train_df'], embeddings, data['swing'],
        data['item2idx'], data['user2idx'],
    )
    pair_loader = DataLoader(pair_dataset, batch_size=config['batch_size'], shuffle=True,
                             collate_fn=collate_fn_pair)

    test_users, _ = build_eval_data(
        data['train_df'], data['test_df'], data['valid_df'],
        embeddings, data['item2idx'], data['user2idx'],
    )

    models_config = [
        ('RQ-VAE+HaMR', 'rqvae'),
        ('ImpVQGAN+HaMR', 'impvqgan'),
    ]

    all_results = {}

    for model_name, base_type in models_config:
        print(f'\n--- {model_name} ---')
        model_results = {'HR@5': [], 'HR@10': [], 'NDCG@5': [], 'NDCG@10': [], 'Entropy': []}

        for seed in seeds:
            print(f'  Seed={seed}')
            set_seed(seed)

            # 阶段 1: 预训练 RQ-VAE / ImpVQGAN baseline
            if base_type == 'rqvae':
                pretrain_model = RQVAE(**model_config)
            else:
                pretrain_model = ImprovedVQGAN(**model_config)

            pretrain_model = pretrain_model.to(device)
            single_dataset = ItemDataset(all_items, embeddings)
            single_loader = DataLoader(single_dataset, batch_size=config['batch_size'], shuffle=True,
                                       collate_fn=collate_fn_single)
            opt1 = torch.optim.Adam(pretrain_model.parameters(), lr=config['lr'],
                                    weight_decay=config['weight_decay'])
            pretrain_model, _ = train_rqvae_baseline(pretrain_model, single_loader, opt1, device,
                                                      model_type=base_type,
                                                      epochs=config['max_epochs'], patience=config['patience'])

            # 阶段 2: 加载预训练权重到 QuaSID，λ_cl=0，加 HaMR 微调
            quasid_model = QuaSID(**model_config, lambda_cl=0.0,
                                  pre_quant_mode='l2_norm' if base_type == 'impvqgan' else 'identity',
                                  normalize_codebook=(base_type == 'impvqgan'))
            quasid_model = quasid_model.to(device)

            # 加载编码器、量化器、解码器权重
            pretrain_state = pretrain_model.state_dict()
            quasid_state = quasid_model.state_dict()
            for k in pretrain_state:
                if k in quasid_state:
                    quasid_state[k] = pretrain_state[k]
            quasid_model.load_state_dict(quasid_state)

            opt2 = torch.optim.Adam(quasid_model.parameters(), lr=config['lr'],
                                    weight_decay=config['weight_decay'])
            quasid_model, history = train_quasid_hamr_only(quasid_model, pair_loader, opt2, device,
                                                            epochs=config['max_epochs'], patience=config['patience'])

            # 评估
            if use_tiger:
                tiger_results, _ = run_tiger_evaluation(
                    quasid_model, embeddings, data['train_df'], data['test_df'], data['valid_df'],
                    data['item2idx'], data['user2idx'], device,
                    batch_size=config['batch_size'],
                    lr=config['lr'],
                    weight_decay=config['weight_decay'],
                    epochs=config.get('tiger_epochs', 200),
                    patience=config.get('patience', 10),
                    beam_size=config.get('beam_size', 100),
                )
                eval_results = {
                    'HR@5': tiger_results['HR@5'],
                    'HR@10': tiger_results['HR@10'],
                    'NDCG@5': tiger_results['NDCG@5'],
                    'NDCG@10': tiger_results['NDCG@10'],
                }
            else:
                eval_results = evaluate_cosine(quasid_model, embeddings, test_users, all_items, device)
            entropy, usage, n_used = compute_entropy(quasid_model, embeddings, all_items, device)

            model_results['HR@5'].append(eval_results['HR@5'])
            model_results['HR@10'].append(eval_results['HR@10'])
            model_results['NDCG@5'].append(eval_results['NDCG@5'])
            model_results['NDCG@10'].append(eval_results['NDCG@10'])
            model_results['Entropy'].append(entropy)

            print(f'    HR@5={eval_results["HR@5"]:.4f}, HR@10={eval_results["HR@10"]:.4f}, '
                  f'NDCG@5={eval_results["NDCG@5"]:.4f}, Entropy={entropy:.4f}')

            del pretrain_model, quasid_model
            torch.cuda.empty_cache()

        avg_results = {k: (np.mean(v), np.std(v)) for k, v in model_results.items()}
        all_results[model_name] = avg_results

        print(f'  {model_name} 均值: HR@5={avg_results["HR@5"][0]:.4f}±{avg_results["HR@5"][1]:.4f}, '
              f'NDCG@5={avg_results["NDCG@5"][0]:.4f}±{avg_results["NDCG@5"][1]:.4f}')

    return all_results


def run_table5(data_dir, device, seeds, config, use_tiger=False):
    """
    Table 5 — 消融实验

    2 个变体 × 5 seeds:
    - QuaSID w/o CVPM (use_hamr=True, use_cvpm=False)
    - QuaSID w/o HaMR (use_hamr=False, → 仅对比学习)
    """
    eval_method = 'TIGER' if use_tiger else 'Cosine'
    print('\n' + '='*80)
    print(f'Table 5: 消融实验 (评估方法: {eval_method})')
    print('='*80)

    data = torch.load(os.path.join(data_dir, f'amazon_Beauty_processed_5core.pt'), map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    embed_dim = data['embed_dim']
    all_items = sorted(embeddings.keys())
    model_config = get_model_config(embed_dim)

    pair_dataset = ContrastivePairDataset(
        data['train_df'], embeddings, data['swing'],
        data['item2idx'], data['user2idx'],
    )
    pair_loader = DataLoader(pair_dataset, batch_size=config['batch_size'], shuffle=True,
                             collate_fn=collate_fn_pair)

    test_users, _ = build_eval_data(
        data['train_df'], data['test_df'], data['valid_df'],
        embeddings, data['item2idx'], data['user2idx'],
    )

    # w/o HaMR: 需要简化版模型（仅对比学习，无 HaMR）
    # w/o CVPM: QuaSID with use_cvpm=False

    variants = [
        ('w/o CVPM', {'use_cvpm': False, 'use_hamr': True}),
        ('w/o HaMR', {'use_cvpm': True, 'use_hamr': False}),
    ]

    all_results = {}

    for var_name, var_config in variants:
        print(f'\n--- {var_name} ---')
        model_results = {'HR@5': [], 'HR@10': [], 'NDCG@5': [], 'NDCG@10': [], 'Entropy': []}

        for seed in seeds:
            print(f'  Seed={seed}')
            set_seed(seed)

            if var_config['use_hamr']:
                model = QuaSID(**model_config, use_cvpm=var_config['use_cvpm'])
            else:
                # w/o HaMR: λ_full=0, λ_partial=0, 仅对比学习 + 重建
                model = QuaSID(**model_config,
                               use_cvpm=False,
                               lambda_full=0.0,
                               lambda_partial=0.0)

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                         weight_decay=config['weight_decay'])

            model, history = train_quasid(model, pair_loader, optimizer, device,
                                          epochs=config['max_epochs'], patience=config['patience'])

            if use_tiger:
                tiger_results, _ = run_tiger_evaluation(
                    model, embeddings, data['train_df'], data['test_df'], data['valid_df'],
                    data['item2idx'], data['user2idx'], device,
                    batch_size=config['batch_size'],
                    lr=config['lr'],
                    weight_decay=config['weight_decay'],
                    epochs=config.get('tiger_epochs', 200),
                    patience=config.get('patience', 10),
                    beam_size=config.get('beam_size', 100),
                )
                eval_results = {
                    'HR@5': tiger_results['HR@5'],
                    'HR@10': tiger_results['HR@10'],
                    'NDCG@5': tiger_results['NDCG@5'],
                    'NDCG@10': tiger_results['NDCG@10'],
                }
            else:
                eval_results = evaluate_cosine(model, embeddings, test_users, all_items, device)
            entropy, usage, n_used = compute_entropy(model, embeddings, all_items, device)

            model_results['HR@5'].append(eval_results['HR@5'])
            model_results['HR@10'].append(eval_results['HR@10'])
            model_results['NDCG@5'].append(eval_results['NDCG@5'])
            model_results['NDCG@10'].append(eval_results['NDCG@10'])
            model_results['Entropy'].append(entropy)

            print(f'    HR@5={eval_results["HR@5"]:.4f}, HR@10={eval_results["HR@10"]:.4f}, '
                  f'NDCG@5={eval_results["NDCG@5"]:.4f}, Entropy={entropy:.4f}')

            del model
            torch.cuda.empty_cache()

        avg_results = {k: (np.mean(v), np.std(v)) for k, v in model_results.items()}
        all_results[var_name] = avg_results

        print(f'  {var_name} 均值: HR@5={avg_results["HR@5"][0]:.4f}±{avg_results["HR@5"][1]:.4f}, '
              f'NDCG@5={avg_results["NDCG@5"][0]:.4f}±{avg_results["NDCG@5"][1]:.4f}')

    return all_results


# ==============================================================================
# 工具函数
# ==============================================================================


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_results_table(results, title):
    """打印结果表格"""
    print(f'\n{title}')
    print('-' * 80)
    print(f'{"Model":<25s} {"HR@5":>10s} {"HR@10":>10s} {"NDCG@5":>10s} {"NDCG@10":>10s} {"Entropy":>10s}')
    print('-' * 80)
    for name, metrics in results.items():
        hr5 = f'{metrics["HR@5"][0]:.4f}±{metrics["HR@5"][1]:.4f}'
        hr10 = f'{metrics["HR@10"][0]:.4f}±{metrics["HR@10"][1]:.4f}'
        n5 = f'{metrics["NDCG@5"][0]:.4f}±{metrics["NDCG@5"][1]:.4f}'
        n10 = f'{metrics["NDCG@10"][0]:.4f}±{metrics["NDCG@10"][1]:.4f}'
        ent = f'{metrics["Entropy"][0]:.2f}±{metrics["Entropy"][1]:.2f}'
        print(f'{name:<25s} {hr5:>10s} {hr10:>10s} {n5:>10s} {n10:>10s} {ent:>10s}')
    print('-' * 80)


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--category', type=str, default='Beauty')
    parser.add_argument('--tables', type=str, default='2,3,5',
                        help='要运行的表格，逗号分隔: 2,3,5')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1024',
                        help='随机种子，逗号分隔')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_tiger', action='store_true', default=False,
                        help='使用 TIGER 生成式检索评估 (替代余弦相似度)')
    parser.add_argument('--beam_size', type=int, default=100,
                        help='TIGER beam search 大小')
    parser.add_argument('--tiger_epochs', type=int, default=200,
                        help='TIGER 训练 epoch 数')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'GPU: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else "N/A"}')
    print(f'VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB')

    tables = [int(t.strip()) for t in args.tables.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'beam_size': args.beam_size,
        'tiger_epochs': args.tiger_epochs,
    }

    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join('../results', f'paper_strict_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)

    # 保存实验配置
    with open(os.path.join(result_dir, 'config.json'), 'w') as f:
        json.dump({
            'tables': tables,
            'seeds': seeds,
            'config': config,
            'data_dir': args.data_dir,
            'category': args.category,
            'device': str(device),
        }, f, indent=2)

    all_results = {}

    if 2 in tables:
        results = run_table2(args.data_dir, device, seeds, config, use_tiger=args.use_tiger)
        all_results['table2'] = results
        print_results_table(results, 'Table 2 结果')
        with open(os.path.join(result_dir, 'table2.json'), 'w') as f:
            json.dump({k: {mk: [mv[0], mv[1]] for mk, mv in v.items()} for k, v in results.items()}, f, indent=2)

    if 3 in tables:
        results = run_table3(args.data_dir, device, seeds, config, use_tiger=args.use_tiger)
        all_results['table3'] = results
        print_results_table(results, 'Table 3 结果')
        with open(os.path.join(result_dir, 'table3.json'), 'w') as f:
            json.dump({k: {mk: [mv[0], mv[1]] for mk, mv in v.items()} for k, v in results.items()}, f, indent=2)

    if 5 in tables:
        results = run_table5(args.data_dir, device, seeds, config, use_tiger=args.use_tiger)
        all_results['table5'] = results
        print_results_table(results, 'Table 5 结果')
        with open(os.path.join(result_dir, 'table5.json'), 'w') as f:
            json.dump({k: {mk: [mv[0], mv[1]] for mk, mv in v.items()} for k, v in results.items()}, f, indent=2)

    print(f'\n结果已保存到: {result_dir}')
    print('实验完成！')


if __name__ == '__main__':
    main()
