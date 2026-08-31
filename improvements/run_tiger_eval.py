"""
对指定 QuaSID 变体跑完整 TIGER 评估
====================================
用法:
  cd improvements
  python run_tiger_eval.py --variant sparse_25 --seed 42 --beam_size 100 --tiger_epochs 200
"""

import sys, os, json, time, gc, argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import (ContrastivePairDataset, collate_fn_pair, build_eval_data)
from tiger import run_tiger_evaluation
from curriculum_hamr import train_quasid_standard, compute_entropy
from quasid_fast import FAST_CONFIGS

MODEL_CFG = {
    'input_dim': 768, 'hidden_dim': 512, 'latent_dim': 32,
    'n_embed': 256, 'n_codebook': 3, 'decay': 0.99,
    'beta': 0.25, 'dropout': 0.1, 'restart_unused_codes': True,
    'tau': 0.07, 'lambda_cl': 0.1,
    'm_full': 0.8, 'm_partial': 0.5, 'R': 1,
    'lambda_full': 0.2, 'lambda_partial': 0.1,
    'use_cvpm': True,
}


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True,
                        choices=list(FAST_CONFIGS.keys()),
                        help='要评估的变体名')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100, help='SID 训练 epoch')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('--tiger_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    vcfg = FAST_CONFIGS[args.variant]

    print(f'设备: {device}')
    print(f'变体: [{args.variant}] {vcfg["desc"]}')
    print(f'Seed: {args.seed}')
    print(f'Beam size: {args.beam_size}')
    print(f'TIGER epochs: {args.tiger_epochs}')

    # ── 加载数据 ──
    print('\n加载数据...')
    data = torch.load('../data/amazon_Beauty_processed_5core.pt',
                      map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    all_items = sorted(embeddings.keys())

    pair_ds = ContrastivePairDataset(data['train_df'], embeddings, data['swing'],
                                     data['item2idx'], data['user2idx'])
    pair_loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn_pair)

    # ── 训练 SID 模型 ──
    set_seed(args.seed)
    model = vcfg['factory'](MODEL_CFG).to(device)
    gc.collect(); torch.cuda.empty_cache()

    print(f'\n训练 SID 模型 ({args.epochs} epochs)...')
    t0 = time.time()
    model, history = train_quasid_standard(
        model, pair_loader, device,
        epochs=args.epochs, patience=args.patience, log_interval=10)
    train_time = time.time() - t0
    print(f'SID 训练耗时: {train_time:.0f}s ({train_time/60:.1f}min)')

    # ── 计算熵 ──
    entropy, usage, n_used = compute_entropy(model, embeddings, all_items, device)
    print(f'Entropy: {entropy:.2f}  Usage: {usage*100:.1f}%  N_used: {n_used}')

    # ── TIGER 评估 ──
    print(f'\n{"="*60}')
    print('TIGER 生成式检索评估')
    print(f'{"="*60}')

    tiger_cfg = {
        'batch_size': args.batch_size, 'lr': 3e-4, 'weight_decay': 1e-5,
        'epochs': args.tiger_epochs, 'patience': args.patience,
        'beam_size': args.beam_size,
    }
    results, tiger_model = run_tiger_evaluation(
        model, embeddings, data['train_df'], data['test_df'], data['valid_df'],
        data['item2idx'], data['user2idx'], device, **tiger_cfg)

    # ── 保存 ──
    result_dir = f"../results/tiger_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)

    output = {
        'variant': args.variant, 'desc': vcfg['desc'],
        'seed': args.seed,
        'sid_train_time_s': train_time,
        'tiger_config': tiger_cfg,
        'entropy': entropy, 'usage_rate': usage, 'n_used': n_used,
        **{k: float(v) for k, v in results.items()},
    }
    with open(os.path.join(result_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\n{"="*60}')
    print('TIGER 结果:')
    for k in ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']:
        print(f'  {k}: {results.get(k, 0):.4f}')
    print(f'  Entropy: {entropy:.2f}')
    print(f'结果已保存: {result_dir}')


if __name__ == '__main__':
    main()
