"""
QuaSID 改进对比实验
====================
一次跑完所有改进变体，对比标准 QuaSID:

  A. Hard Negative Mining (5 种配置)
  B. Layer-wise HaMR (3 种配置 + 标准对照)

用法:
  cd improvements
  python run_all_improvements.py                       # 全部 (余弦评估, ~40min)
  python run_all_improvements.py --group hardneg        # 仅 Hard Negative
  python run_all_improvements.py --group layerwise      # 仅 Layer-wise
"""

import sys, os, json, time, gc, argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import (ItemDataset, ContrastivePairDataset,
                        collate_fn_single, collate_fn_pair, build_eval_data)
from curriculum_hamr import (evaluate_cosine, compute_entropy,
                             train_quasid_standard)
from hard_neg_mining import HARDNEG_CONFIGS
from layer_wise_hamr import LAYERWISE_CONFIGS


# ═══════════════════════════════════════════════════════════════════════════════
# 共享配置
# ═══════════════════════════════════════════════════════════════════════════════

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='QuaSID 改进对比实验')
    parser.add_argument('--group', type=str, default='all',
                        choices=['all', 'hardneg', 'layerwise'],
                        help='运行哪组实验')
    parser.add_argument('--seeds', type=str, default='42,123',
                        help='随机种子')
    parser.add_argument('--epochs', type=int, default=100, help='训练 epoch')
    parser.add_argument('--patience', type=int, default=10, help='早停 patience')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    print(f'实验组: {args.group}')
    print(f'Seeds: {seeds}')
    print(f'Epochs: {args.epochs}, Patience: {args.patience}')

    # ── 准备实验配置 ──
    if args.group == 'all':
        experiments = {}
        experiments.update(HARDNEG_CONFIGS)
        experiments.update({f'layer_{k}': v for k, v in LAYERWISE_CONFIGS.items()})
    elif args.group == 'hardneg':
        experiments = HARDNEG_CONFIGS
    else:
        experiments = LAYERWISE_CONFIGS

    # 去重: 两组都有 'standard'，只跑一次
    seen = set()
    experiments_dedup = {}
    for k, v in experiments.items():
        if v['desc'] not in seen:
            seen.add(v['desc'])
            experiments_dedup[k] = v
    experiments = experiments_dedup

    print(f'变体数: {len(experiments)}')
    for k, v in experiments.items():
        print(f'  [{k}] {v["desc"]}')

    # ── 加载数据 ──
    print('\n加载数据...')
    data = torch.load('../data/amazon_Beauty_processed_5core.pt',
                      map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    all_items = sorted(embeddings.keys())

    pair_ds = ContrastivePairDataset(
        data['train_df'], embeddings, data['swing'],
        data['item2idx'], data['user2idx'])
    pair_loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn_pair)

    test_users, _ = build_eval_data(
        data['train_df'], data['test_df'], data['valid_df'],
        embeddings, data['item2idx'], data['user2idx'])
    print(f'测试用户数: {len(test_users)}')

    # ── 结果目录 ──
    result_dir = f"../results/imprv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)

    # ── 运行 ──
    all_results = {}

    for var_name, vcfg in experiments.items():
        print(f'\n{"="*60}')
        print(f'[{var_name}] {vcfg["desc"]}')
        print(f'{"="*60}')

        metrics_keys = ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']
        agg = {k: [] for k in metrics_keys + ['Entropy', 'UsageRate']}

        for seed in seeds:
            print(f'\n  --- Seed={seed} ---')
            set_seed(seed)

            try:
                model = vcfg['factory'](MODEL_CFG).to(device)
                clean_gpu()

                t0 = time.time()
                model, history = train_quasid_standard(
                    model, pair_loader, device,
                    epochs=args.epochs, patience=args.patience, log_interval=20)
                print(f'  训练耗时: {time.time()-t0:.0f}s')

                eval_r = evaluate_cosine(model, embeddings, test_users, all_items, device)
                entropy, usage, n_used = compute_entropy(model, embeddings, all_items, device)

                for k in metrics_keys:
                    agg[k].append(eval_r[k])
                agg['Entropy'].append(entropy)
                agg['UsageRate'].append(usage)

                print(f'  HR@5={eval_r["HR@5"]:.4f}  HR@10={eval_r["HR@10"]:.4f}  '
                      f'NDCG@5={eval_r["NDCG@5"]:.4f}  Entropy={entropy:.2f}  '
                      f'Usage={usage*100:.1f}%')

                del model; clean_gpu()

            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback
                traceback.print_exc()
                clean_gpu()

        if agg['HR@5']:
            all_results[var_name] = {
                'desc': vcfg['desc'],
                **{k: f'{np.mean(agg[k]):.4f}±{np.std(agg[k]):.4f}' for k in metrics_keys},
                'Entropy': f'{np.mean(agg["Entropy"]):.2f}±{np.std(agg["Entropy"]):.2f}',
                'UsageRate': f'{np.mean(agg["UsageRate"])*100:.1f}%',
            }
            print(f'\n  [{var_name}] HR@5={all_results[var_name]["HR@5"]}  '
                  f'NDCG@5={all_results[var_name]["NDCG@5"]}')

    # ── 汇总表 ──
    print(f'\n{"="*90}')
    print(f'改进对比结果 (group={args.group}, {len(seeds)} seeds)')
    print(f'{"="*90}')
    hdr = f'{"变体":<25s} {"HR@5":>14s} {"HR@10":>14s} {"NDCG@5":>14s} {"Entropy":>10s}'
    print(hdr)
    print('-' * 90)
    for name, r in all_results.items():
        print(f'{name:<25s} {r["HR@5"]:>14s} {r["HR@10"]:>14s} '
              f'{r["NDCG@5"]:>14s} {r["Entropy"]:>10s}')

    # ── 找出最佳 ──
    best = max(all_results.items(),
               key=lambda x: float(x[1]['HR@5'].split('±')[0]))
    print(f'\n最佳: [{best[0]}] {best[1]["desc"]}  HR@5={best[1]["HR@5"]}')

    # ── 保存 ──
    save_path = os.path.join(result_dir, 'results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'group': args.group, 'seeds': seeds,
                'epochs': args.epochs, 'patience': args.patience,
            },
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {save_path}')


if __name__ == '__main__':
    main()
