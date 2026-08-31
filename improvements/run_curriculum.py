"""
Curriculum HaMR 对比实验
========================
比较标准 QuaSID vs 三种课程策略 (linear / cosine / r_anneal)

用法:
  cd improvements
  python run_curriculum.py                     # 余弦评估 (快速, ~15min)
  python run_curriculum.py --use_tiger          # TIGER 评估 (完整, ~3h)
  python run_curriculum.py --variants linear,cosine  # 只跑指定变体
"""

import sys, os, json, time, gc, argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# 路径设置
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import (ItemDataset, ContrastivePairDataset,
                        collate_fn_single, collate_fn_pair, build_eval_data)
from quasid import QuaSID
from curriculum_hamr import (QuaSIDCurriculum, train_curriculum,
                             train_quasid_standard, evaluate_cosine, compute_entropy)
from tiger import run_tiger_evaluation


# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_CFG = {
    'input_dim': 768, 'hidden_dim': 512, 'latent_dim': 32,
    'n_embed': 256, 'n_codebook': 3, 'decay': 0.99,
    'beta': 0.25, 'dropout': 0.1, 'restart_unused_codes': True,
    # QuaSID 超参数 (论文默认)
    'tau': 0.07, 'lambda_cl': 0.1,
    'm_full': 0.8, 'm_partial': 0.5, 'R': 1,
    'lambda_full': 0.2, 'lambda_partial': 0.1,
    'use_cvpm': True,
}

# 三种课程策略
CURRICULUM_VARIANTS = {
    'standard': {
        'desc': '标准 QuaSID (m_full=0.8 全程固定)',
        'factory': lambda: QuaSID(**MODEL_CFG),
        'trainer': 'standard',
    },
    'linear': {
        'desc': 'Curriculum-Linear: margin 从 0.08 → 0.8 线性增长 (warmup=20)',
        'factory': lambda: QuaSIDCurriculum(
            warmup_epochs=20, schedule_mode='linear',
            margin_start_ratio=0.1, **MODEL_CFG),
        'trainer': 'curriculum',
    },
    'cosine': {
        'desc': 'Curriculum-Cosine: margin 余弦曲线增长, 前期慢后期快 (warmup=20)',
        'factory': lambda: QuaSIDCurriculum(
            warmup_epochs=20, schedule_mode='cosine',
            margin_start_ratio=0.1, **MODEL_CFG),
        'trainer': 'curriculum',
    },
    'r_anneal': {
        'desc': 'Curriculum-R-Anneal: R 从 2→1 退火 + margin 线性增长 (warmup=20)',
        'factory': lambda: QuaSIDCurriculum(
            warmup_epochs=20, schedule_mode='r_anneal',
            margin_start_ratio=0.1, **MODEL_CFG),
        'trainer': 'curriculum',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════════════════════

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


def main():
    parser = argparse.ArgumentParser(description='Curriculum HaMR 对比实验')
    parser.add_argument('--variants', type=str, default='standard,linear,cosine,r_anneal',
                        help='要运行的变体 (逗号分隔)')
    parser.add_argument('--seeds', type=str, default='42',
                        help='随机种子')
    parser.add_argument('--epochs', type=int, default=100, help='训练 epoch 数')
    parser.add_argument('--patience', type=int, default=10, help='早停 patience')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--use_tiger', action='store_true',
                        help='使用 TIGER 评估 (耗时较长)')
    parser.add_argument('--beam_size', type=int, default=50)
    parser.add_argument('--tiger_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    variant_list = [v.strip() for v in args.variants.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    eval_method = 'TIGER' if args.use_tiger else 'Cosine'
    print(f'设备: {device}')
    print(f'评估方法: {eval_method}')
    print(f'变体: {variant_list}')
    print(f'Seeds: {seeds}')
    print(f'Epochs: {args.epochs}, Patience: {args.patience}')

    # ── 加载数据 ──
    print('\n加载数据...')
    data = torch.load('../data/amazon_Beauty_processed_5core.pt',
                      map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    all_items = sorted(embeddings.keys())

    single_ds = ItemDataset(all_items, embeddings)
    pair_ds = ContrastivePairDataset(
        data['train_df'], embeddings, data['swing'],
        data['item2idx'], data['user2idx'])
    pair_loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn_pair)

    test_users, _ = build_eval_data(
        data['train_df'], data['test_df'], data['valid_df'],
        embeddings, data['item2idx'], data['user2idx'])
    print(f'测试用户数: {len(test_users)}')

    # ── 创建结果目录 ──
    result_dir = f"../results/curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)

    # ── 运行实验 ──
    all_results = {}

    for variant_name in variant_list:
        if variant_name not in CURRICULUM_VARIANTS:
            print(f'\n未知变体: {variant_name}, 跳过')
            continue

        vcfg = CURRICULUM_VARIANTS[variant_name]
        print(f'\n{"="*60}')
        print(f'[{variant_name}] {vcfg["desc"]}')
        print(f'{"="*60}')

        variant_results = {'HR@5': [], 'HR@10': [], 'NDCG@5': [], 'NDCG@10': [],
                           'Entropy': [], 'UsageRate': []}

        for seed in seeds:
            print(f'\n  --- Seed={seed} ---')
            set_seed(seed)

            try:
                # 创建模型
                model = vcfg['factory']().to(device)
                clean_gpu()

                # 训练
                t0 = time.time()
                if vcfg['trainer'] == 'curriculum':
                    model, history = train_curriculum(
                        model, pair_loader, device,
                        epochs=args.epochs, patience=args.patience,
                        log_interval=10)
                else:
                    model, history = train_quasid_standard(
                        model, pair_loader, device,
                        epochs=args.epochs, patience=args.patience,
                        log_interval=10)
                train_time = time.time() - t0
                print(f'  训练耗时: {train_time:.1f}s')

                # 评估
                if args.use_tiger:
                    print('  运行 TIGER 评估...')
                    tiger_cfg = {
                        'batch_size': args.batch_size, 'lr': 3e-4,
                        'weight_decay': 1e-5, 'epochs': args.tiger_epochs,
                        'patience': args.patience, 'beam_size': args.beam_size,
                    }
                    results, _ = run_tiger_evaluation(
                        model, embeddings, data['train_df'], data['test_df'],
                        data['valid_df'], data['item2idx'], data['user2idx'],
                        device, **tiger_cfg)
                    eval_results = {k: results[k] for k in
                                    ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']}
                else:
                    raw_eval = evaluate_cosine(
                        model, embeddings, test_users, all_items, device)
                    # 只保留 variant_results 中已有的 key
                    eval_results = {k: raw_eval[k] for k in variant_results
                                    if k in raw_eval}

                entropy, usage_rate, n_used = compute_entropy(
                    model, embeddings, all_items, device)

                for k, v in eval_results.items():
                    variant_results[k].append(v)
                variant_results['Entropy'].append(entropy)
                variant_results['UsageRate'].append(usage_rate)

                print(f'  HR@5={eval_results["HR@5"]:.4f}  '
                      f'HR@10={eval_results["HR@10"]:.4f}  '
                      f'NDCG@5={eval_results["NDCG@5"]:.4f}  '
                      f'Entropy={entropy:.2f}  Usage={usage_rate*100:.1f}%')

                del model; clean_gpu()

            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback
                traceback.print_exc()
                clean_gpu()

        # 汇总
        if variant_results['HR@5']:
            avg = {k: (np.mean(v), np.std(v)) for k, v in variant_results.items()}
            all_results[variant_name] = {
                'desc': vcfg['desc'],
                'HR@5': f'{avg["HR@5"][0]:.4f}±{avg["HR@5"][1]:.4f}',
                'HR@10': f'{avg["HR@10"][0]:.4f}±{avg["HR@10"][1]:.4f}',
                'NDCG@5': f'{avg["NDCG@5"][0]:.4f}±{avg["NDCG@5"][1]:.4f}',
                'NDCG@10': f'{avg["NDCG@10"][0]:.4f}±{avg["NDCG@10"][1]:.4f}',
                'Entropy': f'{avg["Entropy"][0]:.2f}±{avg["Entropy"][1]:.2f}',
                'UsageRate': f'{avg["UsageRate"][0]*100:.1f}%±{avg["UsageRate"][1]*100:.1f}%',
            }
            print(f'\n  [{variant_name}] 均值: '
                  f'HR@5={all_results[variant_name]["HR@5"]}  '
                  f'NDCG@5={all_results[variant_name]["NDCG@5"]}')

    # ── 打印对比表 ──
    print(f'\n{"="*80}')
    print('Curriculum HaMR 对比结果')
    print(f'{"="*80}')
    header = f'{"变体":<25s} {"HR@5":>12s} {"HR@10":>12s} {"NDCG@5":>12s} {"Entropy":>10s} {"Usage":>8s}'
    print(header)
    print('-' * 80)
    for name, r in all_results.items():
        print(f'{name:<25s} {r["HR@5"]:>12s} {r["HR@10"]:>12s} '
              f'{r["NDCG@5"]:>12s} {r["Entropy"]:>10s} {r["UsageRate"]:>8s}')

    # ── 保存 ──
    save_path = os.path.join(result_dir, 'results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'variants': variant_list, 'seeds': seeds,
                'epochs': args.epochs, 'patience': args.patience,
                'eval_method': eval_method, 'use_tiger': args.use_tiger,
                'beam_size': args.beam_size, 'tiger_epochs': args.tiger_epochs,
            },
            'results': all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {save_path}')


if __name__ == '__main__':
    main()
