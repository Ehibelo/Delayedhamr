"""
QuaSID-Fast 加速对比实验
=========================
对比标准 QuaSID vs 四种加速策略，测量实际训练时间 + 效果。

用法:
  cd improvements
  python run_fast.py --epochs 50 --seeds 42,123
"""

import sys, os, json, time, gc, argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import (ContrastivePairDataset, collate_fn_pair, build_eval_data)
from curriculum_hamr import (evaluate_cosine, compute_entropy, train_quasid_standard)
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


def clean_gpu():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', type=str,
                        default='standard,sparse_25,sparse_10,lite,lite_nocl')
    parser.add_argument('--seeds', type=str, default='42,123')
    parser.add_argument('--epochs', type=int, default=50, help='训练 epoch')
    parser.add_argument('--patience', type=int, default=10, help='早停')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    variant_list = [v.strip() for v in args.variants.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f'设备: {device}')
    print(f'变体: {variant_list}')
    print(f'Epochs: {args.epochs}')

    # ── 加载数据 ──
    data = torch.load('../data/amazon_Beauty_processed_5core.pt',
                      map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    all_items = sorted(embeddings.keys())

    pair_ds = ContrastivePairDataset(data['train_df'], embeddings, data['swing'],
                                     data['item2idx'], data['user2idx'])
    pair_loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn_pair)
    test_users, _ = build_eval_data(data['train_df'], data['test_df'], data['valid_df'],
                                    embeddings, data['item2idx'], data['user2idx'])

    result_dir = f"../results/fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)

    all_results = {}
    timings = {}  # 记录训练时间

    for var_name in variant_list:
        if var_name not in FAST_CONFIGS:
            print(f'未知变体: {var_name}, 跳过')
            continue
        vcfg = FAST_CONFIGS[var_name]

        print(f'\n{"="*60}')
        print(f'[{var_name}] {vcfg["desc"]}')
        print(f'{"="*60}')

        metrics_keys = ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']
        agg = {k: [] for k in metrics_keys + ['Entropy']}
        var_times = []

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
                elapsed = time.time() - t0
                var_times.append(elapsed)

                eval_r = evaluate_cosine(model, embeddings, test_users, all_items, device)
                entropy, usage, _ = compute_entropy(model, embeddings, all_items, device)

                for k in metrics_keys:
                    agg[k].append(eval_r[k])
                agg['Entropy'].append(entropy)

                print(f'  训练耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)')
                print(f'  HR@5={eval_r["HR@5"]:.4f}  HR@10={eval_r["HR@10"]:.4f}  '
                      f'NDCG@5={eval_r["NDCG@5"]:.4f}')

                del model; clean_gpu()
            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback; traceback.print_exc()
                clean_gpu()

        if agg['HR@5']:
            std_time = np.mean(var_times) if var_name == 'standard' else None
            speedup = std_time / np.mean(var_times) if std_time and var_name != 'standard' else None
            timings[var_name] = np.mean(var_times)

            all_results[var_name] = {
                'desc': vcfg['desc'],
                **{k: f'{np.mean(agg[k]):.4f}±{np.std(agg[k]):.4f}' for k in metrics_keys},
                'Entropy': f'{np.mean(agg["Entropy"]):.2f}',
                'TrainTime': f'{np.mean(var_times):.0f}s',
                'Speedup': f'{speedup:.2f}×' if speedup else '1.00× (基准)',
            }

    # ── 汇总表 ──
    print(f'\n{"="*100}')
    print(f'QuaSID-Fast 加速对比 ({args.epochs} epochs × {len(seeds)} seeds)')
    print(f'{"="*100}')
    hdr = f'{"变体":<20s} {"HR@5":>14s} {"NDCG@5":>14s} {"训练时间":>10s} {"加速比":>8s} {"效果损失":>8s}'
    print(hdr)
    print('-' * 100)

    std_hr5 = float(all_results.get('standard', {}).get('HR@5', '0').split('±')[0])

    for name, r in all_results.items():
        hr5 = float(r['HR@5'].split('±')[0])
        loss = f'{(1 - hr5/std_hr5)*100:+.1f}%' if name != 'standard' else '—'
        print(f'{name:<20s} {r["HR@5"]:>14s} {r["NDCG@5"]:>14s} '
              f'{r["TrainTime"]:>10s} {r["Speedup"]:>8s} {loss:>8s}')

    save_path = os.path.join(result_dir, 'results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({'config': {'epochs': args.epochs, 'seeds': seeds},
                   'results': all_results}, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {save_path}')


if __name__ == '__main__':
    main()
