"""
AMP 混合精度加速对比
====================
对比 fp32 vs AMP(fp16) 训练速度 + 精度，验证 AMP 无损加速。

用法:
  cd improvements
  python run_amp.py --epochs 100 --seeds 42,123
"""

import sys, os, json, time, gc, argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import (ContrastivePairDataset, collate_fn_pair, build_eval_data)
from curriculum_hamr import evaluate_cosine, compute_entropy
from amp_training import train_quasid_amp
from quasid import QuaSID

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
    parser.add_argument('--seeds', type=str, default='42,123')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    print(f'Epochs: {args.epochs}, Seeds: {seeds}')

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

    result_dir = f"../results/amp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)

    experiments = {
        'fp32': {'desc': '标准 fp32 (对照)', 'use_amp': False},
        'amp':  {'desc': 'AMP fp16 混合精度', 'use_amp': True},
    }

    all_results = {}

    for exp_name, exp_cfg in experiments.items():
        print(f'\n{"="*60}')
        print(f'[{exp_name}] {exp_cfg["desc"]}')
        print(f'{"="*60}')

        metrics_keys = ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']
        agg = {k: [] for k in metrics_keys + ['Entropy']}
        var_times = []

        for seed in seeds:
            print(f'\n  --- Seed={seed} ---')
            set_seed(seed)
            try:
                model = QuaSID(**MODEL_CFG).to(device)
                clean_gpu()

                t0 = time.time()
                model, history = train_quasid_amp(
                    model, pair_loader, device,
                    epochs=args.epochs, patience=args.patience,
                    use_amp=exp_cfg['use_amp'], log_interval=20)
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
            all_results[exp_name] = {
                'desc': exp_cfg['desc'],
                **{k: f'{np.mean(agg[k]):.4f}±{np.std(agg[k]):.4f}' for k in metrics_keys},
                'Entropy': f'{np.mean(agg["Entropy"]):.2f}',
                'TrainTime': f'{np.mean(var_times):.0f}s',
            }

    # ── 汇总 ──
    if 'fp32' in all_results and 'amp' in all_results:
        fp32_time = float(all_results['fp32']['TrainTime'].replace('s', ''))
        amp_time = float(all_results['amp']['TrainTime'].replace('s', ''))
        speedup = fp32_time / amp_time

        fp32_hr5 = float(all_results['fp32']['HR@5'].split('±')[0])
        amp_hr5 = float(all_results['amp']['HR@5'].split('±')[0])
        hr5_diff = (amp_hr5 - fp32_hr5) / fp32_hr5 * 100

        print(f'\n{"="*70}')
        print(f'AMP 加速对比 ({args.epochs} epochs × {len(seeds)} seeds)')
        print(f'{"="*70}')
        print(f'{"":<15s} {"HR@5":>14s} {"NDCG@5":>14s} {"训练时间":>10s} {"加速比":>8s}')
        print(f'{"-"*65}')
        print(f'{"fp32 (标准)":<15s} {all_results["fp32"]["HR@5"]:>14s} '
              f'{all_results["fp32"]["NDCG@5"]:>14s} {all_results["fp32"]["TrainTime"]:>10s} {"1.00×":>8s}')
        print(f'{"AMP (fp16)":<15s} {all_results["amp"]["HR@5"]:>14s} '
              f'{all_results["amp"]["NDCG@5"]:>14s} {all_results["amp"]["TrainTime"]:>10s} '
              f'{speedup:.2f}×{"":>4s}')
        print(f'\n加速比: {speedup:.2f}×  精度变化: {hr5_diff:+.1f}%')

    save_path = os.path.join(result_dir, 'results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({'config': {'epochs': args.epochs, 'seeds': seeds},
                   'speedup': speedup if 'speedup' in dir() else None,
                   'results': all_results}, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {save_path}')


if __name__ == '__main__':
    main()
