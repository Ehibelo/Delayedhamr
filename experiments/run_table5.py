"""
Table 5 Experiment: Ablation Study
====================================
Ablates QuaSID components to measure individual contributions:
  - w/o CVPM: removes Collision-Vulnerable Pair Mining (keeps HaMR only)
  - w/o HaMR: removes Hamming-distance-Aware Margin Regularization (keeps CVPM only)

Usage:
    python experiments/run_table5.py [--models wo_CVPM,wo_HaMR]
                                     [--seeds 42]
                                     [--beam_size 50]
                                     [--tiger_epochs 100]

Output:
    results/table5/{variant_name}.json
"""
import torch, numpy as np, gc, os, json, sys, argparse, time
from datetime import datetime
from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

from data_utils import ContrastivePairDataset, collate_fn_pair
from quasid import QuaSID
from tiger import run_tiger_evaluation


MODEL_CFG = {
    'input_dim': 768, 'hidden_dim': 512, 'latent_dim': 32,
    'n_embed': 256, 'n_codebook': 3, 'decay': 0.99,
    'beta': 0.25, 'dropout': 0.1, 'restart_unused_codes': True,
}
TIGER_CFG_DEFAULT = {
    'batch_size': 256, 'lr': 3e-4, 'weight_decay': 1e-5,
    'epochs': 100, 'patience': 5, 'beam_size': 50,
}


class TeeLogger:
    def __init__(self, path, orig):
        self.f = open(path, 'a', encoding='utf-8', buffering=1)
        self.o = orig
    def write(self, m): self.o.write(m); self.f.write(m)
    def flush(self): self.o.flush(); self.f.flush()


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def set_seed(s):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def clean_gpu():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description='Table 5: Ablation Study')
    parser.add_argument('--models', type=str, default='wo_CVPM,wo_HaMR')
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--beam_size', type=int, default=50)
    parser.add_argument('--tiger_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    tiger_cfg = {**TIGER_CFG_DEFAULT,
                 'beam_size': args.beam_size, 'epochs': args.tiger_epochs,
                 'batch_size': args.batch_size}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data = torch.load('../data/amazon_Beauty_processed_5core.pt',
                      map_location='cpu', weights_only=False)
    embeddings = data['embeddings']

    pair_ds = ContrastivePairDataset(data['train_df'], embeddings, data['swing'],
                                      data['item2idx'], data['user2idx'])
    pair_dl = DataLoader(pair_ds, batch_size=256, shuffle=True,
                         collate_fn=collate_fn_pair)

    result_dir = f"../results/table5_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, 'run.log')
    sys.stdout = TeeLogger(log_file, sys.stdout)
    sys.stderr = TeeLogger(log_file, sys.stderr)

    log("=" * 60)
    log("TABLE 5: ABLATION STUDY")
    log("=" * 60)

    ALL_RESULTS = {}

    ABLATION_SPECS = {
        'wo_CVPM': {
            'use_cvpm': False, 'lambda_full': 0.2, 'lambda_partial': 0.1,
            'desc': 'QuaSID without CVPM (HaMR only)',
        },
        'wo_HaMR': {
            'use_cvpm': False, 'lambda_full': 0.0, 'lambda_partial': 0.0,
            'desc': 'QuaSID without HaMR (CVPM only) — note: HaMR off = λ=0',
        },
    }

    for seed in seeds:
        set_seed(seed)
        log(f"\nSeed {seed}")

        for var_name in model_list:
            spec = ABLATION_SPECS.get(var_name)
            if not spec:
                log(f"Unknown variant: {var_name}, skipping")
                continue

            log(f"\n{'='*60}")
            log(f"[{var_name}] {spec['desc']}")
            log(f"{'='*60}")

            try:
                model = QuaSID(**MODEL_CFG,
                               use_cvpm=spec['use_cvpm'],
                               lambda_full=spec['lambda_full'],
                               lambda_partial=spec['lambda_partial']).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
                model.train()
                best_loss, best_state, pat = float('inf'), None, 0

                for ep in range(100):
                    losses = []
                    for tf, ta, ti, tai in pair_dl:
                        tf, ta = tf.to(device), ta.to(device)
                        ti, tai = ti.to(device), tai.to(device)
                        feats = torch.cat([tf, ta], dim=0)
                        out, qloss, codes, z, z_q = model(feats)
                        d = model.compute_loss(out, qloss, codes, z, tf, ta, ti, tai)
                        optimizer.zero_grad()
                        d['loss_total'].backward()
                        optimizer.step()
                        losses.append(d['loss_total'].item())

                    avg = np.mean(losses)
                    if avg < best_loss:
                        best_loss = avg; pat = 0
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        pat += 1

                    if (ep + 1) % 10 == 0:
                        log(f'  Ep {ep+1}: loss={avg:.6f}  best={best_loss:.6f}  pat={pat}/5')
                    if pat >= 5:
                        log(f'  Early stop @ ep {ep+1}')
                        break

                model.load_state_dict(best_state)
                clean_gpu()
                log(f'  Train done. Best loss={best_loss:.6f}')

                # TIGER evaluation
                log("TIGER evaluation...")
                results, _ = run_tiger_evaluation(
                    model, embeddings, data['train_df'], data['test_df'],
                    data['valid_df'], data['item2idx'], data['user2idx'],
                    device, **tiger_cfg)
                log(f'  ==> {var_name}: HR@5={results["HR@5"]:.4f}  '
                    f'HR@10={results["HR@10"]:.4f}  '
                    f'NDCG@5={results["NDCG@5"]:.4f}  '
                    f'NDCG@10={results["NDCG@10"]:.4f}')

                key = f'{var_name}_seed{seed}'
                ALL_RESULTS[key] = {k: float(v) for k, v in results.items()}
                with open(os.path.join(result_dir, f'{var_name}_seed{seed}.json'), 'w') as f:
                    json.dump(ALL_RESULTS[key], f, indent=2)

                del model; clean_gpu()

            except Exception as e:
                log(f'  ERROR [{var_name}]: {e}')
                import traceback; traceback.print_exc()
                clean_gpu()
                continue

    with open(os.path.join(result_dir, 'all_results.json'), 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)

    log(f"\n{'='*60}")
    log("TABLE 5 COMPLETE!")
    log(f"Results saved to: {result_dir}")
    for k, v in ALL_RESULTS.items():
        log(f"  {k}: HR@5={v.get('HR@5',0):.4f}")


if __name__ == '__main__':
    main()
