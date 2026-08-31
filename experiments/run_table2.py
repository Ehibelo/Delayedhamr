"""
Table 2 Experiment: Baseline Comparison
=========================================
Runs RQ-VAE, Improved VQGAN, SimRQ, RQ-VAE-Rotation, QuaSID
Evaluates with TIGER generative retrieval (beam search + Trie).

Usage:
    python experiments/run_table2.py [--models RQVAE,ImpVQGAN,SimRQ,Rotation,QuaSID]
                                     [--seeds 42,123,456,789,1024]
                                     [--beam_size 50]
                                     [--tiger_epochs 100]

Output:
    results/table2/{model_name}.json
"""
import torch, numpy as np, gc, os, json, sys, argparse, time
from datetime import datetime
from torch.utils.data import DataLoader

# Ensure we're in src/ directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

from data_utils import ItemDataset, ContrastivePairDataset, collate_fn_single, collate_fn_pair
from rqvae import RQVAE
from quasid import (QuaSID, ImprovedVQGAN, RQVAERotation, SimRQ,
                     RQKMeans, GRVQ, RQOPQ, compute_hamr_loss)
from tiger import run_tiger_evaluation
from run_paper_experiments import compute_entropy


# ── Config ──────────────────────────────────────────────
MODEL_CFG = {
    'input_dim': 768, 'hidden_dim': 512, 'latent_dim': 32,
    'n_embed': 256, 'n_codebook': 3, 'decay': 0.99,
    'beta': 0.25, 'dropout': 0.1, 'restart_unused_codes': True,
}

TIGER_CFG_DEFAULT = {
    'batch_size': 256, 'lr': 3e-4, 'weight_decay': 1e-5,
    'epochs': 100, 'patience': 5, 'beam_size': 50,
}


# ── Logging ─────────────────────────────────────────────
class TeeLogger:
    """Redirect stdout/stderr to log file + terminal."""
    def __init__(self, log_path, original):
        self.f = open(log_path, 'a', encoding='utf-8', buffering=1)
        self.o = original
    def write(self, m):
        self.o.write(m); self.f.write(m)
    def flush(self):
        self.o.flush(); self.f.flush()


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def clean_gpu():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    time.sleep(1)


# ── Model Training ──────────────────────────────────────
def train_sid_model(model, dataloader, device, epochs=100, patience=5,
                    model_name='model'):
    """Train a standard SID model (RQ-VAE, ImpVQGAN, Rotation)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    model.train()
    best_loss, best_state, pat = float('inf'), None, 0

    # Determine compute_loss argument style by model_name
    _needs_z_zq = model_name in ('ImpVQGAN',)
    _needs_z_only = model_name in ('SimRQ',)

    for ep in range(epochs):
        losses = []
        for feats, idx in dataloader:
            feats = feats.to(device)
            out, qloss, codes, z, z_q = model(feats)
            if _needs_z_zq:
                d = model.compute_loss(out, qloss, feats, z, z_q)
            elif _needs_z_only:
                d = model.compute_loss(out, qloss, feats, z)
            else:
                d = model.compute_loss(out, qloss, feats)
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
            log(f'  [{model_name}] Ep {ep+1}: loss={avg:.6f}  best={best_loss:.6f}  pat={pat}/{patience}')
        if pat >= patience:
            log(f'  [{model_name}] Early stop @ ep {ep+1}')
            break

    model.load_state_dict(best_state)
    return model, best_loss


def train_quasid_model(model, pair_loader, device, epochs=100, patience=5,
                       model_name='QuaSID'):
    """Train QuaSID with HaMR + CVPM + contrastive loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    model.train()
    best_loss, best_state, pat = float('inf'), None, 0

    for ep in range(epochs):
        losses = []
        for tf, ta, ti, tai in pair_loader:
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
            log(f'  [{model_name}] Ep {ep+1}: loss={avg:.6f}  best={best_loss:.6f}  pat={pat}/{patience}')
        if pat >= patience:
            log(f'  [{model_name}] Early stop @ ep {ep+1}')
            break

    model.load_state_dict(best_state)
    return model, best_loss


# ── Main ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Table 2: Baseline Comparison')
    parser.add_argument('--models', type=str,
                        default='RQVAE,RQKMeans,GRVQ,ImpVQGAN,RQOPQ,SimRQ,Rotation,QuaSID',
                        help='Models to run (comma-separated)')
    parser.add_argument('--seeds', type=str, default='42',
                        help='Random seeds')
    parser.add_argument('--beam_size', type=int, default=50)
    parser.add_argument('--tiger_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    tiger_cfg = {**TIGER_CFG_DEFAULT,
                 'beam_size': args.beam_size,
                 'epochs': args.tiger_epochs,
                 'batch_size': args.batch_size}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")
    log(f"Models: {model_list}")
    log(f"Seeds: {seeds}")
    log(f"TIGER config: {tiger_cfg}")

    # Load data
    data = torch.load('../data/amazon_Beauty_processed_5core.pt',
                      map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    all_items = sorted(embeddings.keys())

    # Dataloaders
    single_ds = ItemDataset(all_items, embeddings)
    single_dl = DataLoader(single_ds, batch_size=256, shuffle=True,
                           collate_fn=collate_fn_single)
    pair_ds = ContrastivePairDataset(data['train_df'], embeddings, data['swing'],
                                      data['item2idx'], data['user2idx'])
    pair_dl = DataLoader(pair_ds, batch_size=256, shuffle=True,
                         collate_fn=collate_fn_pair)

    result_dir = f"../results/table2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(result_dir, 'run.log')
    tee_stdout = TeeLogger(log_file, sys.stdout)
    tee_stderr = TeeLogger(log_file, sys.stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    log("=" * 60)
    log("TABLE 2: BASELINE COMPARISON")
    log("=" * 60)

    all_results = {}

    # ── Model definitions ──
    MODEL_REGISTRY = {
        'RQVAE': {
            'factory': lambda: RQVAE(**MODEL_CFG),
            'trainer': 'single',
            'desc': 'RQ-VAE (Residual Quantization VAE)',
        },
        'RQKMeans': {
            'factory': lambda: RQKMeans(**MODEL_CFG),
            'trainer': 'single',
            'desc': 'RQ-KMeans (KMeans++ codebook initialization)',
        },
        'GRVQ': {
            'factory': lambda: GRVQ(**MODEL_CFG),
            'trainer': 'single',
            'desc': 'GRVQ (Gumbel-softmax RQ-VAE)',
        },
        'ImpVQGAN': {
            'factory': lambda: ImprovedVQGAN(**MODEL_CFG),
            'trainer': 'single',
            'desc': 'Improved VQGAN (L2-norm + cosine assignment)',
        },
        'RQOPQ': {
            'factory': lambda: RQOPQ(**MODEL_CFG),
            'trainer': 'single',
            'desc': 'RQ-OPQ (Optimized Product Quantization rotation)',
        },
        'SimRQ': {
            'factory': lambda: SimRQ(**MODEL_CFG),
            'trainer': 'simrq',
            'desc': 'SimRQ (frozen codebooks + similarity preserving)',
        },
        'Rotation': {
            'factory': lambda: RQVAERotation(**MODEL_CFG),
            'trainer': 'single',
            'desc': 'RQ-VAE-Rotation (QR orthogonal rotation)',
        },
        'QuaSID': {
            'factory': lambda: QuaSID(**MODEL_CFG),
            'trainer': 'quasid',
            'desc': 'QuaSID (HaMR + CVPM + contrastive)',
        },
    }

    for seed in seeds:
        log(f"\n{'─'*40}")
        log(f"Seed {seed}")
        set_seed(seed)

        for model_name in model_list:
            if model_name not in MODEL_REGISTRY:
                log(f"  Unknown model: {model_name}, skipping")
                continue

            reg = MODEL_REGISTRY[model_name]
            log(f"\n{'='*60}")
            log(f"[{model_name}] {reg['desc']}")
            log(f"{'='*60}")

            try:
                # Build model (CPU first → GPU for CUDA safety)
                model = reg['factory']()
                model = model.to(device)
                clean_gpu()

                # KMeans init for RQKMeans (must be called before training)
                if hasattr(model, 'kmeans_init_codebooks'):
                    log(f'  [{model_name}] Running KMeans++ codebook initialization...')
                    model.kmeans_init_codebooks(single_dl, device)
                    clean_gpu()

                # Train
                if reg['trainer'] == 'single':
                    model, best_loss = train_sid_model(
                        model, single_dl, device, model_name=model_name)
                elif reg['trainer'] == 'simrq':
                    # SimRQ codebooks already frozen in __init__
                    model, best_loss = train_sid_model(
                        model, single_dl, device, model_name=model_name)
                elif reg['trainer'] == 'quasid':
                    model, best_loss = train_quasid_model(
                        model, pair_dl, device, model_name=model_name)

                # TIGER evaluation
                log(f'  [{model_name}] Starting TIGER evaluation...')
                results, _ = run_tiger_evaluation(
                    model, embeddings, data['train_df'], data['test_df'],
                    data['valid_df'], data['item2idx'], data['user2idx'],
                    device, **tiger_cfg)

                # Entropy (returns tuple: entropy, usage_rate, n_used)
                entropy, usage_rate, n_used = compute_entropy(model, embeddings, all_items, device)
                results['entropy'] = entropy
                results['model'] = model_name
                results['seed'] = seed

                log(f'  ==> {model_name}: HR@5={results["HR@5"]:.4f}  '
                    f'HR@10={results["HR@10"]:.4f}  '
                    f'NDCG@5={results["NDCG@5"]:.4f}  '
                    f'NDCG@10={results["NDCG@10"]:.4f}  '
                    f'Entropy={entropy:.2f}')

                # Save
                key = f'{model_name}_seed{seed}'
                all_results[key] = {k: float(v) if isinstance(v, (int, float))
                                    else v for k, v in results.items()}
                save_path = os.path.join(result_dir, f'{model_name}_seed{seed}.json')
                with open(save_path, 'w') as f:
                    json.dump(all_results[key], f, indent=2)

                del model; clean_gpu()

            except Exception as e:
                log(f'  ERROR [{model_name}]: {e}')
                import traceback
                traceback.print_exc()
                clean_gpu()
                continue

    # Save summary
    summary_path = os.path.join(result_dir, 'all_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    log(f"\n{'='*60}")
    log("TABLE 2 COMPLETE!")
    log(f"Results saved to: {result_dir}")
    for k, v in all_results.items():
        log(f"  {k}: HR@5={v.get('HR@5',0):.4f}  NDCG@5={v.get('NDCG@5',0):.4f}")


if __name__ == '__main__':
    main()
