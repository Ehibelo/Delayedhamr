"""
Table 3 Experiment: HaMR Plug-and-Play
=======================================
Applies HaMR (Hamming-distance-Aware Margin Regularization) loss to
baseline SID models (RQ-VAE, Improved VQGAN) as a plug-and-play component.

Two-stage training:
  1. Pretrain baseline model (standard reconstruction)
  2. Finetune with HaMR loss on contrastive pairs

Usage:
    python experiments/run_table3.py [--models RQ-VAE+HaMR,ImpVQGAN+HaMR]
                                     [--seeds 42]
                                     [--beam_size 50]
                                     [--tiger_epochs 100]

Output:
    results/table3/{model_name}.json
"""
import torch, numpy as np, gc, os, json, sys, argparse, time
from datetime import datetime
from torch.utils.data import DataLoader

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

from data_utils import ItemDataset, ContrastivePairDataset, collate_fn_single, collate_fn_pair
from rqvae import RQVAE
from quasid import QuaSID, ImprovedVQGAN, compute_hamr_loss
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
HAMR_CFG = {
    'm_full': 0.8, 'm_partial': 0.5, 'R': 1,
    'lambda_full': 0.2, 'lambda_partial': 0.1,
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
    parser = argparse.ArgumentParser(description='Table 3: HaMR Plug-and-Play')
    parser.add_argument('--models', type=str, default='RQ-VAE+HaMR,ImpVQGAN+HaMR')
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

    # Load data
    data = torch.load('../data/amazon_Beauty_processed_5core.pt',
                      map_location='cpu', weights_only=False)
    embeddings = data['embeddings']
    all_items = sorted(embeddings.keys())

    single_ds = ItemDataset(all_items, embeddings)
    single_dl = DataLoader(single_ds, batch_size=256, shuffle=True,
                           collate_fn=collate_fn_single)
    pair_ds = ContrastivePairDataset(data['train_df'], embeddings, data['swing'],
                                      data['item2idx'], data['user2idx'])
    pair_dl = DataLoader(pair_ds, batch_size=256, shuffle=True,
                         collate_fn=collate_fn_pair)

    result_dir = f"../results/table3_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, 'run.log')
    sys.stdout = TeeLogger(log_file, sys.stdout)
    sys.stderr = TeeLogger(log_file, sys.stderr)

    log("=" * 60)
    log("TABLE 3: HAMR PLUG-AND-PLAY")
    log("=" * 60)

    ALL_RESULTS = {}

    MODEL_SPECS = {
        'RQ-VAE+HaMR': {
            'base': 'rqvae', 'pre_quant': 'identity', 'norm_cb': False,
            'desc': 'RQ-VAE + HaMR finetune',
        },
        'ImpVQGAN+HaMR': {
            'base': 'impvqgan', 'pre_quant': 'l2_norm', 'norm_cb': True,
            'desc': 'Improved VQGAN + HaMR finetune',
        },
    }

    for seed in seeds:
        set_seed(seed)
        log(f"\nSeed {seed}")

        for model_name in model_list:
            spec = MODEL_SPECS.get(model_name)
            if not spec:
                log(f"Unknown model: {model_name}, skipping")
                continue

            log(f"\n{'='*60}")
            log(f"[{model_name}] {spec['desc']}")
            log(f"{'='*60}")

            try:
                # ── Stage 1: Pretrain baseline ──
                log("Stage 1: Pretrain baseline...")
                if spec['base'] == 'rqvae':
                    pretrain = RQVAE(**MODEL_CFG).to(device)
                else:
                    pretrain = ImprovedVQGAN(**MODEL_CFG).to(device)

                opt1 = torch.optim.Adam(pretrain.parameters(), lr=3e-4, weight_decay=1e-5)
                pretrain.train()
                bl, bs, pat = float('inf'), None, 0
                for ep in range(100):
                    losses = []
                    for feats, idx in single_dl:
                        feats = feats.to(device)
                        out, qloss, codes, z, zq = pretrain(feats)
                        if spec['base'] == 'impvqgan':
                            d = pretrain.compute_loss(out, qloss, feats, z, zq)
                        else:
                            d = pretrain.compute_loss(out, qloss, feats)
                        opt1.zero_grad(); d['loss_total'].backward(); opt1.step()
                        losses.append(d['loss_total'].item())
                    avg = np.mean(losses)
                    if avg < bl:
                        bl = avg; pat = 0
                        bs = {k: v.cpu().clone() for k, v in pretrain.state_dict().items()}
                    else:
                        pat += 1
                    if (ep + 1) % 10 == 0:
                        log(f'  Pretrain Ep {ep+1}: loss={avg:.6f}')
                    if pat >= 5: break
                pretrain.load_state_dict(bs)
                pretrain_state = {k: v.cpu().clone() for k, v in pretrain.state_dict().items()}
                del pretrain, opt1; clean_gpu()
                log(f'  Pretrain done. Best loss={bl:.6f}')

                # ── Stage 2: HaMR finetune ──
                log("Stage 2: HaMR finetune...")
                quasid = QuaSID(**MODEL_CFG, lambda_cl=0.0,
                                pre_quant_mode=spec['pre_quant'],
                                normalize_codebook=spec['norm_cb']).to(device)
                # Load pretrained weights
                qs = quasid.state_dict()
                for k in pretrain_state:
                    if k in qs:
                        qs[k] = pretrain_state[k].to(device)
                quasid.load_state_dict(qs)
                del pretrain_state

                opt2 = torch.optim.Adam(quasid.parameters(), lr=3e-4, weight_decay=1e-5)
                quasid.train()
                bl, bs, pat = float('inf'), None, 0
                for ep in range(100):
                    losses = []
                    for tf, ta, ti, tai in pair_dl:
                        tf, ta = tf.to(device), ta.to(device)
                        ti, tai = ti.to(device), tai.to(device)
                        feats = torch.cat([tf, ta], dim=0)
                        out, qloss, codes, z, z_q = quasid(feats)
                        loss_recon = torch.nn.functional.mse_loss(out, feats)
                        loss_total = loss_recon + quasid.beta * qloss
                        item_ids = torch.cat([ti, tai], dim=0)
                        loss_hamr = compute_hamr_loss(
                            codes, z, item_ids, **HAMR_CFG,
                            use_cvpm=True, trigger_item_ids=ti, target_item_ids=tai)
                        loss_total = loss_total + loss_hamr
                        opt2.zero_grad(); loss_total.backward(); opt2.step()
                        losses.append(loss_total.item())
                    avg = np.mean(losses)
                    if avg < bl:
                        bl = avg; pat = 0
                        bs = {k: v.cpu().clone() for k, v in quasid.state_dict().items()}
                    else:
                        pat += 1
                    if (ep + 1) % 10 == 0:
                        log(f'  HaMR Ep {ep+1}: loss={avg:.6f}')
                    if pat >= 5: break
                quasid.load_state_dict(bs)
                clean_gpu()
                log(f'  HaMR done. Best loss={bl:.6f}')

                # ── TIGER evaluation ──
                log("TIGER evaluation...")
                results, _ = run_tiger_evaluation(
                    quasid, embeddings, data['train_df'], data['test_df'],
                    data['valid_df'], data['item2idx'], data['user2idx'],
                    device, **tiger_cfg)
                log(f'  ==> {model_name}: HR@5={results["HR@5"]:.4f}  '
                    f'HR@10={results["HR@10"]:.4f}  '
                    f'NDCG@5={results["NDCG@5"]:.4f}  '
                    f'NDCG@10={results["NDCG@10"]:.4f}')

                key = f'{model_name}_seed{seed}'
                ALL_RESULTS[key] = {k: float(v) for k, v in results.items()}
                with open(os.path.join(result_dir, f'{model_name}_seed{seed}.json'), 'w') as f:
                    json.dump(ALL_RESULTS[key], f, indent=2)

                del quasid; clean_gpu()

            except Exception as e:
                log(f'  ERROR [{model_name}]: {e}')
                import traceback; traceback.print_exc()
                clean_gpu()
                continue

    with open(os.path.join(result_dir, 'all_results.json'), 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)

    log(f"\n{'='*60}")
    log("TABLE 3 COMPLETE!")
    log(f"Results saved to: {result_dir}")
    for k, v in ALL_RESULTS.items():
        log(f"  {k}: HR@5={v.get('HR@5',0):.4f}")


if __name__ == '__main__':
    main()
