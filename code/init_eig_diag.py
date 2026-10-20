"""Initialization key-norm diagnostic for Section 3.1 (Table 3.1).

Reproduces, from scratch, the initialization-time quantities that the paper
claims for the raw-key write:

  * distribution of ||k||^2 over all write positions at initialization
    (mean / median / max, fraction with ||k||^2 > 2)
  * the most negative erase eigenvalue 1 - ||k||^2
  * the first forward loss of the raw-key and normalized-key models

Model: norm_curriculum.Model(V=64, d=64, d_mem=16, key_norm=...) with seed 0,
i.e. exactly the model used by rawk_log.py for the Section 3.2 trajectory.
Write positions are e_prev[:, 1:] of each batch (the model writes at t >= 1).

Run:  python init_eig_diag.py [--batches N] [--out init_eig_diag_out.txt]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import norm_curriculum as nc

torch.set_num_threads(1)


def first_loss(key_norm, seed=0, B=32, V=64):
    torch.manual_seed(seed)
    random.seed(seed)
    m = nc.Model(V, d=64, d_mem=16, key_norm=key_norm)
    x, tgt, qs = nc.make_mqar(B, 8, 32, V)
    out = m(x)
    loss = torch.nn.functional.cross_entropy(
        out[:, qs:qs + 8].reshape(-1, V), tgt.reshape(-1))
    return float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="init_eig_diag_out.txt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    m = nc.Model(64, d=64, d_mem=16, key_norm=False)  # raw-key model

    all_n2 = []
    per_batch = []
    for b in range(args.batches):
        x, tgt, qs = nc.make_mqar(32, 8, 32, 64)
        with torch.no_grad():
            e = m.emb(x) + m.pos(torch.arange(x.size(1))[None])
            e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
            k = m.mem.Wk(e_prev[:, 1:])
            n2 = (k.norm(dim=-1) ** 2).reshape(-1)
        all_n2.append(n2.numpy())
        per_batch.append({
            "batch": b,
            "mean": float(n2.mean()),
            "median": float(n2.median()),
            "max": float(n2.max()),
            "frac_gt_2": float((n2 > 2).double().mean()),
            "n_positions": int(n2.numel()),
        })

    a = np.concatenate(all_n2)
    np.save("init_eig_diag_k2.npy", a)
    counterfactual_betas = [0.03125, 0.0625, 0.125, 0.17678, 0.25, 0.5, 1.0]
    cf = {}
    for b in counterfactual_betas:
        cf[f"{b:g}"] = {
            "frac_gt_2": float((b * a > 2).mean()),
            "mean_b2": float((b * a).mean()),
        }
    res = {
        "config": {
            "seed": args.seed,
            "batches": args.batches,
            "batch_size": 32,
            "d_model": 64,
            "d_mem": 16,
            "n_positions_total": int(a.size),
            "n_positions_per_batch": per_batch[0]["n_positions"],
        },
        "pooled": {
            "mean_k2": float(a.mean()),
            "median_k2": float(np.median(a)),
            "max_k2": float(a.max()),
            "min_k2": float(a.min()),
            "p95_k2": float(np.percentile(a, 95)),
            "frac_gt_2": float((a > 2).mean()),
            "n_le_2": int((a <= 2).sum()),
            "most_negative_eigenvalue": float(1.0 - a.max()),
        },
        "first_forward_loss": {
            "raw": first_loss(False, seed=args.seed),
            "normalized": first_loss(True, seed=args.seed),
        },
        "raw_key_counterfactual_at_init": cf,
        "per_batch": per_batch,
    }

    lines = []
    lines.append("=" * 74)
    lines.append("Initialization key-norm diagnostic (raw-key model, seed "
                 f"{args.seed}, d_mem=16)")
    lines.append(f"batches={args.batches}  positions/batch="
                 f"{res['config']['n_positions_per_batch']}  total="
                 f"{res['config']['n_positions_total']}")
    lines.append("=" * 74)
    p = res["pooled"]
    lines.append("POOLED over all write positions")
    lines.append(f"  mean   ||k||^2          = {p['mean_k2']:.4f}")
    lines.append(f"  median ||k||^2          = {p['median_k2']:.4f}")
    lines.append(f"  p95    ||k||^2          = {p['p95_k2']:.4f}")
    lines.append(f"  max    ||k||^2          = {p['max_k2']:.4f}")
    lines.append(f"  min    ||k||^2          = {p['min_k2']:.4f}")
    lines.append(f"  fraction with ||k||^2>2  = {p['frac_gt_2']:.6f}"
                 f"   ({p['n_le_2']} of {res['config']['n_positions_total']} at or below 2)")
    lines.append(f"  most negative eigenvalue  = {p['most_negative_eigenvalue']:.4f}")
    f = res["first_forward_loss"]
    lines.append("FIRST FORWARD LOSS")
    lines.append(f"  raw key       = {f['raw']:.6e}")
    lines.append(f"  normalized key= {f['normalized']:.6f}")
    lines.append("RAW-KEY COUNTERFACTUAL AT INITIALIZATION")
    lines.append("  fraction of write positions with beta*||k||^2 > 2")
    for b in counterfactual_betas:
        lines.append(f"  beta={b:<8g} mean beta*||k||^2 = {cf[f'{b:g}']['mean_b2']:8.4f}"
                     f"   frac>2 = {cf[f'{b:g}']['frac_gt_2']:.6f}")
    lines.append("PER BATCH")
    for r in per_batch:
        lines.append(f"  batch {r['batch']:2d}: mean={r['mean']:.4f} "
                     f"median={r['median']:.4f} max={r['max']:.4f} "
                     f"frac>2={r['frac_gt_2']:.6f}")
    out = "\n".join(lines) + "\n"
    print(out, end="")
    Path(args.out).write_text(out, encoding="utf-8")
    Path("init_eig_diag.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {args.out} and init_eig_diag.json")


if __name__ == "__main__":
    main()
