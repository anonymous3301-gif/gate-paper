"""Decay-only ablation for the Section 7 GDN/Delta check.

DeltaMem with exactly one added degree of freedom: a scalar decay gate
alpha = sigmoid(log_alpha).  There is no convolution, no beta parameterization,
and no GDN gate composition.

Conditions:
  - no_decay: alpha = 1.0, fixed (within-architecture Delta baseline)
  - decay_init_X: learned scalar alpha initialized to sigmoid(X)

The primary readout is MQAR accuracy at J=8; J=0/16/32 are diagnostics.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaDecayMem(nn.Module):
    def __init__(self, d, d_mem, init_log_alpha=3.0, learn_decay=True):
        super().__init__()
        self.d_mem = d_mem
        self.learn_decay = learn_decay
        self.Wk = nn.Linear(d, d_mem, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d_mem, bias=False)
        if learn_decay:
            self.log_alpha = nn.Parameter(torch.tensor([float(init_log_alpha)]))
        else:
            self.register_buffer("log_alpha", torch.tensor([float(init_log_alpha)]))

    def forward(self, e):
        B, T, D = e.shape
        S = e.new_zeros(B, self.d_mem, D)
        e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
        if self.learn_decay:
            alpha = torch.sigmoid(self.log_alpha)
        else:
            alpha = e.new_ones(())
        reads = []
        for t in range(T):
            r = torch.einsum("bm,bmd->bd", self.Wq(e[:, t]), S)
            k = self.Wk(e_prev[:, t])
            v = self.Wv(e[:, t])
            k_n = F.normalize(k, dim=-1)
            pred = torch.einsum("bmd,bm->bd", S, k_n)
            S = alpha * S + k_n.unsqueeze(-1) * (v - pred).unsqueeze(1)
            reads.append(r)
        return torch.stack(reads, 1)


class Model(nn.Module):
    def __init__(self, V, d=64, d_mem=64, init_log_alpha=3.0, learn_decay=True):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(2048, d)
        self.mem = DeltaDecayMem(d, d_mem, init_log_alpha, learn_decay)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        B, T = x.shape
        e = self.emb(x) + self.pos(torch.arange(T))[None]
        return self.head(self.mem(e))


def make_mqar(B, n_pairs, n_dist, V):
    L = 2 * n_pairs + n_dist + n_pairs
    x = torch.zeros(B, L, dtype=torch.long)
    tgt = torch.zeros(B, n_pairs, dtype=torch.long)
    for b in range(B):
        keys = torch.randint(0, V, (n_pairs,))
        vals = torch.randint(0, V, (n_pairs,))
        idx = 0
        for i in range(n_pairs):
            x[b, idx] = keys[i]
            idx += 1
            x[b, idx] = vals[i]
            idx += 1
        for _ in range(n_dist):
            x[b, idx] = random.randint(0, V - 1)
            idx += 1
        perm = torch.randperm(n_pairs)
        for i in range(n_pairs):
            x[b, idx] = keys[perm[i]]
            tgt[b, i] = vals[perm[i]]
            idx += 1
    return x, tgt, 2 * n_pairs + n_dist


def make_mqar_junk(B, n_pairs, n_dist, J, V):
    L = 2 * n_pairs + n_dist + J + n_pairs
    x = torch.zeros(B, L, dtype=torch.long)
    tgt = torch.zeros(B, n_pairs, dtype=torch.long)
    for b in range(B):
        keys = torch.randint(0, V, (n_pairs,))
        vals = torch.randint(0, V, (n_pairs,))
        idx = 0
        for i in range(n_pairs):
            x[b, idx] = keys[i]
            idx += 1
            x[b, idx] = vals[i]
            idx += 1
        for _ in range(n_dist + J):
            x[b, idx] = random.randint(0, V - 1)
            idx += 1
        perm = torch.randperm(n_pairs)
        for i in range(n_pairs):
            x[b, idx] = keys[perm[i]]
            tgt[b, i] = vals[perm[i]]
            idx += 1
    return x, tgt, 2 * n_pairs + n_dist + J


def train_eval(seed, init_log_alpha=3.0, learn_decay=True, d_mem=64, steps=1500,
               lr=1e-3, B=32, V=64):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    m = Model(V, d=64, d_mem=d_mem, init_log_alpha=init_log_alpha,
              learn_decay=learn_decay)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        m.train()
        x, tgt, qs = make_mqar(B, 8, 32, V)
        out = m(x)
        loss = F.cross_entropy(out[:, qs:qs + 8].reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    res = {}
    with torch.no_grad():
        for J in [0, 8, 16, 32]:
            accs = []
            for _ in range(20):
                x, tgt, qs = make_mqar_junk(128, 8, 32, J, V)
                out = m(x)
                pred = out[:, qs:qs + 8].argmax(-1)
                accs.append((pred == tgt).float().mean().item())
            res[J] = sum(accs) / len(accs)
    alpha = torch.sigmoid(m.mem.log_alpha).item() if learn_decay else 1.0
    return res, alpha


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n_below_0.45": int((a < 0.45).sum()),
        "n_below_0.02": int((a < 0.02).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--inits", default="0.5,1.0,2.0")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="delta_decay_only_results.json")
    ap.add_argument("--log", default="delta_decay_only_out.txt")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s]
    inits = [float(s) for s in args.inits.split(",") if s]

    conditions = [("no_decay", None, False)]
    conditions += [(f"decay_init_{x:g}", x, True) for x in inits]

    out = {
        "config": {
            "seeds": seeds,
            "inits": inits,
            "steps": args.steps,
            "threads": args.threads,
            "d_model": 64,
            "d_mem": 64,
            "batch": 32,
            "lr": 1e-3,
            "n_pairs": 8,
            "n_distractors": 32,
            "eval_batches": 20,
            "eval_batch": 128,
        },
        "conditions": {},
    }

    log_lines = []
    def log(s):
        print(s, flush=True)
        log_lines.append(s)
        Path(args.log).write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log("=" * 78)
    log("Decay-only ablation: DeltaMem + scalar decay (no conv, no beta gate)")
    log(f"seeds={seeds} inits={inits} steps={args.steps} threads={args.threads}")
    log("=" * 78)

    t0 = time.time()
    for name, init, learn in conditions:
        rows = []
        for seed in seeds:
            ts = time.time()
            res, alpha = train_eval(
                seed, init_log_alpha=(3.0 if init is None else init),
                learn_decay=learn, d_mem=64, steps=args.steps,
            )
            row = {"seed": seed, "alpha": float(alpha),
                   **{f"J{J}": float(res[J]) for J in [0, 8, 16, 32]}}
            rows.append(row)
            log(f"{name:18s} seed={seed:2d} alpha={alpha:.4f} "
                f"J0={res[0]:.4f} J8={res[8]:.4f} "
                f"J16={res[16]:.4f} J32={res[32]:.4f} "
                f"({time.time()-ts:.1f}s)")
        j8 = [r["J8"] for r in rows]
        summary = {"J8": summarize(j8)}
        for J in [0, 16, 32]:
            summary[f"J{J}"] = summarize([r[f"J{J}"] for r in rows])
        out["conditions"][name] = {
            "init_log_alpha": init,
            "learn_decay": learn,
            "rows": rows,
            "summary": summary,
        }
        log(f"--- {name}: J8 mean={summary['J8']['mean']:.4f} "
            f"std={summary['J8']['std']:.4f} median={summary['J8']['median']:.4f} "
            f"range=[{summary['J8']['min']:.4f},{summary['J8']['max']:.4f}] "
            f"<0.45={summary['J8']['n_below_0.45']}/{summary['J8']['n']} "
            f"<0.02={summary['J8']['n_below_0.02']}/{summary['J8']['n']}")
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    log(f"=== done in {time.time()-t0:.1f}s ===")
    log(f"wrote {args.out} and {args.log}")


if __name__ == "__main__":
    main()
