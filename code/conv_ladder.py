"""Four-stage ladder with and without a causal convolution (Section 7 control).

Pre-registration and predictions: conv_ladder_predictions.md (written before
this run).

The ladder stages are the budget-matched ones of budget_match.py, re-used
verbatim through the import, and the convolution is the same depthwise causal
conv (kernel 4, groups=d, residual) as in conv_compare.py, so that the
no-conv cells must reproduce the budget-matched ladder rows exactly
(protocol check) and the conv cells are comparable with conv_compare_out.txt.

Run: python conv_ladder.py [--out conv_ladder_out.txt]
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

torch.set_num_threads(2)


class CausalConv(nn.Module):
    """Identical to conv_compare.CausalConv (depthwise, k=4, residual)."""

    def __init__(self, d, k=4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, padding=k - 1, groups=d)
        self.k = k

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.conv(h)[:, :, :x.size(1)]
        return x + h.transpose(1, 2)


class DeltaMem(nn.Module):
    """Identical update rule to budget_match.DeltaMem."""

    def __init__(self, d, d_mem, key_norm=True, beta_mode='one', beta_value=1.0):
        super().__init__()
        self.d_mem = d_mem
        self.key_norm = key_norm
        self.beta_mode = beta_mode
        self.beta_value = beta_value
        self.Wk = nn.Linear(d, d_mem, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d_mem, bias=False)

    def forward(self, e):
        B, T, D = e.shape
        S = e.new_zeros(B, self.d_mem, D)
        e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
        reads = []
        for t in range(T):
            r = torch.einsum('bm,bmd->bd', self.Wq(e[:, t]), S)
            k = self.Wk(e_prev[:, t]); v = self.Wv(e[:, t])
            k_use = F.normalize(k, dim=-1) if self.key_norm else k
            pred = torch.einsum('bmd,bm->bd', S, k_use)
            err = v - pred
            if self.beta_mode == 'one':
                beta = 1.0
            elif self.beta_mode == 'fixed':
                beta = self.beta_value
            elif self.beta_mode == 'kaczmarz':
                beta = (1.0 / (k.norm(dim=-1, keepdim=True) ** 2 + 1e-6)).unsqueeze(-1)
            else:
                raise ValueError(self.beta_mode)
            S = S + beta * k_use.unsqueeze(-1) * err.unsqueeze(1)
            reads.append(r)
        return torch.stack(reads, 1)


class Model(nn.Module):
    def __init__(self, V, d=64, d_mem=16, key_norm=True, beta_mode='one',
                 beta_value=1.0, use_conv=False):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(2048, d)
        self.conv = CausalConv(d) if use_conv else None
        self.mem = DeltaMem(d, d_mem, key_norm, beta_mode, beta_value)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        B, T = x.shape
        e = self.emb(x) + self.pos(torch.arange(T))[None]
        if self.conv is not None:
            e = self.conv(e)
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
            x[b, idx] = keys[i]; idx += 1
            x[b, idx] = vals[i]; idx += 1
        for _ in range(n_dist):
            x[b, idx] = random.randint(0, V - 1); idx += 1
        perm = torch.randperm(n_pairs)
        for i in range(n_pairs):
            x[b, idx] = keys[perm[i]]; tgt[b, i] = vals[perm[i]]; idx += 1
    return x, tgt, 2 * n_pairs + n_dist


def train_eval(stage, use_conv, seed, d_mem=16, steps=1500, lr=3e-4, B=32, V=64):
    key_norm, beta_mode, beta_value = {
        0: (False, 'one', 1.0),
        1: (True, 'one', 1.0),
        2: (True, 'fixed', 0.125),
        3: (True, 'kaczmarz', 1.0),
    }[stage]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    m = Model(V, d=64, d_mem=d_mem, key_norm=key_norm, beta_mode=beta_mode,
              beta_value=beta_value, use_conv=use_conv)
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
    with torch.no_grad():
        accs = []
        for _ in range(20):
            x, tgt, qs = make_mqar(128, 8, 32, V)
            out = m(x)
            pred = out[:, qs:qs + 8].argmax(-1)
            accs.append((pred == tgt).float().mean().item())
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--d_mem", type=int, default=16)
    ap.add_argument("--out", default="conv_ladder_out.txt")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    out = {"config": {"seeds": seeds, "steps": args.steps, "d_mem": args.d_mem,
                       "d_model": 64, "V": 64, "batch": 32, "lr": 3e-4,
                       "conv": "depthwise causal conv, k=4, groups=d, residual",
                       "protocol": "Section 2.4 (AdamW, clip 1.0, 20x128 eval)"},
           "conditions": {}}
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)
        Path(args.out).write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log("=" * 78)
    log("Four-stage ladder x {no conv, conv}  (pre-registered in "
        "conv_ladder_predictions.md)")
    log(f"seeds={seeds} steps={args.steps} d_mem={args.d_mem}")
    log("=" * 78)

    t0 = time.time()
    for use_conv in (False, True):
        for stage in (0, 1, 2, 3):
            accs = []
            for seed in seeds:
                ts = time.time()
                a = train_eval(stage, use_conv, seed, d_mem=args.d_mem,
                               steps=args.steps)
                accs.append(a)
                log(f"conv={str(use_conv):5s} stage={stage} seed={seed} "
                    f"acc={a:.4f} ({time.time()-ts:.1f}s)")
            m = float(np.mean(accs))
            s = float(np.std(accs, ddof=1))
            key = f"stage{stage}|conv={use_conv}"
            out["conditions"][key] = {"accs": accs, "mean": m, "std": s}
            log(f"--- {key}: mean={m:.4f} std={s:.4f} "
                f"per-seed={'/'.join(f'{a:.4f}' for a in accs)}")
            Path("conv_ladder.json").write_text(json.dumps(out, indent=2),
                                                encoding="utf-8")

    # stage-to-stage paired deltas within each conv condition
    log("")
    log("paired stage-to-stage deltas (seed-paired)")
    for use_conv in (False, True):
        for a, b, name in ((0, 1, "stability"), (1, 2, "erase strength"),
                           (2, 3, "gating")):
            da = out["conditions"][f"stage{a}|conv={use_conv}"]["accs"]
            db = out["conditions"][f"stage{b}|conv={use_conv}"]["accs"]
            d = np.array(db) - np.array(da)
            log(f"  conv={str(use_conv):5s} {name:14s} "
                f"per-seed={', '.join(f'{v:+.4f}' for v in d)}  mean={d.mean():+.4f}")
    log("")
    log(f"=== done in {time.time()-t0:.1f}s ===")
    log(f"wrote {args.out} and conv_ladder.json")


if __name__ == "__main__":
    main()
