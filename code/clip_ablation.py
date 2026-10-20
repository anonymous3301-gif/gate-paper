"""Clip ablation: is the Section 3.2 frozen trap an artifact of clipping?

Section 3.2 reports the raw-key model freezing for ~800 steps under
global-norm clipping at 1.0 (loss oscillating between 1e18 and 1e20, infinite
gradient norm, key norms pinned at initialization scale), then escaping by
shrinking the keys. The paper states claim (i) -- the raw-key update is
unstable at initialization -- as clip-independent, and claim (ii) -- the
frozen trap plus norm-collapse escape -- as conditional on clip = 1.0. This
script runs the control for claim (ii).

Conditions (all 1500 steps, d_mem=16, d=64, lr 3e-4, batch 32, V=64, MQAR
8 pairs / 32 distractors, evaluation at 32 distractors, 20 batches of 128):

  raw,        clip=1.0     (reproduces the Section 3.2 trajectory, 3 seeds)
  raw,        clip=10.0    (finite, larger clip)
  raw,        no clip      (no clipping at all)
  normalized, no clip      (control: the normalized model should be fine)

The raw no-clip condition is expected to produce non-finite updates rather
than a freeze: that is the point of the control. The script records the step at
which the run stops being finite instead of silently continuing.

Run:  python clip_ablation.py [--out clip_ablation_out.txt]
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import norm_curriculum as nc

torch.set_num_threads(2)


def train_eval(key_norm, clip, seed, steps=1500, lr=3e-4, B=32, V=64, log_every=10):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    m = nc.Model(V, d=64, d_mem=16, key_norm=key_norm)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    rows = []
    nan_step = None
    for step in range(steps):
        m.train()
        x, tgt, qs = nc.make_mqar(B, 8, 32, V)
        out = m(x)
        loss = F.cross_entropy(out[:, qs:qs + 8].reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        # total gradient norm is measured in every condition (it is the quantity
        # Section 3.2 uses to explain the freeze); clipping is applied only
        # when a clip value is given.
        tot = 0.0
        for p in m.parameters():
            if p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
        gn = float('inf') if tot == 0.0 else (tot ** 0.5)
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
        if not torch.isfinite(loss):
            nan_step = step
            rows.append((step, float('nan'), float('nan'), float('nan'), gn))
            break
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                e = m.emb(x) + m.pos(torch.arange(x.size(1))[None])
                e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
                k = m.mem.Wk(e_prev[:, 1:])
                n2 = k.norm(dim=-1) ** 2
                rows.append((step, loss.item(), n2.mean().item(), n2.max().item(), gn))

    # evaluation
    acc = float('nan')
    if nan_step is None:
        m.eval()
        with torch.no_grad():
            accs = []
            for _ in range(20):
                x, tgt, qs = nc.make_mqar(128, 8, 32, 64)
                out = m(x)
                pred = out[:, qs:qs + 8].argmax(-1)
                accs.append((pred == tgt).float().mean().item())
            acc = sum(accs) / len(accs)
    return rows, nan_step, acc


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    if np.isnan(a).all():
        return {"n": int(a.size), "mean": float('nan'), "std": float('nan'),
                "median": float('nan'), "min": float('nan'), "max": float('nan'),
                "n_below_0.45": 0, "n_below_0.02": 0}
    ok = a[~np.isnan(a)]
    return {
        "n": int(a.size),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a, ddof=1)) if len(ok) > 1 else 0.0,
        "median": float(np.nanmedian(a)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "n_below_0.45": int((ok < 0.45).sum()),
        "n_below_0.02": int((ok < 0.02).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default="clip_ablation_out.txt")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    conditions = [
        ("raw_clip1.0", False, 1.0),
        ("raw_clip10.0", False, 10.0),
        ("raw_noclip", False, None),
        ("norm_noclip", True, None),
    ]

    out = {"config": {"seeds": seeds, "steps": args.steps, "d_model": 64,
                       "d_mem": 16, "lr": 3e-4, "batch": 32, "V": 64,
                       "n_pairs": 8, "n_distractors": 32,
                       "eval_batches": 20, "eval_batch": 128,
                       "note": "clip=None means no gradient clipping at all"},
           "conditions": {}}

    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)
        Path(args.out).write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    log("=" * 78)
    log("Clip ablation: raw vs normalized key, clip in {1.0, 10.0, none}")
    log(f"seeds={seeds} steps={args.steps}")
    log("=" * 78)

    t0 = time.time()
    for name, key_norm, clip in conditions:
        accs, nan_steps, freezes, losses = [], [], [], []
        for seed in seeds:
            ts = time.time()
            rows, nan_step, acc = train_eval(key_norm, clip, seed, steps=args.steps)
            accs.append(acc)
            nan_steps.append(nan_step)
            # freeze detection: largest logged loss before the first finite-loss step
            big = [r[1] for r in rows if np.isfinite(r[1]) and r[1] > 1e6]
            freezes.append(len(big))
            losses.append(min([r[1] for r in rows if np.isfinite(r[1])], default=float('nan')))
            log(f"{name:14s} seed={seed} acc={acc:.4f} nan_step={nan_step} "
                f"logged_steps_over_1e6={len(big)} "
                f"min_finite_loss={losses[-1]:.4f} ({time.time()-ts:.1f}s)")
            Path("clip_ablation_log.csv").open("a", encoding="utf-8").write(
                "".join(f"{name},{seed},{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n"
                         for r in rows))
        s = summarize(accs)
        out["conditions"][name] = {
            "key_norm": key_norm, "clip": clip, "accs": accs,
            "nan_steps": nan_steps, "logged_steps_over_1e6": freezes,
            "min_finite_loss": losses, "summary": s,
        }
        log(f"--- {name}: acc mean={s['mean']:.4f} std={s['std']:.4f} "
            f"median={s['median']:.4f} range=[{s['min']:.4f},{s['max']:.4f}] "
            f"<0.45={s['n_below_0.45']}/{s['n']} nan_seeds="
            f"{sum(1 for x in nan_steps if x is not None)}/{len(seeds)}")
        Path("clip_ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    log(f"=== done in {time.time()-t0:.1f}s ===")
    log(f"wrote {args.out}, clip_ablation.json, clip_ablation_log.csv")


if __name__ == "__main__":
    main()
