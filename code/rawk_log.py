import torch, torch.nn.functional as F
import random, csv
import norm_curriculum as nc
torch.set_num_threads(2)

def run(key_norm, seed=0, steps=1500, log_every=10):
    torch.manual_seed(seed); random.seed(seed)
    m = nc.Model(64, d=64, d_mem=16, key_norm=key_norm)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    rows = []
    nan_step = None
    for step in range(steps):
        m.train()
        x, tgt, qs = nc.make_mqar(32, 8, 32, 64)
        out = m(x)
        loss = F.cross_entropy(out[:, qs:qs+8].reshape(-1, 64), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if not torch.isfinite(loss):
            nan_step = step
            rows.append((step, float('nan'), float('nan'), float('nan'), float(gn)))
            break
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                e = m.emb(x) + m.pos(torch.arange(x.size(1))[None])
                e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
                k = m.mem.Wk(e_prev[:, 1:])
                n2 = k.norm(dim=-1) ** 2
                rows.append((step, loss.item(), n2.mean().item(), n2.max().item(), float(gn)))
    return rows, nan_step

if __name__ == '__main__':
    rows = {}
    for key_norm in [False, True]:
        name = 'norm' if key_norm else 'raw'
        r, nan_step = run(key_norm)
        rows[name] = r
        print(f'=== {name} (key_norm={key_norm}) nan_step={nan_step} ===', flush=True)
        print(f'{"step":>6s} {"loss":>9s} {"E_k2":>9s} {"max_k2":>10s} {"grad":>8s}', flush=True)
        for row in r[::max(1, len(r)//15)]:
            print(f'{row[0]:6d} {row[1]:9.4f} {row[2]:9.3f} {row[3]:10.3f} {row[4]:8.3f}', flush=True)
        print(f'  last: step={r[-1][0]} loss={r[-1][1]:.4f} E_k2={r[-1][2]:.3f} max_k2={r[-1][3]:.3f}', flush=True)
    with open('rawk_log.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'step', 'loss', 'E_k2', 'max_k2', 'grad_norm'])
        for name, r in rows.items():
            for row in r:
                w.writerow([name] + list(row))
    print('wrote rawk_log.csv', flush=True)
