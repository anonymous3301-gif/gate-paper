import torch, torch.nn.functional as F
import numpy as np
import beta_power as bp

def mean_inv_norm2(m, n_batches=4, B=256, V=64):
    """Mean of 1/||k||^2 over all write positions t=1..T-1 (not just key positions)."""
    m.eval()
    vals = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, _, _ = bp.make_mqar(B, 8, 32, V)
            e = m.emb(x) + m.pos(torch.arange(x.size(1))[None])
            e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
            k = m.mem.Wk(e_prev[:, 1:])
            vals.append(1.0 / (k.norm(dim=-1) ** 2 + 1e-6))
    v = torch.cat([z.reshape(-1) for z in vals])
    return v.mean().item()

def eval_rule(m, rule, c=1.0, V=64, n_batches=20):
    """rule: fixed (const 0.125), const (const c), kacz (gate mean E), gate (gate mean 0.125 via c)."""
    m.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, tgt, qs = bp.make_mqar(128, 8, 32, V)
            e = m.emb(x) + m.pos(torch.arange(x.size(1))[None])
            e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
            B, T, D = e.shape
            S = e.new_zeros(B, m.mem.d_mem, D)
            reads = []
            for t in range(T):
                r = torch.einsum('bm,bmd->bd', m.mem.Wq(e[:, t]), S)
                k = m.mem.Wk(e_prev[:, t]); v = m.mem.Wv(e[:, t])
                k_n = F.normalize(k, dim=-1)
                pred = torch.einsum('bmd,bm->bd', S, k_n)
                err = v - pred
                if rule == 'fixed':
                    beta = 0.125
                elif rule == 'const':
                    beta = c
                elif rule == 'kacz':
                    beta = (1.0 / (k.norm(dim=-1, keepdim=True) ** 2 + 1e-6)).unsqueeze(-1)
                elif rule == 'gate':
                    beta = (c / (k.norm(dim=-1, keepdim=True) ** 2 + 1e-6)).unsqueeze(-1)
                S = S + beta * k_n.unsqueeze(-1) * err.unsqueeze(1)
                reads.append(r)
            out = m.head(torch.stack(reads, 1))
            pred = out[:, qs:qs+8].argmax(-1)
            accs.append((pred == tgt).float().mean().item())
    return float(np.mean(accs))

if __name__ == '__main__':
    print(f'{"train":>10s} {"seed":>5s} {"Einv_all":>9s} {"fixed":>8s} {"kacz":>8s} {"gate_m":>8s} {"const_m":>9s}', flush=True)
    for train_mode in ['fixed', 'kaczmarz']:
        for seed in range(10):
            m = bp.train_model(train_mode, 24, seed, beta_fixed=0.125)
            if m is None:
                print(f'{train_mode:>10s} {seed:5d} {"NaN":>9s}', flush=True)
                continue
            e = mean_inv_norm2(m)
            a_fixed = eval_rule(m, 'fixed')
            a_kacz = eval_rule(m, 'kacz')
            a_gate_m = eval_rule(m, 'gate', c=0.125 / e)   # gate shape, mean beta = 0.125
            a_const_m = eval_rule(m, 'const', c=e)         # constant, mean beta = E[1/||k||^2]
            print(f'{train_mode:>10s} {seed:5d} {e:9.4f} {a_fixed:8.4f} {a_kacz:8.4f} {a_gate_m:8.4f} {a_const_m:9.4f}', flush=True)
