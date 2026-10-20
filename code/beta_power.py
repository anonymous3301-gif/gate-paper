import torch, torch.nn as nn, torch.nn.functional as F
import random, numpy as np, sys
torch.set_num_threads(2)

class Mem(nn.Module):
    def __init__(self, d, d_mem, beta_mode='one', mu=1.0, beta_fixed=1.0):
        super().__init__()
        self.d_mem = d_mem; self.beta_mode = beta_mode
        self.mu = mu; self.beta_fixed = beta_fixed
        self.Wk = nn.Linear(d, d_mem, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d_mem, bias=False)
        self.register_buffer('power_ema', torch.zeros(()))
        self.register_buffer('ema_ready', torch.zeros((), dtype=torch.bool))
        self.b2_sum = 0.0; self.b2_cnt = 0; self.b2_over = 0; self.b2_vals = []
    def forward(self, e):
        B, T, D = e.shape
        S = e.new_zeros(B, self.d_mem, D)
        e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
        reads = []
        for t in range(T):
            r = torch.einsum('bm,bmd->bd', self.Wq(e[:, t]), S)
            k = self.Wk(e_prev[:, t]); v = self.Wv(e[:, t])
            k_n = F.normalize(k, dim=-1)
            pred = torch.einsum('bmd,bm->bd', S, k_n)
            err = v - pred
            if self.beta_mode == 'one':
                beta = 1.0
            elif self.beta_mode == 'kaczmarz':
                beta = (1.0 / (k.norm(dim=-1, keepdim=True)**2 + 1e-6)).unsqueeze(-1)
            elif self.beta_mode == 'fixed':
                beta = self.beta_fixed
            elif self.beta_mode == 'ema':
                p = (k.detach().norm(dim=-1)**2).mean()
                if self.training:
                    if not bool(self.ema_ready):
                        self.power_ema.copy_(p); self.ema_ready.fill_(True)
                    else:
                        self.power_ema.mul_(0.99).add_(0.01 * p)
                beta = self.mu / (self.power_ema + 1e-6)
            S = S + beta * k_n.unsqueeze(-1) * err.unsqueeze(1)
            if self.training and self.beta_mode in ('fixed', 'ema'):
                with torch.no_grad():
                    b2 = (beta * k.norm(dim=-1)**2).reshape(-1)
                    self.b2_sum += b2.sum().item()
                    self.b2_cnt += b2.numel()
                    self.b2_over += (b2 > 2).sum().item()
                    self.b2_vals.append(b2.detach().cpu())
            reads.append(r)
        return torch.stack(reads, 1)

class Model(nn.Module):
    def __init__(self, V, d=64, d_mem=16, beta_mode='one', mu=1.0, beta_fixed=1.0):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(2048, d)
        self.mem = Mem(d, d_mem, beta_mode, mu, beta_fixed)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        B, T = x.shape
        e = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None]
        return self.head(self.mem(e))

def make_mqar(B, n_pairs, n_dist, V):
    L = 2*n_pairs + n_dist + n_pairs
    x = torch.zeros(B, L, dtype=torch.long); tgt = torch.zeros(B, n_pairs, dtype=torch.long)
    for b in range(B):
        keys = torch.randint(0, V, (n_pairs,)); vals = torch.randint(0, V, (n_pairs,))
        idx = 0
        for i in range(n_pairs):
            x[b, idx] = keys[i]; idx += 1; x[b, idx] = vals[i]; idx += 1
        for _ in range(n_dist): x[b, idx] = random.randint(0, V-1); idx += 1
        perm = torch.randperm(n_pairs)
        for i in range(n_pairs):
            x[b, idx] = keys[perm[i]]; tgt[b, i] = vals[perm[i]]; idx += 1
    return x, tgt, 2*n_pairs + n_dist

def train_model(beta_mode, d_mem, seed, mu=1.0, beta_fixed=1.0, steps=1500, lr=3e-4, B=32, V=64):
    torch.manual_seed(seed); random.seed(seed)
    m = Model(V, d=64, d_mem=d_mem, beta_mode=beta_mode, mu=mu, beta_fixed=beta_fixed)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for step in range(steps):
        m.train()
        x, tgt, qs = make_mqar(B, 8, 32, V)
        out = m(x)
        loss = F.cross_entropy(out[:, qs:qs+8].reshape(-1, V), tgt.reshape(-1))
        if not torch.isfinite(loss):
            return None
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m

def evaluate(m, V=64, measure_E=True):
    m.eval()
    accs = []; evals = []
    with torch.no_grad():
        for bi in range(20):
            x, tgt, qs = make_mqar(128, 8, 32, V)
            out = m(x)
            pred = out[:, qs:qs+8].argmax(-1)
            accs.append((pred == tgt).float().mean().item())
            if measure_E and bi < 4:
                e = m.emb(x) + m.pos(torch.arange(x.size(1))[None])
                e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
                k = m.mem.Wk(e_prev[:, 1:])
                evals.append((k.norm(dim=-1)**2).reshape(-1))
    E_post = torch.cat(evals).mean().item() if evals else None
    return sum(accs)/len(accs), E_post

def pilot_E(m, V=64):
    m.eval()
    with torch.no_grad():
        vals = []
        for _ in range(4):
            x, _, _ = make_mqar(256, 8, 32, V)
            e = m.emb(x) + m.pos(torch.arange(x.size(1))[None])
            e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
            k = m.mem.Wk(e_prev[:, 1:])
            vals.append((k.norm(dim=-1)**2).reshape(-1))
        v = torch.cat(vals)
        return v.mean().item(), v.quantile(0.95).item()

def stats_from(m):
    if m.mem.b2_cnt == 0:
        return dict(b2_mean=float('nan'), b2_p95=float('nan'), over2=float('nan'), ema=m.mem.power_ema.item())
    vals = torch.cat(m.mem.b2_vals)
    return dict(b2_mean=vals.mean().item(), b2_p95=vals.quantile(0.95).item(),
                over2=m.mem.b2_over/m.mem.b2_cnt, ema=m.mem.power_ema.item())

def run_fixed():
    print(f'{"d_mem":>6s} {"mu":>5s} {"seed":>5s} {"acc":>8s} {"beta":>9s} {"E_ref":>8s} {"E_post":>8s} {"bE_post":>8s} {"b2_mean":>8s} {"b2_p95":>8s} {"over2":>7s}', flush=True)
    for d_mem in [16, 32]:
        pilot = train_model('kaczmarz', d_mem, 0)
        E_ref, E_ref_p95 = pilot_E(pilot)
        pacc, _ = evaluate(pilot, measure_E=False)
        print(f'PILOT d_mem={d_mem} E_ref={E_ref:.4f} p95={E_ref_p95:.4f} acc={pacc:.4f}', flush=True)
        for mu in [0.5, 1.0, 2.0]:
            beta = mu / E_ref
            for seed in [0, 1, 2]:
                m = train_model('fixed', d_mem, seed, beta_fixed=beta)
                if m is None:
                    print(f'{d_mem:6d} {mu:5.1f} {seed:5d} {"NaN":>8s}', flush=True)
                    continue
                acc, E_post = evaluate(m)
                st = stats_from(m)
                print(f'{d_mem:6d} {mu:5.1f} {seed:5d} {acc:8.4f} {beta:9.4f} {E_ref:8.3f} {E_post:8.3f} {beta*E_post:8.3f} {st["b2_mean"]:8.3f} {st["b2_p95"]:8.3f} {st["over2"]:7.4f}', flush=True)

def run_ema():
    print(f'{"d_mem":>6s} {"mu":>5s} {"seed":>5s} {"acc":>8s} {"E_post":>8s} {"b2_mean":>8s} {"b2_p95":>8s} {"over2":>7s} {"ema":>8s}', flush=True)
    for d_mem in [16, 32]:
        for mu in [1.0, 2.0]:
            for seed in [0, 1, 2]:
                m = train_model('ema', d_mem, seed, mu=mu)
                if m is None:
                    print(f'{d_mem:6d} {mu:5.1f} {seed:5d} {"NaN":>8s}', flush=True)
                    continue
                acc, E_post = evaluate(m)
                st = stats_from(m)
                print(f'{d_mem:6d} {mu:5.1f} {seed:5d} {acc:8.4f} {E_post:8.3f} {st["b2_mean"]:8.3f} {st["b2_p95"]:8.3f} {st["over2"]:7.4f} {st["ema"]:8.4f}', flush=True)

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'fixed'
    if mode == 'fixed':
        run_fixed()
    elif mode == 'ema':
        run_ema()
    else:
        print('unknown mode', mode)
