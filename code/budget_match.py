import torch, torch.nn as nn, torch.nn.functional as F
import random, numpy as np, json, time
torch.set_num_threads(2)

# ============================================================
# DeltaMem with configurable (key_norm, beta_mode, beta_value)
# ============================================================
class DeltaMem(nn.Module):
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
                beta = (1.0 / (k.norm(dim=-1, keepdim=True)**2 + 1e-6)).unsqueeze(-1)
            else:
                raise ValueError(self.beta_mode)
            S = S + beta * k_use.unsqueeze(-1) * err.unsqueeze(1)
            reads.append(r)
        return torch.stack(reads, 1)

class Model(nn.Module):
    def __init__(self, V, d=64, d_mem=16, key_norm=True, beta_mode='one', beta_value=1.0):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(2048, d)
        self.mem = DeltaMem(d, d_mem, key_norm, beta_mode, beta_value)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        B, T = x.shape
        e = self.emb(x) + self.pos(torch.arange(T))[None]
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

def train_eval(cfg, seed, n_pairs=8, V=64, B=32):
    torch.manual_seed(seed); random.seed(seed)
    m = Model(V, d=64, d_mem=cfg['d_mem'], key_norm=cfg['key_norm'],
              beta_mode=cfg['beta_mode'], beta_value=cfg['beta_value'])
    opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'])
    losses_log = []
    for step in range(cfg['steps']):
        m.train()
        x, tgt, qs = make_mqar(B, n_pairs, 32, V)
        out = m(x)
        loss = F.cross_entropy(out[:, qs:qs+n_pairs].reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % max(1, cfg['steps']//5) == 0 or step == cfg['steps']-1:
            losses_log.append((step, loss.item()))
    m.eval()
    with torch.no_grad():
        accs = []
        for _ in range(20):
            x, tgt, qs = make_mqar(128, n_pairs, 32, V)
            out = m(x)
            pred = out[:, qs:qs+n_pairs].argmax(-1)
            accs.append((pred == tgt).float().mean().item())
    return float(np.mean(accs)), losses_log

# ============================================================
# 4-stage ladder × 2 budgets × 3 seeds, d_mem=16
# ============================================================
CONFIGS = {
    'stage0_raw':      dict(d_mem=16, key_norm=False, beta_mode='one',      beta_value=1.0),
    'stage1_norm':     dict(d_mem=16, key_norm=True,  beta_mode='one',      beta_value=1.0),
    'stage2_fixed':    dict(d_mem=16, key_norm=True,  beta_mode='fixed',    beta_value=0.125),
    'stage3_kaczmarz': dict(d_mem=16, key_norm=True,  beta_mode='kaczmarz', beta_value=1.0),
}
BUDGETS = [1500, 3000]
SEEDS = [0, 1, 2]
LR = 3e-4

results = []
t0 = time.time()

for budget in BUDGETS:
    for name, cfg_base in CONFIGS.items():
        cfg = dict(cfg_base)
        cfg['steps'] = budget
        cfg['lr'] = LR
        for seed in SEEDS:
            acc, losses = train_eval(cfg, seed)
            results.append({
                'stage': name, 'budget': budget, 'seed': seed,
                'acc': acc, 'losses': losses,
                'key_norm': cfg['key_norm'],
                'beta_mode': cfg['beta_mode'],
                'beta_value': cfg['beta_value'],
            })
            print(f'{name:18s}  budget={budget:5d}  seed={seed}  acc={acc:.4f}  '
                  f'final_loss={losses[-1][1]:.4f}', flush=True)

# ============================================================
# Summary table (markdown + json)
# ============================================================
print('\n=== Summary: acc mean ± std over 3 seeds ===')
print(f'{"stage":18s} {"budget=1500":>18s} {"budget=3000":>18s} {"Δ(3000-1500)":>16s}')
summary = {}
for name in CONFIGS:
    row = {}
    for budget in BUDGETS:
        accs = [r['acc'] for r in results if r['stage']==name and r['budget']==budget]
        row[budget] = (float(np.mean(accs)), float(np.std(accs, ddof=1)))
    d = row[3000][0] - row[1500][0]
    summary[name] = {'1500': row[1500], '3000': row[3000], 'delta': d}
    print(f'{name:18s} {row[1500][0]:.4f}±{row[1500][1]:.4f}    '
          f'{row[3000][0]:.4f}±{row[3000][1]:.4f}    {d:+.4f}')

# Write the JSON before any human-readable comparison prints. The original
# run completed all 24 models but crashed on a Windows cp932 UnicodeEncodeError
# in the em-dash print below, so the JSON was never written.
with open('budget_match_results.json', 'w', encoding='utf-8') as f:
    json.dump({'results': results, 'summary': summary}, f, indent=2)
print(f'\nSaved: budget_match_results.json  (elapsed {time.time()-t0:.0f}s)')

# ============================================================
# Key comparisons (for Section 3-4 attribution)
# ============================================================
print('\n=== Key comparisons ===')
print('1. stage1_norm (beta=1, 1500) vs beta_power mu=1.0 (0.8513) -- budget-confound check')
print(f'   stage1_norm @1500 = {summary["stage1_norm"]["1500"][0]:.4f}')
print(f'   stage1_norm @3000 = {summary["stage1_norm"]["3000"][0]:.4f}')
print('   -> if it approaches 0.85 at 3000 steps, budget confound; otherwise implementation difference')
print()
print('2. erase-strength contribution (stage1 -> stage2, per budget)')
for b in BUDGETS:
    s1 = summary['stage1_norm'][str(b)][0]
    s2 = summary['stage2_fixed'][str(b)][0]
    print(f'   budget={b}: {s1:.4f} -> {s2:.4f}  (delta={s2-s1:+.4f}, relative {(s2-s1)/max(s1,1e-6)*100:+.1f}%)')
print()
print('3. gating contribution (stage2 -> stage3, per budget)')
for b in BUDGETS:
    s2 = summary['stage2_fixed'][str(b)][0]
    s3 = summary['stage3_kaczmarz'][str(b)][0]
    print(f'   budget={b}: {s2:.4f} -> {s3:.4f}  (delta={s3-s2:+.4f}, relative {(s3-s2)/max(s2,1e-6)*100:+.1f}%)')
