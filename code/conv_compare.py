import torch, torch.nn as nn, torch.nn.functional as F
import random, numpy as np
torch.set_num_threads(2)

class CausalConv(nn.Module):
    def __init__(self, d, k=4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, padding=k-1, groups=d)
        self.k = k
    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.conv(h)[:, :, :x.size(1)]
        return x + h.transpose(1, 2)

class HebbMem(nn.Module):
    def __init__(self, d, d_mem):
        super().__init__()
        self.d_mem = d_mem
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
            S = S + k.unsqueeze(-1) * v.unsqueeze(1)
            reads.append(r)
        return torch.stack(reads, 1)

class DeltaMem(nn.Module):
    def __init__(self, d, d_mem):
        super().__init__()
        self.d_mem = d_mem
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
            k_n = F.normalize(k, dim=-1)
            pred = torch.einsum('bmd,bm->bd', S, k_n)
            S = S + k_n.unsqueeze(-1) * (v - pred).unsqueeze(1)
            reads.append(r)
        return torch.stack(reads, 1)

class Model(nn.Module):
    def __init__(self, V, d=64, d_mem=32, mode='delta', use_conv=False):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(2048, d)
        self.conv = CausalConv(d) if use_conv else None
        mem_cls = DeltaMem if mode == 'delta' else HebbMem
        self.mem = mem_cls(d, d_mem); self.head = nn.Linear(d, V)
    def forward(self, x):
        B, T = x.shape
        e = self.emb(x) + self.pos(torch.arange(T))[None]
        if self.conv is not None: e = self.conv(e)
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

def train_eval(mode, use_conv, seed, d_mem=32, steps=1500, lr=3e-4, B=32, V=64):
    torch.manual_seed(seed); random.seed(seed)
    m = Model(V, d=64, d_mem=d_mem, mode=mode, use_conv=use_conv)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for step in range(steps):
        m.train()
        x, tgt, qs = make_mqar(B, 8, 32, V)
        out = m(x)
        loss = F.cross_entropy(out[:, qs:qs+8].reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad():
        accs = []
        for _ in range(20):
            x, tgt, qs = make_mqar(128, 8, 32, V)
            out = m(x)
            pred = out[:, qs:qs+8].argmax(-1)
            accs.append((pred == tgt).float().mean().item())
    return sum(accs)/len(accs)

if __name__ == '__main__':
    print(f'{"mode":>8s} {"conv":>5s} {"d_mem":>6s} {"seed0":>8s} {"seed1":>8s} {"seed2":>8s} {"mean":>8s}')
    for mode in ['hebbian', 'delta']:
        for use_conv in [False, True]:
            for d_mem in [16, 32]:
                accs = [train_eval(mode, use_conv, s, d_mem=d_mem) for s in [0,1,2]]
                print(f'{mode:>8s} {str(use_conv):>5s} {d_mem:>6d} ' + ' '.join(f'{a:8.4f}' for a in accs) + f' {np.mean(accs):8.4f}', flush=True)