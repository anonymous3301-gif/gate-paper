import torch, torch.nn as nn, torch.nn.functional as F
import random, numpy as np
torch.set_num_threads(2)

class DeltaDecayMem(nn.Module):
    def __init__(self, d, d_mem):
        super().__init__()
        self.d_mem = d_mem
        self.Wk = nn.Linear(d, d_mem, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d_mem, bias=False)
        self.log_alpha = nn.Parameter(torch.tensor([3.0]))
    def forward(self, e):
        B, T, D = e.shape
        S = e.new_zeros(B, self.d_mem, D)
        e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
        alpha = torch.sigmoid(self.log_alpha)
        reads = []
        for t in range(T):
            r = torch.einsum('bm,bmd->bd', self.Wq(e[:, t]), S)
            k = self.Wk(e_prev[:, t]); v = self.Wv(e[:, t])
            k_n = F.normalize(k, dim=-1)
            pred = torch.einsum('bmd,bm->bd', S, k_n)
            S = alpha * S + k_n.unsqueeze(-1) * (v - pred).unsqueeze(1)
            reads.append(r)
        return torch.stack(reads, 1)

class Model(nn.Module):
    def __init__(self, V, d=64, d_mem=64):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(2048, d)
        self.mem = DeltaDecayMem(d, d_mem); self.head = nn.Linear(d, V)
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

def make_mqar_junk(B, n_pairs, n_dist, J, V):
    L = 2*n_pairs + n_dist + J + n_pairs
    x = torch.zeros(B, L, dtype=torch.long); tgt = torch.zeros(B, n_pairs, dtype=torch.long)
    for b in range(B):
        keys = torch.randint(0, V, (n_pairs,)); vals = torch.randint(0, V, (n_pairs,))
        idx = 0
        for i in range(n_pairs):
            x[b, idx] = keys[i]; idx += 1; x[b, idx] = vals[i]; idx += 1
        for _ in range(n_dist + J): x[b, idx] = random.randint(0, V-1); idx += 1
        perm = torch.randperm(n_pairs)
        for i in range(n_pairs):
            x[b, idx] = keys[perm[i]]; tgt[b, i] = vals[perm[i]]; idx += 1
    return x, tgt, 2*n_pairs + n_dist + J

def train_eval(seed, d_mem=64, steps=1500, lr=1e-3, B=32, V=64):
    torch.manual_seed(seed); random.seed(seed)
    m = Model(V, d=64, d_mem=d_mem)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for step in range(steps):
        m.train()
        x, tgt, qs = make_mqar(B, 8, 32, V)
        out = m(x)
        loss = F.cross_entropy(out[:, qs:qs+8].reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    res = {}
    with torch.no_grad():
        for J in [0, 8, 16, 32]:
            accs = []
            for _ in range(20):
                x, tgt, qs = make_mqar_junk(128, 8, 32, J, V)
                out = m(x)
                pred = out[:, qs:qs+8].argmax(-1)
                accs.append((pred == tgt).float().mean().item())
            res[J] = sum(accs)/len(accs)
    alpha = torch.sigmoid(m.mem.log_alpha).item()
    return res, alpha

if __name__ == '__main__':
    print(f'{"seed":>4s} {"alpha":>7s} {"J=0":>8s} {"J=8":>8s} {"J=16":>8s} {"J=32":>8s}')
    all_res = {J: [] for J in [0,8,16,32]}
    for seed in [0,1,2]:
        res, alpha = train_eval(seed)
        for J in all_res: all_res[J].append(res[J])
        print(f'{seed:4d} {alpha:7.3f} ' + ' '.join(f'{res[J]:8.4f}' for J in [0,8,16,32]), flush=True)
    print(f'{"mean":>4s} {"":>7s} ' + ' '.join(f'{np.mean(all_res[J]):8.4f}' for J in [0,8,16,32]))
    print(f'{"std":>4s} {"":>7s} ' + ' '.join(f'{np.std(all_res[J]):8.4f}' for J in [0,8,16,32]))