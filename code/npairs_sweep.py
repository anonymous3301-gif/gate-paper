import torch, torch.nn as nn, torch.nn.functional as F
import random, numpy as np
torch.set_num_threads(2)

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
    def __init__(self, V, d=64, d_mem=32):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(2048, d)
        self.mem = DeltaMem(d, d_mem); self.head = nn.Linear(d, V)
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

def train_measure(n_pairs, seed=0, d_mem=32, steps=1500, V=64, B=32):
    torch.manual_seed(seed); random.seed(seed)
    m = Model(V, d=64, d_mem=d_mem)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    for step in range(steps):
        m.train()
        x, tgt, qs = make_mqar(B, n_pairs, 32, V)
        out = m(x)
        loss = F.cross_entropy(out[:, qs:qs+n_pairs].reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad():
        accs = []
        for _ in range(20):
            x, tgt, qs = make_mqar(128, n_pairs, 32, V)
            out = m(x)
            pred = out[:, qs:qs+n_pairs].argmax(-1)
            accs.append((pred == tgt).float().mean().item())
        acc = sum(accs)/len(accs)
        x, tgt, qs = make_mqar(128, n_pairs, 32, V)
        e = m.emb(x) + m.pos(torch.arange(x.size(1)))[None]
        e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
        ks = torch.stack([m.mem.Wk(e_prev[:, 2*i+1]) for i in range(n_pairs)])
        K = ks.reshape(-1, d_mem)
        # E[(k.k')^2] and k_norm
        ki = ks.reshape(-1, d_mem)
        kj = ks.reshape(-1, d_mem)
        # pairwise cos^2
        K_hat = F.normalize(ki, dim=-1)
        cos_mat = K_hat @ K_hat.T
        mask = ~torch.eye(cos_mat.size(0), dtype=torch.bool)
        E_cos2 = (cos_mat[mask]**2).mean().item()
        k_norms = ki.norm(dim=-1)
        E_norm4 = (k_norms**4).mean().item()
        E_dot_sq = E_norm4 * E_cos2
        mean_abs_cos = cos_mat[mask].abs().mean().item()
        k_norm_mean = k_norms.mean().item()
        sv = torch.linalg.svdvals(K)
        er = ((sv**2).sum() / (sv[0]**2)).item()
        G = K.T @ K
        kappa = (sv[0]/sv[-1]).item() if sv[-1] > 0 else float('inf')
        sigma_min = sv[-1].item()
    return acc, E_dot_sq, E_cos2, mean_abs_cos, k_norm_mean, er, kappa, sigma_min

if __name__ == '__main__':
    print(f'{"n_pairs":>8s} {"acc":>8s} {"E[(k.k)^2]":>12s} {"E[cos2]":>9s} {"mean|cos|":>10s} {"k_norm":>8s} {"eff_rank":>9s} {"kappa":>12s} {"sig_min":>10s}')
    print("-" * 95)
    for n_pairs in [4, 8, 16, 32]:
        acc, ed, ec2, mac, kn, er, kappa, sigmin = train_measure(n_pairs)
        print(f'{n_pairs:8d} {acc:8.4f} {ed:12.4f} {ec2:9.4f} {mac:10.4f} {kn:8.2f} {er:9.2f} {kappa:12.2e} {sigmin:10.2e}', flush=True)