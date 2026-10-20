import torch, torch.nn as nn, torch.nn.functional as F
import random, statistics


def make_mqar(B, n_pairs, n_dist, V, device):
    L = 2*n_pairs + n_dist + n_pairs
    x = torch.zeros(B, L, dtype=torch.long, device=device)
    targets = torch.zeros(B, n_pairs, dtype=torch.long, device=device)
    for b in range(B):
        keys = torch.randint(0, V, (n_pairs,), device=device)
        vals = torch.randint(0, V, (n_pairs,), device=device)
        idx = 0
        for i in range(n_pairs):
            x[b, idx] = keys[i]; idx += 1
            x[b, idx] = vals[i]; idx += 1
        for _ in range(n_dist):
            x[b, idx] = random.randint(0, V-1); idx += 1
        perm = torch.randperm(n_pairs, device=device)
        for i in range(n_pairs):
            x[b, idx] = keys[perm[i]]
            targets[b, i] = vals[perm[i]]
            idx += 1
    return x, targets, 2*n_pairs + n_dist


class Mem(nn.Module):
    def __init__(self, d, d_mem, mode):
        super().__init__()
        self.mode = mode; self.d_mem = d_mem
        self.Wk = nn.Linear(d, d_mem); self.Wv = nn.Linear(d, d); self.Wq = nn.Linear(d, d_mem)
    def forward(self, e):
        B, T, _ = e.shape
        S = e.new_zeros(B, self.d_mem, e.size(-1))
        e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
        reads = []
        for t in range(T):
            r = torch.einsum('bm,bmd->bd', self.Wq(e[:, t]), S)
            k = self.Wk(e_prev[:, t]); v = self.Wv(e[:, t])
            if self.mode == 'hebbian':
                S = S + k.unsqueeze(-1) * v.unsqueeze(1)
            else:
                k_n = F.normalize(k, dim=-1)
                pred = torch.einsum('bmd,bm->bd', S, k_n)
                err = v - pred
                S = S + k_n.unsqueeze(-1) * err.unsqueeze(1)
            reads.append(r)
        return torch.stack(reads, 1)


class Model(nn.Module):
    def __init__(self, V, d=64, d_mem=32, mode='hebbian'):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(2048, d)
        self.mem = Mem(d, d_mem, mode)
        self.head = nn.Linear(d*2, V)
    def forward(self, x):
        B, T = x.shape
        e = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None]
        reads = self.mem(e)
        return self.head(torch.cat([e, reads], -1))


def train(mode, seed, n_pairs=8, steps=1500, V=64, B=32):
    torch.manual_seed(seed)
    model = Model(V, mode=mode)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(steps):
        model.train()
        x, tgt, qs = make_mqar(B, n_pairs, 32, V, 'cpu')
        out = model(x)
        loss = F.cross_entropy(out[:, qs:qs+n_pairs].reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, seed, n_pairs=8, V=64):
    torch.manual_seed(seed+1000)
    model.eval()
    with torch.no_grad():
        accs = []
        for _ in range(20):
            x, tgt, qs = make_mqar(128, n_pairs, 32, V, 'cpu')
            out = model(x)
            pred = out[:, qs:qs+n_pairs].argmax(-1)
            accs.append((pred == tgt).float().mean().item())
    return sum(accs)/len(accs)


if __name__ == '__main__':
    print("=== 5-seed reproducibility (key cells) ===")
    print(f"{'mode':10s} {'seed0':>8s} {'seed1':>8s} {'seed2':>8s} {'seed3':>8s} {'seed4':>8s} {'mean':>8s} {'std':>8s}")
    print("-" * 80)
    for mode in ['hebbian', 'delta']:
        accs = []
        for seed in range(5):
            m = train(mode, seed=seed)
            a = evaluate(m, seed=seed)
            accs.append(a)
        mean = statistics.mean(accs)
        std = statistics.stdev(accs)
        print(f"{mode:10s} " + " ".join(f"{a:8.4f}" for a in accs) + f" {mean:8.4f} {std:8.4f}")