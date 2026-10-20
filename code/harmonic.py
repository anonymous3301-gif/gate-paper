import torch
import beta_power as bp

def stats(v):
    inv = 1.0 / (v + 1e-6)
    return dict(E2=v.mean().item(), Einv=inv.mean().item(), inv_med=inv.median().item(),
                cv=(inv.std() / inv.mean()).item(), p5=inv.quantile(0.05).item(),
                p95=inv.quantile(0.95).item(), n=v.numel())

def measure(m, V=64):
    m.eval()
    with torch.no_grad():
        vals_all = []; vals_key = []
        for _ in range(4):
            x, _, _ = bp.make_mqar(256, 8, 32, V)
            e = m.emb(x) + m.pos(torch.arange(x.size(1))[None])
            e_prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)
            k = m.mem.Wk(e_prev[:, 1:])
            vals_all.append((k.norm(dim=-1) ** 2).reshape(-1))
            kk = torch.stack([m.mem.Wk(e_prev[:, 2*i+1]) for i in range(8)])
            vals_key.append((kk.norm(dim=-1) ** 2).reshape(-1))
        a = torch.cat(vals_all); b = torch.cat(vals_key)
        return stats(a), stats(b)

if __name__ == '__main__':
    print(f'{"d_mem":>6s} {"acc":>8s} {"pos":>5s} {"E[k2]":>9s} {"E[1/k2]":>9s} {"med(1/k2)":>10s} {"CV":>7s} {"p5":>7s} {"p95":>7s}', flush=True)
    for d_mem in [16, 32, 64, 128]:
        m = bp.train_model('kaczmarz', d_mem, 0)
        acc, _ = bp.evaluate(m, measure_E=False)
        a, b = measure(m)
        for name, s in [('all', a), ('key', b)]:
            print(f'{d_mem:6d} {acc:8.4f} {name:>5s} {s["E2"]:9.2f} {s["Einv"]:9.4f} {s["inv_med"]:10.4f} {s["cv"]:7.3f} {s["p5"]:7.4f} {s["p95"]:7.4f}', flush=True)
