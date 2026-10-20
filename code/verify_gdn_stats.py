import numpy as np
from scipy import stats

gdn = np.array([0.5285, 0.5599, 0.2983, 0.6119, 0.0160, 0.0165, 0.0162, 0.6611, 0.5193, 0.4368])
delta = np.array([0.9027, 0.8941, 0.8975, 0.8970, 0.9126, 0.8961, 0.8903, 0.8990, 0.8931, 0.8864])


def desc(x, name):
    print(f'{name}: n={len(x)} mean={x.mean():.6f} std={x.std(ddof=1):.6f} '
          f'median={np.median(x):.6f} min={x.min():.6f} max={x.max():.6f} '
          f'var={x.var(ddof=1):.6e}')


desc(gdn, 'GDN')
desc(delta, 'Delta')

# Brown-Forsythe: absolute deviations from group medians, one-way ANOVA
ag = np.abs(gdn - np.median(gdn))
ad = np.abs(delta - np.median(delta))
F, p = stats.f_oneway(ag, ad)
print(f'Brown-Forsythe (median-centered): F={F:.4f}, p={p:.6g}')

F2, p2 = stats.levene(gdn, delta, center='median')
print(f'Levene center=median: F={F2:.4f}, p={p2:.6g}')

for thresh in [0.45, 0.02]:
    a = int((gdn < thresh).sum())
    b = len(gdn) - a
    c = int((delta < thresh).sum())
    d = len(delta) - c
    odds, p = stats.fisher_exact([[a, b], [c, d]], alternative='two-sided')
    print(f'Fisher <{thresh}: GDN {a}/{len(gdn)}, Delta {c}/{len(delta)}, '
          f'table=[[{a},{b}],[{c},{d}]], odds={odds:.4g}, p={p:.6g}')

t, p = stats.ttest_ind(gdn, delta, equal_var=False)
print(f'Welch t={t:.4f}, p={p:.6g}')

u, p = stats.mannwhitneyu(gdn, delta, alternative='two-sided')
print(f'Mann-Whitney U={u:.4f}, p={p:.6g}')

print(f'median gap (Delta-GDN)={np.median(delta) - np.median(gdn):.6f}')
print(f'variance ratio GDN/Delta={gdn.var(ddof=1) / delta.var(ddof=1):.4f}')
# Cohen's d is intentionally not reported: the GDN distribution is trimodal
# and the Delta distribution is nearly degenerate, so the pooled-variance and
# same-shape premises of d do not hold.
