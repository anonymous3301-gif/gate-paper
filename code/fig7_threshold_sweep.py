import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

gdn = np.array([0.5285, 0.5599, 0.2983, 0.6119, 0.0160, 0.0165, 0.0162, 0.6611, 0.5193, 0.4368])
delta = np.array([0.9027, 0.8941, 0.8975, 0.8970, 0.9126, 0.8961, 0.8903, 0.8990, 0.8931, 0.8864])


def fisher_p(t):
    gf = int((gdn < t).sum())
    df = int((delta < t).sum())
    table = [[gf, len(gdn) - gf], [df, len(delta) - df]]
    _, p = stats.fisher_exact(table, alternative='two-sided')
    return gf, df, p


thresholds = np.linspace(0.02, 0.90, 220)
p_values = []
gdn_fails = []
delta_fails = []
for t in thresholds:
    gf, df, p = fisher_p(t)
    p_values.append(p)
    gdn_fails.append(gf)
    delta_fails.append(df)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Panel (a): p value vs threshold
ax = axes[0]
ax.axvspan(0.44, 0.897, color='#27AE60', alpha=0.07, zorder=0)
ax.plot(thresholds, p_values, color='#2C3E50', lw=2)
ax.axhline(0.05, color='#C0392B', lw=1, ls='--', label='p = 0.05')
ax.axvline(0.45, color='#2980B9', lw=1, ls=':', label='pre-specified t = 0.45')
ax.set_xlabel('failure threshold t (accuracy)')
ax.set_ylabel('Fisher exact p (two-sided)')
ax.set_yscale('log')
ax.set_title('(a) Threshold sensitivity of Fisher exact test', loc='left', fontweight='bold')
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.25, lw=0.5)

# Panel (b): number of failures vs threshold
ax = axes[1]
ax.axvspan(0.44, 0.897, color='#27AE60', alpha=0.07, zorder=0)
ax.step(thresholds, gdn_fails, where='post', color='#C0392B', lw=2, label='GDN')
ax.step(thresholds, delta_fails, where='post', color='#2980B9', lw=2, label='Delta')
ax.axvline(0.45, color='0.4', lw=1, ls=':', label='pre-specified t = 0.45')
ax.set_xlabel('failure threshold t (accuracy)')
ax.set_ylabel('number of failed seeds')
ax.set_title('(b) Failure counts', loc='left', fontweight='bold')
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.25, lw=0.5)

plt.tight_layout()
plt.savefig('fig7_threshold_sweep.png', dpi=300, bbox_inches='tight')
plt.savefig('fig7_threshold_sweep.pdf', bbox_inches='tight')
plt.show()

# Print key thresholds
print('=== key thresholds ===')
for t in [0.02, 0.10, 0.29, 0.30, 0.4368, 0.44, 0.45, 0.51, 0.5193, 0.52,
          0.66, 0.6611, 0.67, 0.88, 0.886, 0.8864, 0.89, 0.8970, 0.8975]:
    gf, df, p = fisher_p(t)
    print(f't={t:.4f}  GDN fail={gf}/10  Delta fail={df}/10  p={p:.6f}')

# Locate the full contiguous significant region on the step function.
# Evaluate just below/at/above every observed accuracy value.
vals = np.sort(np.unique(np.concatenate([gdn, delta])))
grid = np.concatenate([vals - 1e-9, vals, vals + 1e-9])
grid = grid[(grid > 0.02) & (grid < 0.95)]
ps = np.array([fisher_p(t)[2] for t in grid])
sig = grid[ps < 0.05]
print('=== full p<0.05 region (grid search) ===')
if len(sig):
    print(f'min t = {sig.min():.8f}, max t = {sig.max():.8f}')
else:
    print('none')

# Check the clean reported band densely.
band = np.linspace(0.44, 0.897, 5000)
band_p = np.array([fisher_p(t)[2] for t in band])
print(f'=== [0.44, 0.897] ===')
print(f'min p = {band_p.min():.6e}, max p = {band_p.max():.6f}, all <0.05: {bool((band_p < 0.05).all())}')
