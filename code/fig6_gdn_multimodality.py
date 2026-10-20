import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

gdn = np.array([0.5285, 0.5599, 0.2983, 0.6119, 0.0160, 0.0165, 0.0162, 0.6611, 0.5193, 0.4368])
delta = np.array([0.9027, 0.8941, 0.8975, 0.8970, 0.9126, 0.8961, 0.8903, 0.8990, 0.8931, 0.8864])

fig, ax = plt.subplots(figsize=(6.5, 3.8))

# jitter for visibility
rng = np.random.default_rng(0)
x_gdn = 1 + rng.uniform(-0.12, 0.12, len(gdn))
x_delta = 2 + rng.uniform(-0.12, 0.12, len(delta))

ax.scatter(x_gdn, gdn, s=60, color='#C0392B', alpha=0.85, edgecolor='white', lw=0.8, zorder=3, label='GDN')
ax.scatter(x_delta, delta, s=60, color='#2980B9', alpha=0.85, edgecolor='white', lw=0.8, zorder=3, label='Delta (no decay)')

# pre-specified failure threshold
ax.axhline(0.45, color='0.3', lw=1.2, ls='--', zorder=1)

# complete-collapse threshold
ax.axhline(0.02, color='0.6', lw=0.8, ls=':', zorder=1)

# means
ax.scatter([1], [gdn.mean()], marker='_', s=600, color='#C0392B', lw=3, zorder=4)
ax.scatter([2], [delta.mean()], marker='_', s=600, color='#2980B9', lw=3, zorder=4)

ax.set_xticks([1, 2])
ax.set_xticklabels(['GDN\n(decay)', 'Delta\n(no decay)'])
ax.set_ylabel('MQAR accuracy (J = 8)')
ax.set_ylim(-0.02, 1.0)
ax.set_xlim(0.5, 2.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# annotation: Brown-Forsythe and Fisher
ax.text(0.98, 0.06, 'Brown–Forsythe: p = 0.0022\nFisher (<0.45): 5/10 vs 0/10, p = 0.0325\nMedian gap = 0.419\nVariance ratio = 1325×',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9, zorder=5,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.8', lw=0.6))

handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B',
           markersize=7, label='GDN (decay)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980B9',
           markersize=7, label='Delta (no decay)'),
    Line2D([0], [0], color='0.3', lw=1.2, ls='--',
           label='pre-specified threshold (0.45)'),
    Line2D([0], [0], color='0.6', lw=0.8, ls=':',
           label='chance (1/V = 0.016)'),
]
ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=9,
          handlelength=2.2, labelspacing=0.4)

plt.tight_layout()
plt.savefig('fig6_gdn_multimodality.png', dpi=300, bbox_inches='tight')
plt.savefig('fig6_gdn_multimodality.pdf', bbox_inches='tight')
plt.show()
