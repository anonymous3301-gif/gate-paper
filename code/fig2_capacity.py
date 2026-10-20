"""Figure 2: capacity and fixed-step-size views of the normalized-regime sweep.

Data (all 1500 training steps, 3 seeds, normalized keys, MQAR with 8 pairs
and 32 distractors, evaluation batch 128 x 20 batches):

  panel (a) sources
    beta = 1          : beta_sweep_out.txt   ('one')
    beta = 1/sqrt(d)  : beta_sweep_out.txt   ('inv_sqrt_d')
    beta = 1/d        : beta_sweep_out.txt   ('inv_d')
    beta = 0.125      : budget_match stage 2 (d=16), beta_const_out.txt (d=32)
    Kaczmarz          : beta_sweep_out.txt   ('kaczmarz')

  panel (b) sources
    beta = 0.5 (d=16) : beta_const_out.txt
    beta = 0.25 (d=16): beta_sweep_out.txt ('inv_sqrt_d', d=16)
    beta = 0.0625 d=16: beta_sweep_out.txt ('inv_d', d=16)
    beta = 0.03125/0.1768 (d=32): beta_sweep_out.txt
    beta = 0.0625/0.125 (d=32)  : beta_const_out.txt
    beta = 1.0         : beta_sweep_out.txt ('one')

Every number below is copied from those files; nothing is recomputed from a
model. Error bars are the sample standard deviation (ddof=1) over 3 seeds.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = [16, 32]

# --- panel (a): accuracy vs capacity -----------------------------------------
series = {
    r'$\beta=1$': {
        16: [0.1849, 0.1935, 0.1972],
        32: [0.8649, 0.8000, 0.8408],
    },
    r'$\beta=1/\sqrt{d_{\mathrm{mem}}}$': {
        16: [0.8501, 0.8598, 0.8295],
        32: [0.9435, 0.9091, 0.9378],
    },
    r'$\beta=1/d_{\mathrm{mem}}$': {
        16: [0.8197, 0.8381, 0.7960],
        32: [0.9208, 0.9182, 0.9025],
    },
    r'fixed $\beta=0.125$': {
        16: [0.8686, 0.8754, 0.8489],
        32: [0.9426, 0.9366, 0.9365],
    },
    'Kaczmarz': {
        16: [0.9364, 0.9408, 0.9345],
        32: [0.9472, 0.9482, 0.9470],
    },
}

styles = {
    r'$\beta=1$': dict(color='#C0392B', marker='o', ls='-'),
    r'$\beta=1/\sqrt{d_{\mathrm{mem}}}$': dict(color='#E67E22', marker='s', ls='--'),
    r'$\beta=1/d_{\mathrm{mem}}$': dict(color='#8E44AD', marker='^', ls='--'),
    r'fixed $\beta=0.125$': dict(color='#2980B9', marker='D', ls='-'),
    'Kaczmarz': dict(color='#2C3E50', marker='*', ls='-'),
}

# --- panel (b): accuracy vs effective beta -----------------------------------
beta_sweep = {
    16: {0.0625: [0.8197, 0.8381, 0.7960],
         0.125: [0.8686, 0.8754, 0.8489],
         0.25:  [0.8501, 0.8598, 0.8295],
         0.5:   [0.5726, 0.3616, 0.5453],
         1.0:   [0.1849, 0.1935, 0.1972]},
    32: {0.03125: [0.9208, 0.9182, 0.9025],
         0.0625:  [0.9377, 0.9347, 0.9273],
         0.125:   [0.9426, 0.9366, 0.9365],
         0.17678: [0.9435, 0.9091, 0.9378],
         1.0:     [0.8649, 0.8000, 0.8408]},
}
kaczmarz_mean = {16: 0.9372, 32: 0.9475}


def mstd(v):
    a = np.asarray(v, dtype=float)
    return a.mean(), a.std(ddof=1)


fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))

# --- panel (a) ---------------------------------------------------------------
ax = axes[0]
ax.axhline(1.0 / 64, color='#95A5A6', lw=1, ls=':', zorder=0)
ax.text(15.85, 1.0 / 64 + 0.012, 'chance $1/64$', color='#7F8C8D', fontsize=8)
for name, per_d in series.items():
    st = styles[name]
    xs, ys, es = [], [], []
    for d in D:
        m, s = mstd(per_d[d])
        xs.append(d)
        ys.append(m)
        es.append(s)
    ax.errorbar(xs, ys, yerr=es, capsize=3, lw=1.8, ms=6, label=name, zorder=3, **st)
ax.set_xticks(D)
ax.set_xlim(13, 35)
ax.set_ylim(-0.01, 1.02)
ax.set_xlabel(r'$d_{\mathrm{mem}}$')
ax.set_ylabel('MQAR accuracy')
ax.set_title('(a) Accuracy vs capacity (1500 steps, 3 seeds)', loc='left', fontweight='bold')
ax.legend(fontsize=8, loc='center right', framealpha=0.95)
ax.grid(alpha=0.25, lw=0.6)

# --- panel (b) ---------------------------------------------------------------
ax = axes[1]
for d, color, marker in ((16, '#2980B9', 'D'), (32, '#C0392B', 'o')):
    xs = sorted(beta_sweep[d].keys())
    ys = [mstd(beta_sweep[d][b])[0] for b in xs]
    es = [mstd(beta_sweep[d][b])[1] for b in xs]
    ax.errorbar(xs, ys, yerr=es, capsize=3, lw=1.8, ms=6, color=color, marker=marker,
                label=fr'$d_{{\mathrm{{mem}}}}={d}$')
    ax.axhline(kaczmarz_mean[d], color=color, lw=0.9, ls=':', alpha=0.7)
    dy = -0.035 if d == 16 else 0.012
    va = 'top' if d == 16 else 'bottom'
    ax.text(0.0215, kaczmarz_mean[d] + dy, f'Kaczmarz, $d_{{\\mathrm{{mem}}}}$={d}',
            color=color, fontsize=7.5, va=va)
ax.axvspan(0.02, 2.0, color='#27AE60', alpha=0.05, zorder=0)
ax.axvline(2.0, color='#27AE60', lw=1.0)
ax.text(1.72, 0.28, 'LMS bound\n$\\beta<2$', color='#1E8449', fontsize=7.5, ha='right')
ax.axvline(0.125, color='#2C3E50', lw=0.9, ls='--', alpha=0.6)
ax.text(0.128, 0.055, 'fixed $\\beta=0.125$', color='#2C3E50', fontsize=7.5, rotation=90,
        va='bottom')
ax.set_xscale('log')
ax.set_xlim(0.02, 2.6)
ax.set_ylim(-0.01, 1.02)
ax.set_xticks([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0])
ax.set_xticklabels(['0.031', '0.0625', '0.125', '0.25', '0.5', '1.0', '2.0'])
ax.set_xlabel(r'effective $\beta$ (log scale)')
ax.set_ylabel('MQAR accuracy')
ax.set_title('(b) Accuracy vs effective $\\beta$ (1500 steps, 3 seeds)', loc='left',
             fontweight='bold')
ax.legend(fontsize=8, loc='center left', framealpha=0.95)
ax.grid(alpha=0.25, lw=0.6, which='both')

fig.tight_layout()
fig.savefig(r'C:\ai_exp\fig2_capacity.png', dpi=300, bbox_inches='tight')
fig.savefig(r'C:\ai_exp\fig2_capacity.pdf', bbox_inches='tight')
print('wrote fig2_capacity.png / .pdf')

# --- echo the plotted numbers so the log can be checked against the tables ----
for name, per_d in series.items():
    cells = []
    for d in D:
        m, s = mstd(per_d[d])
        cells.append(f'd={d}: {m:.4f} (sd {s:.4f})')
    print(f'panel (a)  {name:32s} ' + ' | '.join(cells))
for d in D:
    for b in sorted(beta_sweep[d].keys()):
        m, s = mstd(beta_sweep[d][b])
        print(f'panel (b)  d={d:2d} beta={b:<8g} {m:.4f} (sd {s:.4f})')
