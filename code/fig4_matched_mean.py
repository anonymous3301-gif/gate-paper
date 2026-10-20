import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

d_mem = np.array([16, 24, 32])

# ===== Data (mean, 95% CI half-width) =====
d_mem = np.array([16, 24, 32])

# Kaczmarz-trained: const_m drop (Δ = const_m − kacz, negative)
kacz_const_mean = np.array([-0.3798, -0.2266, -0.1611])
kacz_const_lo   = np.array([-0.418, -0.252, -0.198])
kacz_const_hi   = np.array([-0.341, -0.202, -0.124])

# Kaczmarz-trained: gate_m recovery (Δ = gate_m − kacz, near zero, signed)
kacz_gate_mean = np.array([+0.0004, -0.0011, +0.0006])
kacz_gate_ci   = np.array([0.0015, 0.0017, 0.0013])

# Fixed-trained: gate_m drop (Δ = gate_m − fixed, negative)
fixed_gate_mean = np.array([-0.0573, -0.0181, -0.0037])
fixed_gate_lo   = np.array([-0.0758, -0.0245, -0.0188])
fixed_gate_hi   = np.array([-0.0388, -0.0117, +0.0114])

# Fixed-trained: const_m change (Δ = const_m − fixed, near zero, signed)
fixed_const_mean = np.array([+0.0008, +0.0001, -0.0012])
fixed_const_ci   = np.array([0.0020, 0.0018, 0.0020])

# ===== Figure =====
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)

# --- Panel (a): Kaczmarz-trained model ---
ax = axes[0]
ax.axhline(0, color='0.5', lw=0.8, ls='--', zorder=0)

# const_m (mean-matched constant): the loss
ax.errorbar(d_mem, kacz_const_mean,
            yerr=[kacz_const_mean - kacz_const_lo,
                  kacz_const_hi - kacz_const_mean],
            marker='o', ms=7, lw=2, capsize=4,
            color='#C0392B', label=r'mean-matched const. $\beta$')

# gate_m (shape-matched gate): fully recovered
ax.errorbar(d_mem, kacz_gate_mean,
            yerr=[kacz_gate_ci, kacz_gate_ci],
            marker='s', ms=7, lw=2, capsize=4,
            color='#2980B9', label='shape-matched gate')

ax.set_xscale('log', base=2)
ax.set_xticks(d_mem)
ax.set_xticklabels([str(d) for d in d_mem])
ax.set_xlabel(r'$d_{\mathrm{mem}}$')
ax.set_ylabel(r'$\Delta$ accuracy (post-hoc swap $-$ baseline)')
ax.set_title('(a) Kaczmarz-trained model', loc='left', fontweight='bold')
ax.legend(loc='lower right', frameon=False, fontsize=10)
ax.set_ylim(-0.45, 0.05)
ax.grid(True, axis='y', alpha=0.25, lw=0.5)

# Annotate: shape-matched gate is invariant
ax.annotate('fully recovered\n$|\\Delta| \\leq 0.0011$',
            xy=(24, -0.001), xytext=(24, -0.13),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', lw=0.8, color='#2980B9'))

# --- Panel (b): Fixed-trained model ---
ax = axes[1]
ax.axhline(0, color='0.5', lw=0.8, ls='--', zorder=0)

ax.errorbar(d_mem, fixed_gate_mean,
            yerr=[fixed_gate_mean - fixed_gate_lo,
                  fixed_gate_hi - fixed_gate_mean],
            marker='s', ms=7, lw=2, capsize=4,
            color='#2980B9', label='shape-matched gate')

ax.errorbar(d_mem, fixed_const_mean,
            yerr=[fixed_const_ci, fixed_const_ci],
            marker='o', ms=7, lw=2, capsize=4,
            color='#C0392B', label=r'mean-matched const. $\beta$')

ax.set_xscale('log', base=2)
ax.set_xticks(d_mem)
ax.set_xticklabels([str(d) for d in d_mem])
ax.set_xlabel(r'$d_{\mathrm{mem}}$')
ax.set_ylabel(r'$\Delta$ accuracy (post-hoc swap $-$ baseline)')
ax.set_title('(b) Fixed-rule-trained model', loc='left', fontweight='bold')
ax.legend(loc='lower right', frameon=False, fontsize=10)
ax.set_ylim(-0.10, 0.05)
ax.grid(True, axis='y', alpha=0.25, lw=0.5)

# Annotate: shape sensitivity becomes negligible at d=32
ax.annotate('within seed noise\nat $d_{\\mathrm{mem}}=32$',
            xy=(32, -0.004), xytext=(32, -0.06),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', lw=0.8, color='#2980B9'))

# ===== Global styling =====
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('fig4_matched_mean.png', dpi=300, bbox_inches='tight')
fig.savefig('fig4_matched_mean.pdf', bbox_inches='tight')
plt.show()

# ===== Print relative drops (for caption) =====
print('Relative drop, Kaczmarz-trained const_m:')
base = {16: 0.9368, 24: 0.9465, 32: 0.9472}
for d, m in zip(d_mem, kacz_const_mean):
    print(f'  d={d}: {100*abs(m)/base[d]:.1f}%')
