"""Generate executive summary figure for the supervisor brief."""
import sys
sys.path.insert(0, '/Users/alexandresepulvedadedietrich/CUIMC-Appointment-Simulation')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Verified numbers from balking_effect_verification.csv (30 seeds, A_baseline)
utilization = 0.8418
served_rate  = 0.2692

# Outcome decomposition (A_baseline means, 30 seeds)
served    = 0.2692
balked    = 0.3550
canceled  = 0.3248
no_show   = 0.0506
no_offer  = 0.0000
unresolved = 0.0004

# Colors (match plot_style.py palette)
C_SERVED    = '#1f77b4'   # blue
C_BALKED    = '#9467bd'   # purple
C_CANCELED  = '#d62728'   # red
C_NOSHOW    = '#2ca02c'   # green
C_NOOFFER   = '#ff7f0e'   # orange
C_UNRESOLV  = '#bfbfbf'   # grey

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6),
                          gridspec_kw={'width_ratios': [1, 1.4]})

# ---- Left panel: Utilization vs Served Rate ----
ax = axes[0]
labels = ['Utilization\n(slots filled)', 'Served Rate\n(arrivals served)']
values = [utilization, served_rate]
colors = ['#2ca02c', '#1f77b4']
bars = ax.bar(labels, values, color=colors, width=0.42, edgecolor='white', linewidth=0.5)

# Value labels above bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylim(0, 1.10)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7.5)
ax.tick_params(axis='x', labelsize=8)

# Dashed ceiling line with label centered between the two bars in data coords
ax.axhline(y=32/100, color='#7f7f7f', linestyle='--', linewidth=1.0)
ax.text(0.55, 32/100 + 0.025, '32/100 ceiling',
        ha='center', va='bottom', fontsize=7.5, color='dimgray')

ax.set_title('Utilization vs. Access', fontsize=9, fontweight='bold', pad=5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Rate', fontsize=8)

# ---- Right panel: Pie chart ----
ax2 = axes[1]

# Drop the zero no-offer and near-zero unresolved from the pie for clarity;
# combine them into a single tiny "other" slice only if non-zero.
pie_sizes  = [served, balked, canceled, no_show]
pie_colors = [C_SERVED, C_BALKED, C_CANCELED, C_NOSHOW]
pie_labels = ['Served\n26.9%', 'Balked\n35.5%', 'Canceled\n32.5%', 'No-show\n5.1%']

wedge_props = dict(linewidth=0.6, edgecolor='white')
wedges, texts = ax2.pie(
    pie_sizes,
    colors=pie_colors,
    labels=None,
    startangle=90,
    counterclock=False,
    wedgeprops=wedge_props,
    radius=1.0,
)

# Place labels outside the pie with leader lines
label_distance = 1.22
for wedge, lbl in zip(wedges, pie_labels):
    angle = (wedge.theta1 + wedge.theta2) / 2
    x = label_distance * np.cos(np.deg2rad(angle))
    y = label_distance * np.sin(np.deg2rad(angle))
    ha = 'left' if x > 0 else 'right'
    ax2.text(x, y, lbl, ha=ha, va='center', fontsize=7.2, fontweight='bold',
             color=wedge.get_facecolor())

# Footnote: no-offer = 0.0%
ax2.text(0, -1.5, 'No-offer: 0.0%', ha='center', va='center',
         fontsize=6.8, color='#555555', style='italic')

ax2.set_title('Arrival Outcome Decomposition (Baseline)',
              fontsize=9, fontweight='bold', pad=5)
ax2.set_aspect('equal')

# Caption note
fig.text(0.5, -0.04,
         'Baseline: 100 arrivals/day, 32 slots/day, 30 seeds.',
         ha='center', fontsize=7, color='#555555', style='italic')

plt.tight_layout(pad=1.2)
outpath = '/Users/alexandresepulvedadedietrich/CUIMC-Appointment-Simulation/reports/figures/exec_summary_panel.pdf'
plt.savefig(outpath, dpi=150)
print(f'Saved: {outpath}')

# PNG for visual inspection
outpng = outpath.replace('.pdf', '.png')
plt.savefig(outpng, bbox_inches='tight', dpi=150)
print(f'Saved: {outpng}')
