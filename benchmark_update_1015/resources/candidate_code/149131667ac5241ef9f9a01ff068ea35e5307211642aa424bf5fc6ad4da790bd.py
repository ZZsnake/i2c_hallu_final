import numpy as np
import matplotlib.pyplot as plt

# ---- Data (approximate from image) ----
labels = [r'$\mu_{Train}$', r'$\mu_{R}$', r'$\mu_{AUC}$',
          r'$N_{TP}^{eff}$', r'$N_{TP}$']
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

# Values approximated from the image (max around 0.5)
gnn_k1 = [0.36, 0.37, 0.43, 0.28, 0.40]
gnn_k2 = [0.22, 0.284, 0.24, 0.46, 0.29]
gnn_up = [0.48, 0.496, 0.33, 0.24, 0.50]

# Close polygons
def close(vals):
    return vals + vals[:1]

gnn_k1_c = close(gnn_k1)
gnn_k2_c = close(gnn_k2)
gnn_up_c = close(gnn_up)

# ---- Figure and polar axis ----
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)

# Set starting angle and direction to match figure
ax.set_theta_offset(0)           # start at right
ax.set_theta_direction(1)        # counterclockwise

# Set radial limits and grid
ax.set_ylim(0.0, 0.5)
ax.set_yticks([0.1, 0.2, 0.3, 0.4])
ax.set_yticklabels([])
ax.yaxis.grid(True, color='0.8')
ax.xaxis.grid(True, color='0.8')

# Put category labels at correct angles
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)

# ---- Plot each series ----
color1 = 'blue'
color2 = '#ff7f0e'
color3 = '#2ca02c'

# GNN_{k=1}
ax.plot(angles, gnn_k1_c, color=color1, linestyle=':', linewidth=1.5,
        marker='o', markersize=4, label=r'GNN$_{k=1}$')
ax.fill(angles, gnn_k1_c, color=color1, alpha=0.15)

# GNN_{k=2}
ax.plot(angles, gnn_k2_c, color=color2, linestyle=':', linewidth=1.5,
        marker='o', markersize=4, label=r'GNN$_{k=2}$')
ax.fill(angles, gnn_k2_c, color=color2, alpha=0.15)

# GNN^↑
ax.plot(angles, gnn_up_c, color=color3, linestyle=':', linewidth=1.5,
        marker='o', markersize=4, label=r'GNN$^{\uparrow}$')
ax.fill(angles, gnn_up_c, color=color3, alpha=0.15)

# ---- Annotate numeric values near vertices ----
def annotate_values(values, angles, ax, offset=0.015):
    for r, theta in zip(values, angles[:-1]):
        # place text slightly outside the point
        ax.text(theta, r + offset, f"{r:.3f}".rstrip('0').rstrip('.'),
                ha='center', va='center', fontsize=8)

annotate_values(gnn_k1, angles, ax)
annotate_values(gnn_k2, angles, ax)
annotate_values(gnn_up, angles, ax)

# ---- Legend ----
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
                   frameon=True, fontsize=9)
legend.get_frame().set_alpha(0.9)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/radar_12.png", bbox_inches="tight")
