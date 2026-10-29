import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Data for the radar chart
categories = ['μTrain', 'μR', 'μAUC', r'$N_{eff}^P$', r'$N_{TP}$', r'$N_{eff}^P$']
N = len(categories)

# Values for each series (based on visual inspection)
values_gnn_k1 = [0.36, 0.27, 0.43, 0.28, 0.4, 0.24]
values_gnn_k2 = [0.22, 0.284, 0.24, 0.46, 0.29, 0.24]
values_gnn_dagger = [0.48, 0.496, 0.33, 0.24, 0.5, 0.24]

# Close the polygon
values_gnn_k1 += values_gnn_k1[:1]
values_gnn_k2 += values_gnn_k2[:1]
values_gnn_dagger += values_gnn_dagger[:1]

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Initialize figure
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))

# Plot data
ax.plot(angles, values_gnn_k1, 'o-', linewidth=1.5, linestyle=':', 
        color='#4169E1', label=r'$\mathrm{GNN}_{k=1}$', markersize=6)
ax.fill(angles, values_gnn_k1, alpha=0.25, color='#4169E1')

ax.plot(angles, values_gnn_k2, 'o-', linewidth=1.5, linestyle=':', 
        color='#FFA500', label=r'$\mathrm{GNN}_{k=2}$', markersize=6)
ax.fill(angles, values_gnn_k2, alpha=0.25, color='#FFA500')

ax.plot(angles, values_gnn_dagger, 'D-', linewidth=1.5, linestyle=':', 
        color='#2E8B57', label=r'$\mathrm{GNN}^{\dagger}$', markersize=5)
ax.fill(angles, values_gnn_dagger, alpha=0.25, color='#2E8B57')

# Set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)

# Set radial limits and labels
ax.set_ylim(0, 0.5)
ax.set_yticks([0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels(['0.2', '0.3', '0.4', '0.5'], size=9)

# Add gridlines
ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)

# Add value labels on the plot
label_positions_k1 = [(angles[i], values_gnn_k1[i]) for i in range(N)]
label_positions_k2 = [(angles[i], values_gnn_k2[i]) for i in range(N)]
label_positions_dagger = [(angles[i], values_gnn_dagger[i]) for i in range(N)]

for i in range(N):
    ax.text(angles[i], values_gnn_dagger[i] + 0.02, f'{values_gnn_dagger[i]:.2f}', 
            ha='center', va='bottom', fontsize=8, color='#2E8B57')

# Legend
ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05), frameon=True, fontsize=10)

# Set starting angle
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/radar_12.png", bbox_inches="tight")