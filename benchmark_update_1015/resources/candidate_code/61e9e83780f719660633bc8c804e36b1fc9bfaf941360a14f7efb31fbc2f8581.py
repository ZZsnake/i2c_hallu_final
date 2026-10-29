import numpy as np
import matplotlib.pyplot as plt

# Data
labels = [r'$\mu_R$', r'$\mu_{Train}$', r'$N_{TP}$', r'$N_{FP}^{eff}$', r'$\mu_{AUC}$']
data = {
    'GNN_k=1': [0.37, 0.36, 0.4, 0.28, 0.43],
    'GNN_k=2': [0.284, 0.22, 0.29, 0.46, 0.24],
    'GNN^†': [0.496, 0.48, 0.5, 0.24, 0.33]
}

# Colors and styles
colors = ['#0000FF', '#FF8C00', '#008000']
linestyles = ['--', '-.', ':']
markers = ['o', 'o', 'o']
alphas = [0.2, 0.2, 0.2]

# Create figure
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Plot each model
for i, (label, values) in enumerate(data.items()):
    values += values[:1]  # Close the loop
    ax.plot(angles, values, color=colors[i], linestyle=linestyles[i], marker=markers[i], label=label, linewidth=2)
    ax.fill(angles, values, color=colors[i], alpha=alphas[i])

# Add labels
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, fontweight='bold')

# Set radial limits
ax.set_ylim(0, 0.55)

# Add grid
ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

# Add value annotations
for i, (label, values) in enumerate(data.items()):
    for j, val in enumerate(values[:-1]):
        angle = angles[j]
        ax.text(angle, val + 0.01, f'{val:.3f}', ha='center', va='center', fontsize=9, color=colors[i])

# Legend
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=10)

# Title and layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/radar_12.png", bbox_inches="tight")