import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['ResNet18', 'AlexNet', 'ResNet50']
perturbations = ['0%', '50%', '4:8', '75%', '6:8', '87.50%', '7:8']
speedups = {
    'ResNet18': [1.0, 1.8, 2.6, 3.3, 4.6, 4.5, 7.4],
    'AlexNet': [1.0, 1.9, 2.9, 3.4, 5.2, 6.1, 8.7],
    'ResNet50': [1.0, 1.6, 1.8, 2.7, 2.9, 3.9, 4.3]
}

# Colors (matching the image)
colors = ['#E9967A', '#B8860B', '#6B8E23', '#20B2AA', '#4682B4', '#6495ED', '#DA70D6']

# Figure setup
fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor('white')
ax.set_facecolor('#E6E6FA')

# Bar width and positions
bar_width = 0.12
x = np.arange(len(models))

# Plot bars for each perturbation
for i, pert in enumerate(perturbations):
    values = [speedups[model][i] for model in models]
    bars = ax.bar(x + i * bar_width, values, bar_width, label=pert, color=colors[i])
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height:.1f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Customize axes
ax.set_xlabel('')
ax.set_ylabel('Speedup', fontsize=12)
ax.set_xticks(x + bar_width * 3)  # Center the group
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.set_ylim(0, 10)
ax.set_yticks(range(0, 11, 2))
ax.grid(axis='y', linestyle='-', alpha=0.7)

# Legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=7, frameon=False, fontsize=10)

# Title
ax.set_title('Perturbation', fontsize=12, fontweight='bold')

# Tight layout
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/bar_37.png", bbox_inches="tight")