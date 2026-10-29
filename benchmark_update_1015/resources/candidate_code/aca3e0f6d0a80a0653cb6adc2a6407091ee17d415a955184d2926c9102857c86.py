import matplotlib.pyplot as plt

# Set figure size
plt.figure(figsize=(7, 10))

# Dummy data based on visual estimation
data = {
    'DE': {
        'iTrm-All': (0.88, 0.095),
        'iTrm-En': (0.82, 0.096),
        'PatchTST': (0.91, 0.105),
        'TiDE': (0.72, 0.125),
        'TimeXer': (0.96, 0.095)
    },
    'PJM': {
        'iTrm-All': (0.88, 0.48),
        'iTrm-En': (0.62, 0.48),
        'PatchTST': (0.86, 0.46),
        'TiDE': (0.89, 0.58),
        'TimeXer': (0.89, 0.44)
    }
}

# Define colors and markers
colors = {
    'iTrm-All': '#90EE90',   # light green
    'iTrm-En': '#3CB371',    # medium sea green
    'PatchTST': '#20B2AA',   # light sea green
    'TiDE': '#FF8C00',       # dark orange
    'TimeXer': '#DC143C'     # crimson
}

markers = {
    'iTrm-All': 'o',         # hollow circle
    'iTrm-En': 'o',          # solid circle
    'PatchTST': 'o',
    'TiDE': 'o',
    'TimeXer': 'o'
}

# Create subplots
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Plot DE
ax1.set_title('DE', fontsize=12)
ax1.set_xlabel('CKA Similarity', fontsize=10)
ax1.set_ylabel('MSE', fontsize=10)
ax1.set_xlim(0.6, 1.1)
ax1.set_ylim(0.09, 0.13)
ax1.grid(True, linestyle='--', linewidth=0.5)

for method, (x, y) in data['DE'].items():
    if method == 'iTrm-All':
        ax1.scatter(x, y, color=colors[method], marker=markers[method], s=100, edgecolors='black', linewidth=1.5, facecolors='none')
    else:
        ax1.scatter(x, y, color=colors[method], marker=markers[method], s=100)

# Plot PJM
ax2.set_title('PJM', fontsize=12)
ax2.set_xlabel('CKA Similarity', fontsize=10)
ax2.set_ylabel('MSE', fontsize=10)
ax2.set_xlim(0.5, 1.0)
ax2.set_ylim(0.40, 0.60)
ax2.grid(True, linestyle='--', linewidth=0.5)

for method, (x, y) in data['PJM'].items():
    if method == 'iTrm-All':
        ax2.scatter(x, y, color=colors[method], marker=markers[method], s=100, edgecolors='black', linewidth=1.5, facecolors='none')
    else:
        ax2.scatter(x, y, color=colors[method], marker=markers[method], s=100)

# Create legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='iTrm-All', markerfacecolor=colors['iTrm-All'], markersize=10, markeredgecolor='black', markeredgewidth=1.5),
    plt.Line2D([0], [0], marker='o', color='w', label='iTrm-En', markerfacecolor=colors['iTrm-En'], markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='PatchTST', markerfacecolor=colors['PatchTST'], markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='TiDE', markerfacecolor=colors['TiDE'], markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='TimeXer', markerfacecolor=colors['TimeXer'], markersize=10)
]

plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=10)

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/scatter_9.png", bbox_inches="tight")