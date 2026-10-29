import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Define the data
nodes = list(range(1, 12))
strengths = [
    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2, 0.1, 0.3, 0.4],
    [0.4, 0.3, 0.2, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.2, 0.1],
    # Add more strengths as needed
]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Draw the nodes
for i, node in enumerate(nodes):
    ax.scatter(np.cos(2 * np.pi * i / len(nodes)), np.sin(2 * np.pi * i / len(nodes)), s=100, c='blue', zorder=5)

# Draw the chords
colors = cm.viridis(strengths[0])
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        ax.plot([np.cos(2 * np.pi * i / len(nodes)), np.cos(2 * np.pi * j / len(nodes))],
                [np.sin(2 * np.pi * i / len(nodes)), np.sin(2 * np.pi * j / len(nodes))],
                color=colors[j - i - 1], alpha=strengths[0][j - i - 1])

# Set aspect ratio to be equal
ax.set_aspect('equal')

# Hide axes
ax.axis('off')

# Add color bar
sm = cm.ScalarMappable(cmap=cm.viridis)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.1)
cbar.set_label("Edge Strength")
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/6f71e9e4ccd3446192211eba29d72d76.png", bbox_inches="tight")