import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Define nodes, colors, and positions
nodes = range(7)
colors = ['red', 'teal', 'yellow', 'purple', 'blue', 'orange', 'green']
node_labels = ['Coastal', 'Coastal', 'Coastal', 'Mountain', 'Coastal', 'Desert', 'Inland']
edge_labels = {
    (0, 1): 'Temperature',
    (1, 2): 'Humidity',
    (2, 3): 'Wind Speed',
    (3, 4): 'Precipitation',
    (4, 5): 'Pressure',
    (5, 6): 'Visibility'
}

# Calculate circular positions
n = len(nodes)
positions = np.zeros((n, 2))
for i in nodes:
    theta = 2 * np.pi * i / n
    positions[i] = [np.cos(theta), np.sin(theta)]

# Create plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_axis_off()
ax.set_title('Weather Station Data Communication Network', fontsize=14)

# Draw nodes
for i in nodes:
    ax.scatter(positions[i, 0], positions[i, 1], s=400, c=colors[i], 
               edgecolor='black', linewidth=1.5, zorder=5)
    ax.text(positions[i, 0], positions[i, 1], str(i), ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

# Draw edges with labels
for (u, v), label in edge_labels.items():
    dx = positions[v, 0] - positions[u, 0]
    dy = positions[v, 1] - positions[u, 1]
    
    # Draw arrow
    ax.arrow(positions[u, 0], positions[u, 1], dx*0.9, dy*0.9, 
             head_width=0.04, head_length=0.06, fc='black', ec='black',
             length_includes_head=True, zorder=3)
    
    # Add edge label
    ax.text(positions[u, 0] + dx/2, positions[u, 1] + dy/2, label,
            ha='center', va='center', fontsize=9, backgroundcolor='white',
            zorder=4)

# Create legend
legend_elements = [
    Patch(fc='blue', label='Coastal'),
    Patch(fc='red', label='Coastal'),
    Patch(fc='green', label='Inland'),
    Patch(fc='orange', label='Desert'),
    Patch(fc='yellow', label='Coastal'),
    Patch(fc='teal', label='Inland'),
    Patch(fc='purple', label='Mountain')
]
ax.legend(handles=legend_elements, title="Location", bbox_to_anchor=(1.07, 1), loc='upper left')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/graph_8.png", bbox_inches="tight")