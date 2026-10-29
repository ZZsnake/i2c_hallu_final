import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Set figure size
plt.figure(figsize=(8, 8))

# Create 3D axes
ax = plt.axes(projection='3d')

# Define vertices of the cube
vertices = np.array([
    [0, 0, 0],    # A
    [1, 0, 0],    # B
    [1, 1, 0],    # C
    [0, 1, 0],    # D
    [0, 0, 1],    # E
    [1, 0, 1],    # F
    [1, 1, 1],    # G
    [0, 1, 1]     # H
])

# Define edges of the cube
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
]

# Plot edges
for edge in edges:
    ax.plot3D([vertices[edge[0]][0], vertices[edge[1]][0]],
              [vertices[edge[0]][1], vertices[edge[1]][1]],
              [vertices[edge[0]][2], vertices[edge[1]][2]], 'b-', linewidth=2)

# Plot vertices
for i in range(8):
    ax.scatter3D(vertices[i][0], vertices[i][1], vertices[i][2], color='blue', s=50)

# Annotate vertices
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
for i, (x, y, z) in enumerate(vertices):
    ax.text(x, y, z, labels[i], fontsize=12, ha='center', va='center')

# Annotate axes
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)

# Set axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Hide axes for cleaner look
ax.set_axis_off()

# Add point S
S = np.array([0.5, 0.5, 0.5])
ax.scatter3D(S[0], S[1], S[2], color='blue', s=100, marker='o', edgecolor='black')
ax.text(S[0], S[1], S[2], 'S', fontsize=12, ha='center', va='center')

# Adjust view angle for isometric projection
ax.view_init(elev=30, azim=45)

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Remove grid and ticks for cleaner look
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Set title
plt.title('Cube with labeled vertices and point S', fontsize=14, pad=20)

# Show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/AsymptoteGraphicPipeline_Coordinate_system_0-97.png", bbox_inches="tight")