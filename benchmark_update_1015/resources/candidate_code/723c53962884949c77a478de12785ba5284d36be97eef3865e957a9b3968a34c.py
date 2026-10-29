import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set figure size
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Define cube vertices (A, B, C, D, and S)
# A is at origin (0,0,0)
# B is at (1,0,0)
# C is at (1,1,0)
# D is at (0,1,0)
# S is at (1,1,1) — top-right-back corner
vertices = {
    'A': (0, 0, 0),
    'B': (1, 0, 0),
    'C': (1, 1, 0),
    'D': (0, 1, 0),
    'S': (1, 1, 1)
}

# Define edges of the cube (as pairs of vertex labels)
edges = [
    ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'),  # bottom face
    ('A', 'D'), ('D', 'S'), ('S', 'C'), ('C', 'B'),  # vertical and top edges
    ('B', 'S'), ('A', 'S')  # diagonal connections for full cube
]

# Draw edges
for edge in edges:
    p1 = vertices[edge[0]]
    p2 = vertices[edge[1]]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=1)

# Label vertices
for label, (x, y, z) in vertices.items():
    ax.text(x, y, z, label, color='blue', fontsize=10, fontweight='bold')

# Label edges: x, y, z
# x along A to B
ax.text(0.5, 0, 0, 'x', color='blue', fontsize=10)
# y along A to D
ax.text(0, 0.5, 0, 'y', color='blue', fontsize=10)
# z along A to D (vertical, but we label near A-D edge)
ax.text(0, 0, 0.5, 'z', color='blue', fontsize=10)

# Set view angle for isometric look
ax.view_init(elev=20, azim=45)

# Remove axes for clean look
ax.set_axis_off()

# Set equal aspect ratio
ax.set_box_aspect([1,1,1])

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/AsymptoteGraphicPipeline_Coordinate_system_0-97.png", bbox_inches="tight")