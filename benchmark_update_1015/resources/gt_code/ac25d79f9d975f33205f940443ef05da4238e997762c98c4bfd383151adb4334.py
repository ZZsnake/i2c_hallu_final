import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw unit cube
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
])
edges = [
    [vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]],  # bottom
    [vertices[4], vertices[5], vertices[6], vertices[7], vertices[4]],  # top
    [vertices[0], vertices[4]], [vertices[1], vertices[5]],
    [vertices[2], vertices[6]], [vertices[3], vertices[7]]
]
for edge in edges:
    edge = np.array(edge)
    ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], 'b-', alpha=0.1)

# Draw points and labels
points = [
    (0, 0, 0, "A"), (1, 0, 0, "B"), (0, 1, 0, "C"), (0, 0, 1, "D"), (0.6, 0.8, 0.5, "S")
]
for x, y, z, label in points:
    ax.scatter(x, y, z, color='b')
    ax.text(x, y, z, label, color='k', ha='center', va='bottom')

# Draw edges with higher opacity
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], 'b-', alpha=1)  # bottom
ax.plot([0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'b-', alpha=1)  # back
ax.plot([1, 1, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'b-', alpha=1)  # front
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 1], 'b-', alpha=1)  # top

# Axis labels
ax.text(0.5, 0, 0, 'x', color='k', ha='center', va='top')
ax.text(0, 0.5, 0, 'y', color='k', ha='right', va='center')
ax.text(0, 0, 0.5, 'z', color='k', ha='left', va='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_axis_off()

plt.savefig('/home/zhoujiawei/work/image2codeBench/image2code/combine_data/graphic_py_dataset_output/images/AsymptoteGraphicPipeline_Coordinate_system_0-97.png', bbox_inches='tight', pad_inches=0)
plt.close()