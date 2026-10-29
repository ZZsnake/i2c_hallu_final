import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(8, 8))

# Define points
O = np.array([2, 3])
P = np.array([5, 6])

# Draw circle
radius = np.linalg.norm(O - P)
circle = plt.Circle(O, radius, fill=False, color='blue')
ax.add_patch(circle)

# Plot points with labels
ax.plot(O[0], O[1], 'ko')
ax.text(O[0], O[1], 'O (2,3)', ha='right', va='bottom')
ax.plot(P[0], P[1], 'ko')
ax.text(P[0], P[1], 'P (5,6)', ha='left', va='bottom')

# Draw dashed line between O and P
ax.plot([O[0], P[0]], [O[1], P[1]], 'k--')

# Draw axes with arrows
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.arrow(0, 0, 5, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
ax.arrow(0, 0, 0, 6, head_width=0.2, head_length=0.2, fc='k', ec='k')

# Set axis labels and ticks
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xticks(np.arange(0, 7, 1))
ax.set_yticks(np.arange(0, 7, 1))
ax.grid(True)

# Set equal aspect and limits
ax.set_aspect('equal')
ax.set_xlim(0, 7)
ax.set_ylim(0, 7)

# Save the figure
plt.savefig(r'/home/zhoujiawei/work/image2codeBench/image2code/combine_data/graphic_py_dataset_output/images/AsymptoteGraphicPipeline_Analytic_geometry_3-30.png', bbox_inches='tight')
plt.close()