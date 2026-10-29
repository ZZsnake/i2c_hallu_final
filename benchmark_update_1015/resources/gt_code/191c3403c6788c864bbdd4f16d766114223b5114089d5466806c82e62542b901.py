import matplotlib.pyplot as plt
import numpy as np

# Plot ellipse with different aspect ratios
theta = np.linspace(0, 2 * np.pi, 150)
fig, axes = plt.subplots(2, 2, figsize=(9, 7))

# First subplot: default aspect ratio (ellipse)
axes[0, 0].plot(4 * np.cos(theta), 2.5 * np.sin(theta), color='#1f77b4', linewidth=2.5)
axes[0, 0].set_title('Default aspect ratio (elliptical)', fontsize=11, fontweight='bold')

# Second subplot: equal aspect ratio (circle)
axes[0, 1].plot(4 * np.cos(theta), 2.5 * np.sin(theta), color='#d62728', linewidth=2.5, linestyle='--')
axes[0, 1].axis('equal')
axes[0, 1].set_title('Equal axis scaling (circular)', fontsize=11, fontweight='bold')

# Third subplot: equal aspect with adjusted limits
axes[1, 0].plot(4 * np.cos(theta), 2.5 * np.sin(theta), color='#2ca02c', linewidth=2.5, marker='o', markersize=2)
axes[1, 0].axis('equal')
axes[1, 0].set(xlim=(-5, 5), ylim=(-4, 4))
axes[1, 0].set_title('Equal scaling with modified bounds', fontsize=11, fontweight='bold')

# Fourth subplot: aspect ratio adjustment
axes[1, 1].plot(4 * np.cos(theta), 2.5 * np.sin(theta), color='#9467bd', linewidth=2.5, linestyle='-.')
axes[1, 1].set_aspect('equal', 'box')
axes[1, 1].set_title('Aspect ratio set to equal', fontsize=11, fontweight='bold')

fig.suptitle('Geometric Transformations of Conic Sections', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()