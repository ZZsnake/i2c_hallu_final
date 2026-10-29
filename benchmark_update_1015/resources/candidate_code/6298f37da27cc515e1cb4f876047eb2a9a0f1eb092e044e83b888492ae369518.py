import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set figure dimensions
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Torus parameters
R = 2.0  # Major radius
r = 0.8  # Minor radius

# Create parameter grid
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, 2 * np.pi, 20)
u, v = np.meshgrid(u, v)

# Parametric equations for torus
x = (R + r * np.cos(v)) * np.cos(u)
y = (R + r * np.cos(v)) * np.sin(u)
z = r * np.sin(v)

# Plot the wireframe surface
ax.plot_wireframe(x, y, z, color='#1f77b4', linewidth=0.5, alpha=0.7, rstride=2, cstride=2)

# Set axis labels
ax.set_xlabel('X-axis: f(θ, φ)', fontsize=10, labelpad=8)
ax.set_ylabel('Y-axis: g(θ, φ)', fontsize=10, labelpad=8)
ax.set_zlabel('Z-axis: h(θ, φ)', fontsize=10, labelpad=8)

# Set axis limits
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 0.8)

# Set title
ax.set_title('Parametric Surface: Torus Geometry', fontsize=14, fontweight='bold', pad=20)

# Set viewing angle
ax.view_init(elev=25, azim=-60)

# Enable grid
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/a92890ce864f4fa7bf373b05b2fe3111.png", bbox_inches="tight")
