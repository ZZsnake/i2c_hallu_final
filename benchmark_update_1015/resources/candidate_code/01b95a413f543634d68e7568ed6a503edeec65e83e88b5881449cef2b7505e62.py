import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a figure with specified dimensions
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Define the parametric equations for a torus
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

R = 2  # Major radius
r = 1  # Minor radius

X = (R + r * np.cos(phi)) * np.cos(theta)
Y = (R + r * np.cos(phi)) * np.sin(theta)
Z = r * np.sin(phi)

# Plot the torus
ax.plot_surface(X, Y, Z, color='cyan', alpha=0.5, edgecolor='cyan', linewidth=0.5)

# Set labels and title
ax.set_xlabel('X-axis: $f(\\theta, \\phi)$')
ax.set_ylabel('Y-axis: $g(\\theta, \\phi)$')
ax.set_zlabel('Z-axis')
ax.set_title('Parametric Surface: Torus Geometry')

# Set axis limits
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-1, 1)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/a92890ce864f4fa7bf373b05b2fe3111.png", bbox_inches="tight")