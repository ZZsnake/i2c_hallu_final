import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Generate parametric surface data
theta = np.linspace(0, 2 * np.pi, 50)
phi = np.linspace(0, np.pi, 25)
U, V = np.meshgrid(theta, phi)

# Create a torus surface
major_radius, minor_radius = 2.0, 0.8
X = (major_radius + minor_radius * np.cos(V)) * np.cos(U)
Y = (major_radius + minor_radius * np.cos(V)) * np.sin(U)
Z = minor_radius * np.sin(V)

# Plot wireframe with different styling
ax.plot_wireframe(X, Y, Z, color='#2E86AB', linewidth=0.8, 
                 rstride=3, cstride=3, alpha=0.9)

ax.set_title('Parametric Surface: Torus Geometry', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('X-axis: f(θ, φ)', fontsize=11)
ax.set_ylabel('Y-axis: g(θ, φ)', fontsize=11)
ax.set_zlabel('Z-axis: h(θ, φ)', fontsize=11)

ax.grid(True, linestyle='--', alpha=0.6, color='#4A4A4A')

plt.tight_layout()
plt.show()
