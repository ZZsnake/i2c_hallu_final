import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# Figure and 3D axis
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Parametric variables
# Use a torus-like parameterization with a slight vertical scaling
theta = np.linspace(0, 2 * np.pi, 40)  # angle around central axis
phi = np.linspace(0, 2 * np.pi, 40)    # angle around tube
theta, phi = np.meshgrid(theta, phi)

# Torus parameters (major and minor radii)
R = 2.0   # major radius
r = 1.0   # minor radius

# Deform z a bit to resemble the "flared" cylinder/torus hybrid
# Basic torus: x = (R + r*cos(phi))*cos(theta)
#              y = (R + r*cos(phi))*sin(theta)
#              z = r*sin(phi)
# We'll compress z and shift it upward so that it starts near 0
x = (R + r * np.cos(phi)) * np.cos(theta)
y = (R + r * np.cos(phi)) * np.sin(theta)
z = 0.4 * (np.sin(phi) + 1)  # ranges roughly 0 to ~0.8

# Plot wireframe
ax.plot_wireframe(x, y, z, rstride=1, cstride=1,
                  color='#1f77b4', linewidth=0.8)

# Axis labels
ax.set_xlabel(r'X-axis: f($\theta$, $\phi$)', labelpad=10)
ax.set_ylabel(r'Y-axis: g($\theta$, $\phi$)', labelpad=10)
ax.set_zlabel(r'Z-axis: h($\theta$, $\phi$)', labelpad=10)

# Title
ax.set_title('Parametric Surface: Torus Geometry',
             fontsize=16, fontweight='bold', pad=20)

# Set axis limits to mimic the figure
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 0.8)

# Set viewing angle similar to the image
ax.view_init(elev=25, azim=40)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/a92890ce864f4fa7bf373b05b2fe3111.png", bbox_inches="tight")