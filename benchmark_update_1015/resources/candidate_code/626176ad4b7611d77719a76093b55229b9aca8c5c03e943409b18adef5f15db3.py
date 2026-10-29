import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate dummy data
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-(X**2 + Y**2))
Z2 = np.exp(-(X**2 + Y**2) / 2)
Z3 = np.exp(-(X**2 + Y**2) / 4)
Z4 = np.exp(-(X**2 + Y**2) / 8)

# Create figure and subplots
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

# Plot the surfaces
surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', linewidth=0, antialiased=False)
surf2 = ax2.plot_surface(X, Y, Z2, cmap='viridis', linewidth=0, antialiased=False)
surf3 = ax3.plot_surface(X, Y, Z3, cmap='viridis', linewidth=0, antialiased=False)
surf4 = ax4.plot_surface(X, Y, Z4, cmap='viridis', linewidth=0, antialiased=False)

# Set labels and limits
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(0, 4)

# Adjust subplot layout
plt.tight_layout()

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/26b419b4e2ed4c1aacb007011d1172c9.png", bbox_inches="tight")