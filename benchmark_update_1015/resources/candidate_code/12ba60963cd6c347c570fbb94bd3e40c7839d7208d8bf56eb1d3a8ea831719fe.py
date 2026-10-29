import numpy as np
import matplotlib.pyplot as plt

# Set figure size
fig, ax = plt.subplots(figsize=(8, 6))

# Define grid
x = np.linspace(-5, 5, 21)
y = np.linspace(-5, 5, 21)
X, Y = np.meshgrid(x, y)

# Create flow vectors with an oscillatory diagonal groundwater flow pattern
U = 1.1 * (np.sin(Y) + np.cos(X))
V = 1.1 * (np.sin(X) - np.cos(Y))

# Create flow rate (magnitude) for coloring
speed = np.sqrt(U**2 + V**2)

# Create a colormap for flow rate
cmap = plt.cm.plasma
norm = plt.Normalize(vmin=0.7, vmax=2.25)

# Add black streamlines behind the colored quiver arrows
xs = np.linspace(-5, 5, 120)
ys = np.linspace(-5, 5, 120)
XS, YS = np.meshgrid(xs, ys)
US = 1.1 * (np.sin(YS) + np.cos(XS))
VS = 1.1 * (np.sin(XS) - np.cos(YS))
ax.streamplot(xs, ys, US, VS, color='black', density=1.45, linewidth=0.45, arrowsize=0.7, zorder=1)

# Plot quiver with color based on speed
q = ax.quiver(X, Y, U, V, speed, cmap=cmap, norm=norm, scale=22, width=0.0045, headwidth=3, headlength=4, zorder=2)

# Add colorbar
cbar = plt.colorbar(q, ax=ax, shrink=0.8, pad=0.1)
cbar.set_label('Flow Rate (m³/s)', rotation=90, labelpad=10)

# Set axis labels and title
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Groundwater Flow Patterns Under Agricultural Land')

# Add grid with dashed lines
ax.grid(True, linestyle='--', linewidth=0.5)

# Set axis limits
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Adjust layout to prevent clipping
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/quiver_19.png", bbox_inches="tight")
