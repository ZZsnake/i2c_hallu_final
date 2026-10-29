import numpy as np
import matplotlib.pyplot as plt

# Figure setup
plt.figure(figsize=(8, 6))

# Create grid of coordinates
x = np.linspace(-50, 50, 20)
y = np.linspace(-50, 50, 20)
X, Y = np.meshgrid(x, y)

# Simulated vector field (vortical water currents)
# Using smooth sinusoidal functions to mimic the observed pattern
U = np.cos(X / 5.0) * np.sin(Y / 5.0)
V = -np.sin(X / 5.0) * np.cos(Y / 5.0)

# Speed (magnitude) for coloring
speed = np.sqrt(U**2 + V**2)

# Main axes
ax = plt.gca()
ax.set_title("Simulated Water Currents in a Lake")

# Quiver plot
# angles='xy' keeps arrow directions consistent with the coordinate field
q = ax.quiver(
    X, Y, U, V,
    speed,                      # color by speed
    cmap='YlGnBu',
    angles='xy',
    scale=20,                   # adjust to get visually similar arrow size
    width=0.0035
)

# Axes limits and labels
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_xlabel("X Coordinate (m)")
ax.set_ylabel("Y Coordinate (m)")

# Ticks and grid
ticks = np.arange(-40, 41, 20)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.grid(True, linestyle='--', color='gray', alpha=0.5)

# Use the default rectangular plotting area
ax.set_aspect('auto')

# Colorbar
cbar = plt.colorbar(q, ax=ax)
cbar.set_label("Water Speed")

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/quiver_15.png", bbox_inches="tight")
