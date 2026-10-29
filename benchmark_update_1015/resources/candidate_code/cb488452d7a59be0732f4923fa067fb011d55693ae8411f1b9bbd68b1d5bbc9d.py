import matplotlib.pyplot as plt
import numpy as np

# Set figure size
plt.figure(figsize=(6, 6))

# Create a Cartesian grid for radial contour levels
x = np.linspace(-5, 10, 500)
y = np.linspace(-5, 10, 500)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)

# Plot concentric contour levels, including the outer level 12
levels = [2, 4, 6, 8, 10, 12, 14]
cs = plt.contour(X, Y, Z, levels=levels, cmap='winter', linewidths=1.6)

# Add labels at varied positions along the contour paths
manual_positions = [
    (1.4, -1.4),   # 2
    (3.0, -2.8),   # 4
    (-3.7, -4.5),  # 6
    (4.5, 4.2),    # 6
    (-4.0, 7.0),   # 8
    (-4.3, 9.1),   # 10
    (6.6, 7.6),    # 10
    (9.3, -3.4),   # 10
    (7.2, 9.5),    # 12
]
plt.clabel(cs, inline=True, fontsize=10, fmt='%d', manual=manual_positions)

# Set axis limits and aspect ratio to match the framed Cartesian plot
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.gca().set_aspect('equal', adjustable='box')

# Keep visible axes, ticks, tick labels, and rectangular plot frame
plt.xticks(np.arange(-4, 11, 2))
plt.yticks(np.arange(-4, 11, 2))
plt.grid(False)

# Show the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/contour_4.png")