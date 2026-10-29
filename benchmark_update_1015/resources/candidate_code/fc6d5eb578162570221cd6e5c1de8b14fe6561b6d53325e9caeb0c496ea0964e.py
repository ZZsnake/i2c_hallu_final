import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import splprep, splev

# Control points coordinates (arranged to form a heart shape)
control_points = np.array([
    [-2, 1.8],
    [0, 1.9],
    [0.3, 1.9],
    [1, 1.1],
    [0.5, -1.4],
    [1.5, -2],
    [2, -0.3],
    [2.5, 0.2],
    [2, 3],
    [3, 0.2]
])

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))

# Generate smooth Bézier curve using spline interpolation
# Close the curve by appending first point
points_closed = np.vstack([control_points, control_points[0]])
x = points_closed[:, 0]
y = points_closed[:, 1]

# Create smooth curve
tck, u = splprep([x, y], s=0, per=True, k=3)
u_new = np.linspace(0, 1, 500)
x_smooth, y_smooth = splev(u_new, tck)

# Fill the curve with blue color
ax.fill(x_smooth, y_smooth, color='blue', alpha=1.0, zorder=1)

# Add red/magenta outline
ax.plot(x_smooth, y_smooth, color='red', linewidth=2, zorder=2)

# Draw control polygon (lines connecting control points)
ax.plot(points_closed[:, 0], points_closed[:, 1], 
        color='#4080FF', linewidth=1.5, zorder=3, marker='')

# Draw control points
ax.plot(control_points[:, 0], control_points[:, 1], 
        marker='s', markersize=10, markerfacecolor='orange', 
        markeredgecolor='black', markeredgewidth=1.5, 
        linestyle='', zorder=4)

# Set axis properties
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_title('Interactive Bézier Curve Editor - Drag Control Points', 
             fontsize=12, pad=10)

# Set axis ticks
ax.set_xticks(range(-4, 5))
ax.set_yticks(range(-4, 5))

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/483065db713b43e09288c06f98e269b3.png", bbox_inches="tight")