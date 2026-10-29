import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Define control points for the heart shape (8 points total)
# These are arranged to form a symmetric heart using two cubic Bézier curves
control_points = np.array([
    [-2, 2],   # top left
    [0, 1],    # left curve control
    [1, 1],    # right curve control
    [2, 2],    # top right
    [3, 0],    # right side
    [1, -2],   # bottom point
    [-1, -2],  # bottom point (mirrored)
    [-3, 0]    # left side
])

# Create the Bézier curve path for the heart
# We'll use two cubic Bézier curves: left half and right half
verts = [
    (control_points[0, 0], control_points[0, 1]),  # start at top left
    (control_points[1, 0], control_points[1, 1]),  # control point 1
    (control_points[2, 0], control_points[2, 1]),  # control point 2
    (control_points[3, 0], control_points[3, 1]),  # end at top right
    (control_points[4, 0], control_points[4, 1]),  # control point 3
    (control_points[5, 0], control_points[5, 1]),  # bottom right
    (control_points[6, 0], control_points[6, 1]),  # bottom left
    (control_points[7, 0], control_points[7, 1]),  # control point 4
    (control_points[0, 0], control_points[0, 1]),  # close the path back to start
]

codes = [
    Path.MOVETO,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.CLOSEPOLY
]

path = Path(verts, codes)
patch = PathPatch(path, facecolor='blue', edgecolor='red', lw=2, alpha=1.0)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Add the heart shape
ax.add_patch(patch)

# Plot the control polygon (lines connecting control points)
ax.plot(control_points[:, 0], control_points[:, 1], 'b-', lw=1, alpha=0.7)
ax.plot([control_points[-1, 0], control_points[0, 0]], 
        [control_points[-1, 1], control_points[0, 1]], 'b-', lw=1, alpha=0.7)

# Plot the control points as orange squares
ax.plot(control_points[:, 0], control_points[:, 1], 's', 
        markersize=8, markerfacecolor='orange', markeredgecolor='black', markeredgewidth=1)

# Set axis limits and grid
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)
ax.set_title("Interactive Bézier Curve Editor - Drag Control Points", fontsize=12, fontweight='normal')

# Remove axis ticks and labels for clean look
ax.set_xticks(range(-4, 5))
ax.set_yticks(range(-4, 5))
ax.set_xlabel('')
ax.set_ylabel('')

# Ensure equal aspect ratio
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/483065db713b43e09288c06f98e269b3.png", bbox_inches="tight")