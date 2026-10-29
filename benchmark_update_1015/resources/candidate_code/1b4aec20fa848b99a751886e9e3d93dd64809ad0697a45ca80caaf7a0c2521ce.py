import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Define control points for the heart shape (8 points total)
# These are arranged to form an asymmetric compact Bézier shape
control_points = np.array([
    [-1.85, 1.75],   # upper left control
    [0.27, 1.85],    # upper curve point
    [0.67, 1.05],    # upper middle kink
    [2.05, 2.95],    # upper right control
    [2.85, 0.15],    # right side control
    [1.75, -0.35],   # right side curve point
    [1.25, -2.15],   # bottom point
    [0.45, -1.35]    # lower left control
])

# Create the Bézier curve path for the heart
# The shape uses a straight upper segment followed by cubic Bézier sections
verts = [
    (control_points[1, 0], control_points[1, 1]),  # start at upper curve point
    (control_points[2, 0], control_points[2, 1]),  # straight segment to kink
    (control_points[3, 0], control_points[3, 1]),  # right lobe control 1
    (control_points[4, 0], control_points[4, 1]),  # right lobe control 2
    (control_points[5, 0], control_points[5, 1]),  # right side curve point
    (control_points[6, 0], control_points[6, 1]),  # straight segment to bottom
    (control_points[7, 0], control_points[7, 1]),  # lower left control
    (control_points[0, 0], control_points[0, 1]),  # upper left control
    (control_points[1, 0], control_points[1, 1]),  # curve back to start
]

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4,
    Path.LINETO,
    Path.CURVE4,
    Path.CURVE4,
    Path.CURVE4
]

path = Path(verts, codes)
patch = PathPatch(path, facecolor='blue', edgecolor='red', lw=2, alpha=0.8)

# Create the plot
fig, ax = plt.subplots(figsize=(5.5, 4.5))

# Add the heart shape
ax.add_patch(patch)

# Plot the control polygon (lines connecting control points)
ax.plot(control_points[:, 0], control_points[:, 1], 'blue', lw=2, alpha=0.7)
ax.plot([control_points[-1, 0], control_points[0, 0]], 
        [control_points[-1, 1], control_points[0, 1]], 'blue', lw=2, alpha=0.7)

# Plot the control points as orange squares
ax.plot(control_points[:, 0], control_points[:, 1], 's', 
        markersize=8, markerfacecolor='orange', markeredgecolor='black', markeredgewidth=1)

# Set axis limits and grid
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(True, linestyle='--', color='gray', alpha=0.7)
ax.set_title("Interactive Bézier Curve Editor - Drag Control Points", fontsize=12, fontweight='normal')

# Remove axis ticks and labels for clean look
ax.set_xticks(range(-4, 5))
ax.set_yticks(range(-4, 5))
ax.set_xlabel('')
ax.set_ylabel('')

# Use automatic aspect ratio to match the wider plotting area
ax.set_aspect('auto')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/483065db713b43e09288c06f98e269b3.png", bbox_inches="tight")
