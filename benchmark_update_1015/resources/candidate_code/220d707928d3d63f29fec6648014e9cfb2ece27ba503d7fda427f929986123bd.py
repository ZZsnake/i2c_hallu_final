import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, RegularPolygon, Ellipse, FancyArrow, FancyBboxPatch, PathPatch
from matplotlib.path import Path
import matplotlib.lines as mlines

# Create figure and axes
fig, axes = plt.subplots(3, 3, figsize=(7, 7))
plt.subplots_adjust(wspace=0.35, hspace=0.55)

# Utility to standardize axes
def setup_ax(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=12)

# Shape 1: Red circle
ax = axes[0, 0]
setup_ax(ax, "Shape 1")
circle = Circle((0.5, 0.5), 0.28, facecolor='#e41a1c', edgecolor='none')
ax.add_patch(circle)

# Shape 2: Vertical blue rectangle (rounded look via FancyBboxPatch)
ax = axes[0, 1]
setup_ax(ax, "Shape 2")
rect = FancyBboxPatch((0.43, 0.25), 0.14, 0.5,
                      boxstyle="round,pad=0.02,rounding_size=0.05",
                      facecolor='#4c93c5', edgecolor='none')
ax.add_patch(rect)

# Shape 3: Green Pac-Man (circle with wedge removed)
ax = axes[0, 2]
setup_ax(ax, "Shape 3")
# Full circle
pac_circle = Circle((0.5, 0.5), 0.28, facecolor='#4daf4a', edgecolor='none')
ax.add_patch(pac_circle)
# Cut-out wedge (draw white triangle over circle)
wedge = Wedge(center=(0.5, 0.5), r=0.29, theta1=-30, theta2=30,
              facecolor='white', edgecolor='white')
ax.add_patch(wedge)

# Shape 4: Purple hexagon
ax = axes[1, 0]
setup_ax(ax, "Shape 4")
hexagon = RegularPolygon((0.5, 0.5), numVertices=6, radius=0.28,
                         orientation=0, facecolor='#984ea3', edgecolor='none')
ax.add_patch(hexagon)

# Shape 5: Orange ellipse
ax = axes[1, 1]
setup_ax(ax, "Shape 5")
ellipse = Ellipse((0.5, 0.5), width=0.5, height=0.26,
                  facecolor='#ff9900', edgecolor='none')
ax.add_patch(ellipse)

# Shape 6: Yellow NE arrow
ax = axes[1, 2]
setup_ax(ax, "Shape 6")
arrow = FancyArrow(0.3, 0.3, 0.35, 0.35,
                   width=0.03, length_includes_head=True,
                   head_width=0.12, head_length=0.12,
                   facecolor='#ffe900', edgecolor='none')
ax.add_patch(arrow)

# Shape 7: Brown heart-like blob using Path
ax = axes[2, 0]
setup_ax(ax, "Shape 7")
verts = [
    (0.38, 0.25),  # bottom tip
    (0.25, 0.45),
    (0.28, 0.65),
    (0.38, 0.78),
    (0.50, 0.70),
    (0.60, 0.58),
    (0.60, 0.42),
    (0.54, 0.32),
    (0.47, 0.28),
    (0.38, 0.25),  # close
]
codes = [
    Path.MOVETO,
    Path.CURVE3,
    Path.CURVE3,
    Path.CURVE3,
    Path.CURVE3,
    Path.CURVE3,
    Path.CURVE3,
    Path.CURVE3,
    Path.CURVE3,
    Path.CLOSEPOLY,
]
heart_path = Path(verts, codes)
heart = PathPatch(heart_path, facecolor='#a65628', edgecolor='none')
ax.add_patch(heart)

# Shape 8: Pink rounded rectangle
ax = axes[2, 1]
setup_ax(ax, "Shape 8")
rounded_rect = FancyBboxPatch((0.42, 0.26), 0.16, 0.48,
                              boxstyle="round,pad=0.03,rounding_size=0.12",
                              facecolor='#f781bf', edgecolor='none')
ax.add_patch(rounded_rect)

# Shape 9: Grey V checkmark
ax = axes[2, 2]
setup_ax(ax, "Shape 9")
line1 = mlines.Line2D([0.3, 0.5], [0.55, 0.35],
                      color='#9e9e9e', linewidth=6, solid_capstyle='round')
line2 = mlines.Line2D([0.5, 0.7], [0.35, 0.55],
                      color='#9e9e9e', linewidth=6, solid_capstyle='round')
ax.add_line(line1)
ax.add_line(line2)
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/38160eab60da412796ea6ad1ad9a31f5.png", bbox_inches="tight")