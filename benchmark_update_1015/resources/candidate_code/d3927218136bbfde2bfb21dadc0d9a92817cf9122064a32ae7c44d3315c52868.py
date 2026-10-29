import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 1. Setup Figure and Subplots
fig, axs = plt.subplots(3, 3, figsize=(7, 7))
fig.patch.set_facecolor('white')
axs = axs.flatten()

# 2. Define visual properties
colors = {
    'shape1': '#d92129',
    'shape2': '#337ab7',
    'shape3': '#5cb85c',
    'shape4': '#9b59b6',
    'shape5': '#f0ad4e',
    'shape6': '#ffdd00',
    'shape7': '#a05d2c',
    'shape8': '#f781bf',
    'shape9': '#999999'
}
titles = [f'Shape {i}' for i in range(1, 10)]
font_props = {'family': 'serif', 'size': 14}

# 3. Iterate and draw shapes
for i, ax in enumerate(axs):
    # Common axis setup
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.set_title(titles[i], fontdict=font_props, y=1.0)

    shape = None
    # Draw shapes based on index
    if i == 0: # Shape 1: Circle
        shape = patches.Circle((0.5, 0.5), radius=0.3, facecolor=colors['shape1'])
    elif i == 1: # Shape 2: Rectangle
        shape = patches.Rectangle((0.4, 0.25), 0.2, 0.5, facecolor=colors['shape2'], ec='none')
    elif i == 2: # Shape 3: Pac-Man (Wedge)
        shape = patches.Wedge((0.5, 0.5), 0.35, 40, 320, facecolor=colors['shape3'])
    elif i == 3: # Shape 4: Hexagon
        shape = patches.RegularPolygon((0.5, 0.5), numVertices=6, radius=0.3, facecolor=colors['shape4'])
    elif i == 4: # Shape 5: Ellipse
        shape = patches.Ellipse((0.5, 0.5), width=0.7, height=0.4, facecolor=colors['shape5'])
    elif i == 5: # Shape 6: Arrow
        shape = patches.FancyArrow(0.2, 0.2, 0.6, 0.6, width=0.05, head_width=0.2, head_length=0.2, facecolor=colors['shape6'], edgecolor='none')
    elif i == 6: # Shape 7: Irregular Shape
        verts = [
            (0.5, 0.15), (0.3, 0.4), (0.25, 0.7), (0.35, 0.85),
            (0.5, 0.7), (0.65, 0.85), (0.75, 0.7), (0.7, 0.4)
        ]
        shape = patches.Polygon(verts, closed=True, facecolor=colors['shape7'])
    elif i == 7: # Shape 8: Rounded Rectangle
        shape = patches.FancyBboxPatch((0.4, 0.25), 0.2, 0.5,
                                       boxstyle=patches.BoxStyle("Round", pad=0, rounding_size=15),
                                       facecolor=colors['shape8'], edgecolor='none')
    elif i == 8: # Shape 9: Chevron
        ax.plot([0.2, 0.5, 0.8], [0.8, 0.2, 0.8], color=colors['shape9'], linewidth=10, solid_capstyle='butt')
    
    if shape:
        ax.add_patch(shape)

# 4. Final Layout Adjustments
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.2, hspace=0.5)
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/38160eab60da412796ea6ad1ad9a31f5.png", bbox_inches="tight")