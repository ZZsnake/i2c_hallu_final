import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch

# Figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
red = "#d62027"
line_color = "black"

# Backbone
backbone_y = 3
ax.plot([1, 9], [backbone_y, backbone_y], color=line_color, linewidth=2)

# Left triangle (start)
triangle = Polygon([[0.3, backbone_y],
                    [1.0, backbone_y + 0.7],
                    [1.0, backbone_y - 0.7]],
                   closed=True, facecolor=red, edgecolor=red)
ax.add_patch(triangle)

# Right capsule for ISSUE
issue_box = FancyBboxPatch((9, backbone_y - 0.9), 1.1, 1.8,
                           boxstyle="round,pad=0.2",
                           linewidth=0, facecolor=red)
ax.add_patch(issue_box)
ax.text(9 + 1.1/2, backbone_y, "ISSUE",
        ha="center", va="center", color="white",
        fontsize=12, fontweight="bold")

# Helper to draw category box
def draw_category_box(x, y, text):
    w, h = 1.4, 0.6
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="square,pad=0.05",
                         linewidth=0, facecolor=red)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")

# Helper to draw branch and its mini arrows
def draw_branch(anchor_x, anchor_y, end_x, end_y,
                causes, top=True):
    # Diagonal branch line
    ax.plot([anchor_x, end_x], [anchor_y, end_y],
            color=line_color, linewidth=1.6)

    # Compute orientation
    # We'll place arrows roughly parallel to x-axis, spaced along y.
    # For top branches, higher y for first cause; for bottom, lower y.
    n = len(causes)
    # Choose vertical start near end_y and step toward anchor_y
    if top:
        base_y = end_y + 0.2
        dy = -0.4
        text_va = "bottom"
    else:
        base_y = end_y - 0.2
        dy = 0.4
        text_va = "top"

    for i, cause in enumerate(causes):
        y = base_y + i * dy
        x_start = end_x - 2.8  # left
        x_end = end_x - 0.3    # near branch
        ax.annotate("",
                    xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="->",
                                    color=line_color,
                                    linewidth=1))
        ax.text(x_start - 0.1, y, cause,
                ha="right", va=text_va,
                fontsize=8, color="black")

# Branch anchors and endpoints
# Top: MONITORING, DEVELOPMENT, PLANNING
# Bottom: MAINTENANCE, DEPLOYMENT, DESIGN

# MONITORING (top left)
anchor_x = 2.4
anchor_y = backbone_y
end_x = 1.3
end_y = 4.5
draw_branch(anchor_x, anchor_y, end_x, end_y,
            ["Error tracking", "Performance metrics", "User feedback"],
            top=True)
draw_category_box(end_x, end_y + 0.7, "MONITORING")

# DEVELOPMENT (top center)
anchor_x = 4.6
anchor_y = backbone_y
end_x = 3.5
end_y = 4.5
draw_branch(anchor_x, anchor_y, end_x, end_y,
            ["Integration problems", "Coding errors", "Testing gaps"],
            top=True)
draw_category_box(end_x, end_y + 0.7, "DEVELOPMENT")

# PLANNING (top right)
anchor_x = 6.8
anchor_y = backbone_y
end_x = 5.7
end_y = 4.5
draw_branch(anchor_x, anchor_y, end_x, end_y,
            ["Resource allocation", "Budget constraints",
             "Timeline issues", "Scope changes"],
            top=True)
draw_category_box(end_x, end_y + 0.7, "PLANNING")

# MAINTENANCE (bottom left)
anchor_x = 2.4
anchor_y = backbone_y
end_x = 1.3
end_y = 1.5
draw_branch(anchor_x, anchor_y, end_x, end_y,
            ["Technical debt", "Bug fixes", "Updates"],
            top=False)
draw_category_box(end_x, end_y - 0.7, "MAINTENANCE")

# DEPLOYMENT (bottom center)
anchor_x = 4.6
anchor_y = backbone_y
end_x = 3.5
end_y = 1.5
draw_branch(anchor_x, anchor_y, end_x, end_y,
            ["Security concerns", "Server configuration", "Database setup"],
            top=False)
draw_category_box(end_x, end_y - 0.7, "DEPLOYMENT")

# DESIGN (bottom right)
anchor_x = 6.8
anchor_y = backbone_y
end_x = 5.7
end_y = 1.5
draw_branch(anchor_x, anchor_y, end_x, end_y,
            ["Aesthetics", "Technical limitations", "User experience"],
            top=False)
draw_category_box(end_x, end_y - 0.7, "DESIGN")

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/39119959e24c42b490c9dfdcff8f6198.png", bbox_inches="tight")