import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- Data and Configuration ---
# This dictionary holds the content and layout information for the diagram.
ishikawa_data = {
    "issue": "ISSUE",
    "categories": {
        "top": [
            {
                "name": "MONITORING",
                "causes": ["Error tracking", "Performance metrics", "User feedback"],
                "start": (2.5, 5.2), "end": (3.5, 3.0), "sub_bone_len": 1.2
            },
            {
                "name": "DEVELOPMENT",
                "causes": ["Integration problems", "Coding errors", "Testing gaps"],
                "start": (4.8, 5.2), "end": (5.8, 3.0), "sub_bone_len": 1.4
            },
            {
                "name": "PLANNING",
                "causes": ["Resource allocation", "Budget constraints", "Timeline issues", "Scope changes"],
                "start": (7.1, 5.2), "end": (8.1, 3.0), "sub_bone_len": 1.6
            }
        ],
        "bottom": [
            {
                "name": "MAINTENANCE",
                "causes": ["Technical debt", "Bug fixes", "Updates"],
                "start": (2.5, 0.8), "end": (3.5, 3.0), "sub_bone_len": 1.2
            },
            {
                "name": "DEPLOYMENT",
                "causes": ["Security concerns", "Server configuration", "Database setup"],
                "start": (4.8, 0.8), "end": (5.8, 3.0), "sub_bone_len": 1.4
            },
            {
                "name": "DESIGN",
                "causes": ["Aesthetics", "Technical limitations", "User experience"],
                "start": (7.1, 0.8), "end": (8.1, 3.0), "sub_bone_len": 1.6
            }
        ]
    },
    "style": {
        "main_color": "#c00000",
        "text_color": "#222222",
        "spine_y": 3.0,
        "spine_x_range": (0.5, 9.5)
    }
}

def draw_branch(ax, config, is_top):
    """Helper function to draw a main branch with its sub-branches and labels."""
    style = ishikawa_data["style"]
    start_pos = config["start"]
    end_pos = config["end"]
    
    # Main bone (arrow)
    ax.arrow(start_pos[0], start_pos[1], end_pos[0] - start_pos[0], end_pos[1] - start_pos[1],
             head_width=0.1, head_length=0.15, fc=style["text_color"], ec=style["text_color"],
             length_includes_head=True, lw=0.8, width=0.001)

    # Category label box
    label_y_offset = 0.4 if is_top else -0.4
    va = 'bottom' if is_top else 'top'
    ax.text(start_pos[0], start_pos[1] + label_y_offset, config["name"],
            ha='center', va=va, color='white', fontweight='bold', fontsize=9,
            bbox=dict(facecolor=style["main_color"], edgecolor=style["main_color"], boxstyle='square,pad=0.4'))

    # Sub-bones and cause labels
    num_causes = len(config["causes"])
    bone_points_x = np.linspace(start_pos[0], end_pos[0], num_causes + 2)[1:-1]
    bone_points_y = np.linspace(start_pos[1], end_pos[1], num_causes + 2)[1:-1]

    for i, cause in enumerate(config["causes"]):
        sub_bone_end = (bone_points_x[i], bone_points_y[i])
        sub_bone_start_x = sub_bone_end[0] - config["sub_bone_len"]

        # Sub-bone arrow
        ax.arrow(sub_bone_start_x, sub_bone_end[1], config["sub_bone_len"], 0,
                 head_width=0.08, head_length=0.1, fc=style["text_color"], ec=style["text_color"],
                 length_includes_head=True, lw=0.7, width=0.001)

        # Cause text
        ax.text(sub_bone_start_x - 0.1, sub_bone_end[1], cause,
                ha='right', va='center', fontsize=9, color=style["text_color"])

# --- Main Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 6)
ax.axis('off')

style = ishikawa_data["style"]
spine_y = style["spine_y"]
spine_start_x, spine_end_x = style["spine_x_range"]

# Draw the central spine
ax.plot([spine_start_x, spine_end_x], [spine_y, spine_y], color=style["main_color"], linewidth=3)

# Draw the tail (triangle)
tail_points = [[spine_start_x - 0.5, spine_y - 0.3], [spine_start_x, spine_y], [spine_start_x - 0.5, spine_y + 0.3]]
tail = patches.Polygon(tail_points, closed=True, facecolor=style["main_color"], edgecolor=style["main_color"])
ax.add_patch(tail)

# Draw the head (semi-circle with text)
issue_radius = 0.5
wedge = patches.Wedge((spine_end_x, spine_y), issue_radius, -90, 90,
                      facecolor=style["main_color"], edgecolor=style["main_color"])
ax.add_patch(wedge)
ax.text(spine_end_x, spine_y, ishikawa_data["issue"], color='white',
        ha='center', va='center', fontsize=10, fontweight='bold')

# Draw all branches
for category_config in ishikawa_data["categories"]["top"]:
    draw_branch(ax, category_config, is_top=True)

for category_config in ishikawa_data["categories"]["bottom"]:
    draw_branch(ax, category_config, is_top=False)
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/39119959e24c42b490c9dfdcff8f6198.png", bbox_inches="tight")