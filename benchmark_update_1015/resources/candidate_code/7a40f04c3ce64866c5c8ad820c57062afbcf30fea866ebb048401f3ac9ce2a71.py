import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

# Set figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Set background color
ax.set_facecolor('white')

# Remove axes
ax.axis('off')

# Define colors
red = '#D62728'
white = 'white'
black = 'black'

# Draw spine (horizontal red line)
spine_start = 0.1
spine_end = 0.9
spine_y = 0.5
ax.plot([spine_start, spine_end], [spine_y, spine_y], color=red, linewidth=2)

# Draw head (triangle on left)
head_x = spine_start - 0.05
head_y = spine_y
head_width = 0.1
head_height = 0.1
head = patches.Polygon([[head_x, head_y], [head_x + head_width, head_y - head_height/2], 
                        [head_x + head_width, head_y + head_height/2]], color=red)
ax.add_patch(head)

# Draw issue oval on right
issue_x = spine_end + 0.05
issue_y = spine_y
issue_width = 0.15
issue_height = 0.15
issue_oval = patches.Ellipse((issue_x, issue_y), issue_width, issue_height, color=red)
ax.add_patch(issue_oval)
ax.text(issue_x, issue_y, "ISSUE", color=white, fontsize=12, ha='center', va='center', fontweight='bold')

# Define category positions
categories = [
    ("MONITORING", 0.25, 0.7),
    ("DEVELOPMENT", 0.45, 0.7),
    ("PLANNING", 0.65, 0.7),
    ("MAINTENANCE", 0.25, 0.3),
    ("DEPLOYMENT", 0.45, 0.3),
    ("DESIGN", 0.65, 0.3)
]

# Draw category boxes and sub-causes
for name, x_pos, y_pos in categories:
    # Draw category box
    box_width = 0.15
    box_height = 0.05
    box = patches.Rectangle((x_pos - box_width/2, y_pos - box_height/2), box_width, box_height, color=red)
    ax.add_patch(box)
    ax.text(x_pos, y_pos, name, color=white, fontsize=10, ha='center', va='center', fontweight='bold')

    # Define sub-causes for each category
    subcauses = {
        "MONITORING": ["Error tracking", "Performance metrics", "User feedback"],
        "DEVELOPMENT": ["Integration problems", "Coding errors", "Testing gaps"],
        "PLANNING": ["Resource allocation", "Budget constraints", "Timeline issues", "Scope changes"],
        "MAINTENANCE": ["Technical debt", "Bug fixes", "Updates"],
        "DEPLOYMENT": ["Security concerns", "Server configuration", "Database setup"],
        "DESIGN": ["Aesthetics", "Technical limitations", "User experience"]
    }

    # Draw sub-causes and arrows
    for i, subcause in enumerate(subcauses[name]):
        # Position sub-cause text
        if y_pos > 0.5:  # Top categories
            text_y = y_pos - 0.08 - i * 0.04
            arrow_start_y = text_y + 0.02
            arrow_end_y = spine_y + 0.01
        else:  # Bottom categories
            text_y = y_pos + 0.08 + i * 0.04
            arrow_start_y = text_y - 0.02
            arrow_end_y = spine_y - 0.01

        # Draw arrow from sub-cause to spine
        arrow = mlines.Line2D([x_pos - 0.1, x_pos], [arrow_start_y, arrow_end_y], color=black, linewidth=1.5)
        ax.add_line(arrow)

        # Draw sub-cause text
        ax.text(x_pos - 0.15, text_y, subcause, color=black, fontsize=9, ha='right', va='center')

# Adjust layout to prevent clipping
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/39119959e24c42b490c9dfdcff8f6198.png", bbox_inches="tight")