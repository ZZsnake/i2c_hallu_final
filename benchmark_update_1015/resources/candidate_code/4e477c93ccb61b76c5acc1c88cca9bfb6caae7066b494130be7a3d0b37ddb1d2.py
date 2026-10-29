import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import numpy as np

# Set figure size
plt.figure(figsize=(10, 6))

# Central spine (horizontal red arrow)
spine_start = (0.1, 0.5)
spine_end = (0.9, 0.5)
plt.arrow(spine_start[0], spine_start[1], spine_end[0]-spine_start[0], 0, 
          head_width=0.03, head_length=0.03, fc='red', ec='red', linewidth=3)

# Tail (red triangle on left)
tail = patches.Polygon([[0.05, 0.45], [0.05, 0.55], [0.1, 0.5]], closed=True, color='red')
plt.gca().add_patch(tail)

# Head (red semi-circle on right)
head = patches.Wedge(center=(0.9, 0.5), r=0.05, theta1=90, theta2=270, color='red')
plt.gca().add_patch(head)
plt.text(0.91, 0.5, 'ISSUE', ha='left', va='center', color='white', fontweight='bold', fontsize=10)

# Category boxes and branches
categories = [
    # Top branches
    {'name': 'MONITORING', 'y': 0.7, 'causes': ['Error tracking', 'Performance metrics', 'User feedback'], 'side': 'top'},
    {'name': 'DEVELOPMENT', 'y': 0.65, 'causes': ['Integration problems', 'Coding errors', 'Testing gaps'], 'side': 'top'},
    {'name': 'PLANNING', 'y': 0.6, 'causes': ['Resource allocation', 'Budget constraints', 'Timeline issues', 'Scope changes'], 'side': 'top'},
    # Bottom branches
    {'name': 'MAINTENANCE', 'y': 0.3, 'causes': ['Technical debt', 'Bug fixes', 'Updates'], 'side': 'bottom'},
    {'name': 'DEPLOYMENT', 'y': 0.35, 'causes': ['Security concerns', 'Server configuration', 'Database setup'], 'side': 'bottom'},
    {'name': 'DESIGN', 'y': 0.4, 'causes': ['Aesthetics', 'Technical limitations', 'User experience'], 'side': 'bottom'}
]

# Draw category boxes and branches
for cat in categories:
    # Category box
    box = patches.Rectangle((0.15, cat['y']-0.03), 0.1, 0.06, color='red', ec='black')
    plt.gca().add_patch(box)
    plt.text(0.15+0.05, cat['y'], cat['name'], ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    
    # Branch line
    branch_start = (0.25, cat['y'])
    branch_end = (0.9, 0.5)
    plt.plot([branch_start[0], branch_end[0]], [branch_start[1], branch_end[1]], 'k-', linewidth=0.5)
    
    # Cause labels with arrows
    num_causes = len(cat['causes'])
    for i, cause in enumerate(cat['causes']):
        # Position along branch
        t = (i + 0.5) / num_causes
        x = branch_start[0] + t * (branch_end[0] - branch_start[0])
        y = branch_start[1] + t * (branch_end[1] - branch_start[1])
        
        # Offset for text (left for top, right for bottom)
        if cat['side'] == 'top':
            text_x = x - 0.05
            text_y = y + 0.01
            arrow_dx = 0.01
            arrow_dy = -0.01
        else:
            text_x = x + 0.05
            text_y = y - 0.01
            arrow_dx = -0.01
            arrow_dy = 0.01
        
        plt.text(text_x, text_y, cause, ha='right' if cat['side'] == 'top' else 'left', va='center', fontsize=8)
        plt.arrow(x, y, arrow_dx, arrow_dy, head_width=0.005, head_length=0.005, fc='black', ec='black', linewidth=0.5)

# Remove axes
plt.axis('off')

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/39119959e24c42b490c9dfdcff8f6198.png", bbox_inches="tight")