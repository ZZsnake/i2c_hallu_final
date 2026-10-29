import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
red_color = '#C8352E'
black_color = '#000000'
white_color = '#FFFFFF'

# Draw left arrow head
arrow_head = Polygon([[0.3, 3], [0.8, 2.2], [0.8, 3.8]], 
                      facecolor=red_color, edgecolor=red_color, zorder=3)
ax.add_patch(arrow_head)

# Draw central spine
ax.plot([0.8, 9.0], [3, 3], color=red_color, linewidth=3, zorder=2)

# Draw right issue box
issue_box = FancyBboxPatch((9.0, 2.4), 0.9, 1.2, 
                           boxstyle="round,pad=0.05", 
                           facecolor=red_color, edgecolor=red_color, zorder=3)
ax.add_patch(issue_box)
ax.text(9.45, 3, 'ISSUE', ha='center', va='center', 
        fontsize=13, fontweight='bold', color=white_color, zorder=4)

# Category positions and data
categories = [
    # Top row
    {'name': 'MONITORING', 'box_pos': (1.3, 4.8), 'branch_start': (1.9, 3), 
     'branch_end': (1.9, 4.7), 'causes': ['Error tracking', 'Performance metrics', 'User feedback'],
     'side': 'top'},
    {'name': 'DEVELOPMENT', 'box_pos': (3.5, 4.8), 'branch_start': (4.2, 3), 
     'branch_end': (4.2, 4.7), 'causes': ['Integration problems', 'Coding errors', 'Testing gaps'],
     'side': 'top'},
    {'name': 'PLANNING', 'box_pos': (6.2, 4.8), 'branch_start': (7.0, 3), 
     'branch_end': (7.0, 4.7), 'causes': ['Resource allocation', 'Budget constraints', 'Timeline issues', 'Scope changes'],
     'side': 'top'},
    # Bottom row
    {'name': 'MAINTENANCE', 'box_pos': (1.2, 0.3), 'branch_start': (1.9, 3), 
     'branch_end': (1.9, 1.3), 'causes': ['Technical debt', 'Bug fixes', 'Updates'],
     'side': 'bottom'},
    {'name': 'DEPLOYMENT', 'box_pos': (3.6, 0.3), 'branch_start': (4.2, 3), 
     'branch_end': (4.2, 1.3), 'causes': ['Security concerns', 'Server configuration', 'Database setup'],
     'side': 'bottom'},
    {'name': 'DESIGN', 'box_pos': (6.5, 0.3), 'branch_start': (7.0, 3), 
     'branch_end': (7.0, 1.3), 'causes': ['Aesthetics', 'Technical limitations', 'User experience'],
     'side': 'bottom'},
]

# Draw categories
for cat in categories:
    # Draw main branch line
    ax.plot([cat['branch_start'][0], cat['branch_end'][0]], 
            [cat['branch_start'][1], cat['branch_end'][1]], 
            color=black_color, linewidth=1.5, zorder=1)
    
    # Draw category box
    box = FancyBboxPatch(cat['box_pos'], 1.2, 0.5, 
                         boxstyle="square,pad=0.02", 
                         facecolor=red_color, edgecolor=red_color, zorder=3)
    ax.add_patch(box)
    ax.text(cat['box_pos'][0] + 0.6, cat['box_pos'][1] + 0.25, cat['name'], 
            ha='center', va='center', fontsize=9, fontweight='bold', 
            color=white_color, zorder=4)
    
    # Draw sub-causes
    num_causes = len(cat['causes'])
    for i, cause in enumerate(cat['causes']):
        if cat['side'] == 'top':
            # Calculate positions for top branches
            y_branch = cat['branch_end'][1] - 0.2 - (i * 0.4)
            x_start = cat['branch_end'][0]
            x_end = x_start - 0.6
            y_end = y_branch + 0.3
            
            # Draw sub-cause line
            ax.plot([x_start, x_end], [y_branch, y_end], 
                    color=black_color, linewidth=1, zorder=1)
            
            # Draw text
            ax.text(x_end - 0.05, y_end, cause, ha='right', va='center', 
                    fontsize=8, color=black_color)
        else:
            # Calculate positions for bottom branches
            y_branch = cat['branch_end'][1] + 0.2 + (i * 0.4)
            x_start = cat['branch_end'][0]
            x_end = x_start - 0.6
            y_end = y_branch - 0.3
            
            # Draw sub-cause line
            ax.plot([x_start, x_end], [y_branch, y_end], 
                    color=black_color, linewidth=1, zorder=1)
            
            # Draw text
            ax.text(x_end - 0.05, y_end, cause, ha='right', va='center', 
                    fontsize=8, color=black_color)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/39119959e24c42b490c9dfdcff8f6198.png", bbox_inches="tight")