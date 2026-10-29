import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set figure size
plt.figure(figsize=(10, 6))

# Define positions for the main horizontal line and arrowheads
line_x = np.linspace(0.1, 0.9, 100)
line_y = [0.45] * 100

# Plot the main horizontal line
plt.plot(line_x, line_y, color='red', linewidth=4, zorder=1)

# Add arrowhead at the left end
left_arrow = lines.Line2D([0.05, 0.1], [0.45, 0.45], color='red', linewidth=4, zorder=2)
plt.gca().add_line(left_arrow)

# Add arrowhead at the right end
right_arrow = lines.Line2D([0.9, 0.95], [0.45, 0.45], color='red', linewidth=4, zorder=2)
plt.gca().add_line(right_arrow)

# Add the "ISSUE" label at the right end
plt.text(0.95, 0.45, 'ISSUE', ha='center', va='center', fontsize=14, fontweight='bold', color='white', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='red', edgecolor='red'))

# Define positions for the boxes
box_positions = {
    'MONITORING': (0.15, 0.7),
    'DEVELOPMENT': (0.45, 0.7),
    'PLANNING': (0.75, 0.7),
    'MAINTENANCE': (0.15, 0.1),
    'DEPLOYMENT': (0.45, 0.1),
    'DESIGN': (0.75, 0.1)
}

# Define box styles
box_style = dict(boxstyle="round,pad=0.3", facecolor='red', edgecolor='black', linewidth=2)

# Add boxes with labels
for label, (x, y) in box_positions.items():
    plt.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', color='white', 
             bbox=box_style)

# Add text elements for each box
# MONITORING
plt.text(0.15, 0.6, 'Error tracking', ha='center', va='center', fontsize=9, color='black')
plt.text(0.15, 0.55, 'Performance metrics', ha='center', va='center', fontsize=9, color='black')
plt.text(0.15, 0.5, 'User feedback', ha='center', va='center', fontsize=9, color='black')

# DEVELOPMENT
plt.text(0.45, 0.6, 'Integration problems', ha='center', va='center', fontsize=9, color='black')
plt.text(0.45, 0.55, 'Coding errors', ha='center', va='center', fontsize=9, color='black')
plt.text(0.45, 0.5, 'Testing gaps', ha='center', va='center', fontsize=9, color='black')

# PLANNING
plt.text(0.75, 0.6, 'Resource allocation', ha='center', va='center', fontsize=9, color='black')
plt.text(0.75, 0.55, 'Budget constraints', ha='center', va='center', fontsize=9, color='black')
plt.text(0.75, 0.5, 'Timeline issues', ha='center', va='center', fontsize=9, color='black')
plt.text(0.75, 0.45, 'Scope changes', ha='center', va='center', fontsize=9, color='black')

# MAINTENANCE
plt.text(0.15, 0.15, 'Technical debt', ha='center', va='center', fontsize=9, color='black')
plt.text(0.15, 0.1, 'Bug fixes', ha='center', va='center', fontsize=9, color='black')
plt.text(0.15, 0.05, 'Updates', ha='center', va='center', fontsize=9, color='black')

# DEPLOYMENT
plt.text(0.45, 0.15, 'Security concerns', ha='center', va='center', fontsize=9, color='black')
plt.text(0.45, 0.1, 'Server configuration', ha='center', va='center', fontsize=9, color='black')
plt.text(0.45, 0.05, 'Database setup', ha='center', va='center', fontsize=9, color='black')

# DESIGN
plt.text(0.75, 0.15, 'Aesthetics', ha='center', va='center', fontsize=9, color='black')
plt.text(0.75, 0.1, 'Technical limitations', ha='center', va='center', fontsize=9, color='black')
plt.text(0.75, 0.05, 'User experience', ha='center', va='center', fontsize=9, color='black')

# Add arrows from boxes to the main line
# MONITORING
plt.annotate('', xy=(0.15, 0.45), xytext=(0.15, 0.6), arrowprops=dict(arrowstyle='->', color='black', lw=2))

# DEVELOPMENT
plt.annotate('', xy=(0.45, 0.45), xytext=(0.45, 0.6), arrowprops=dict(arrowstyle='->', color='black', lw=2))

# PLANNING
plt.annotate('', xy=(0.75, 0.45), xytext=(0.75, 0.6), arrowprops=dict(arrowstyle='->', color='black', lw=2))

# MAINTENANCE
plt.annotate('', xy=(0.15, 0.45), xytext=(0.15, 0.1), arrowprops=dict(arrowstyle='->', color='black', lw=2))

# DEPLOYMENT
plt.annotate('', xy=(0.45, 0.45), xytext=(0.45, 0.1), arrowprops=dict(arrowstyle='->', color='black', lw=2))

# DESIGN
plt.annotate('', xy=(0.75, 0.45), xytext=(0.75, 0.1), arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Set axis limits and remove ticks
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/39119959e24c42b490c9dfdcff8f6198.png", bbox_inches="tight")