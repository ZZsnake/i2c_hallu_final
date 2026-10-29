import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set figure size
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Define positions for the main horizontal line and arrowheads
line_x = np.linspace(0.1, 0.9, 100)
line_y = [0.45] * 100

# Plot the main horizontal line
plt.plot(line_x, line_y, color='tab:red', linewidth=4, zorder=1)

# Add large triangular tail at the left end
left_tail = patches.Polygon([[0.02, 0.37], [0.10, 0.45], [0.02, 0.53]], closed=True,
                            facecolor='tab:red', edgecolor='tab:red', zorder=2)
ax.add_patch(left_tail)

# Add red semicircular head at the right end
right_head = patches.Wedge((0.90, 0.45), 0.10, -90, 90, facecolor='tab:red', edgecolor='tab:red', zorder=2)
ax.add_patch(right_head)

# Add the "ISSUE" label inside the right semicircular head
plt.text(0.935, 0.45, 'ISSUE', ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=3)

# Define positions for the boxes
box_positions = {
    'MONITORING': (0.18, 0.82),
    'DEVELOPMENT': (0.43, 0.82),
    'PLANNING': (0.68, 0.82),
    'MAINTENANCE': (0.18, 0.08),
    'DEPLOYMENT': (0.43, 0.08),
    'DESIGN': (0.68, 0.08)
}

# Define box styles
box_style = dict(boxstyle="square,pad=0.8", facecolor='tab:red', edgecolor='black', linewidth=1.5)

# Add boxes with labels
for label, (x, y) in box_positions.items():
    plt.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white',
             bbox=box_style, zorder=5)

# Add diagonal arrows from boxes to the main line
# MONITORING
plt.annotate('', xy=(0.36, 0.45), xytext=(0.195, 0.78),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2), zorder=4)

# DEVELOPMENT
plt.annotate('', xy=(0.61, 0.45), xytext=(0.445, 0.78),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2), zorder=4)

# PLANNING
plt.annotate('', xy=(0.86, 0.45), xytext=(0.695, 0.78),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2), zorder=4)

# MAINTENANCE
plt.annotate('', xy=(0.36, 0.45), xytext=(0.195, 0.12),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2), zorder=4)

# DEPLOYMENT
plt.annotate('', xy=(0.61, 0.45), xytext=(0.445, 0.12),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2), zorder=4)

# DESIGN
plt.annotate('', xy=(0.86, 0.45), xytext=(0.695, 0.12),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2), zorder=4)

# Helper for cause text with small horizontal arrows pointing into the diagonal bones
def cause_label(text, x_text, y, x_arrow_end):
    plt.text(x_text, y, text, ha='right', va='center', fontsize=9, color='black', zorder=6)
    plt.annotate('', xy=(x_arrow_end, y), xytext=(x_text + 0.005, y),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.0), zorder=6)

# Add text elements and individual arrows for each cause
# MONITORING
cause_label('Error tracking', 0.18, 0.66, 0.255)
cause_label('Performance metrics', 0.205, 0.61, 0.280)
cause_label('User feedback', 0.23, 0.56, 0.305)

# DEVELOPMENT
cause_label('Integration problems', 0.43, 0.66, 0.505)
cause_label('Coding errors', 0.455, 0.61, 0.530)
cause_label('Testing gaps', 0.48, 0.56, 0.555)

# PLANNING
cause_label('Resource allocation', 0.68, 0.71, 0.735)
cause_label('Budget constraints', 0.705, 0.66, 0.760)
cause_label('Timeline issues', 0.73, 0.61, 0.785)
cause_label('Scope changes', 0.755, 0.56, 0.810)

# MAINTENANCE
cause_label('Technical debt', 0.23, 0.33, 0.305)
cause_label('Bug fixes', 0.205, 0.28, 0.280)
cause_label('Updates', 0.18, 0.23, 0.255)

# DEPLOYMENT
cause_label('Security concerns', 0.48, 0.33, 0.555)
cause_label('Server configuration', 0.455, 0.28, 0.530)
cause_label('Database setup', 0.43, 0.23, 0.505)

# DESIGN
cause_label('Aesthetics', 0.73, 0.33, 0.805)
cause_label('Technical limitations', 0.705, 0.28, 0.780)
cause_label('User experience', 0.68, 0.23, 0.755)

# Set axis limits and remove ticks
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/39119959e24c42b490c9dfdcff8f6198.png")
