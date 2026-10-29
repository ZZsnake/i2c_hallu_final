import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.patch.set_facecolor('white')
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Node positions (normalized coordinates)
top = (0.5, 0.8)
bottom_left = (0.2, 0.2)
bottom_right = (0.8, 0.2)

# Define graph configurations for each panel
# Format: (label, [(from_node, to_node), ...])
configs = [
    ('A1', []),
    ('A2', [(bottom_left, bottom_right)]),
    ('B1', [(bottom_left, bottom_right), (bottom_right, bottom_left)]),
    ('B2', [(top, bottom_left), (top, bottom_right)]),
    ('C1', [(bottom_left, top), (bottom_right, top)]),
    ('C2', [(top, bottom_left), (bottom_left, bottom_right)]),
    ('D1', [(top, bottom_left), (bottom_left, bottom_right), (bottom_right, bottom_left)]),
    ('D2', [(top, bottom_left), (bottom_left, bottom_right), (bottom_right, bottom_left)]),
    ('E1', [(top, bottom_left), (bottom_left, bottom_right), (top, bottom_right)]),
    ('E2', [(top, bottom_left), (top, bottom_right), (bottom_right, bottom_left)]),
    ('F1', [(top, bottom_left), (bottom_left, bottom_right), (bottom_right, bottom_left)]),
    ('F2', [(top, bottom_left), (bottom_left, bottom_right), (bottom_right, top)]),
    ('G1', [(bottom_left, top), (bottom_right, top), (bottom_left, bottom_right), (bottom_right, bottom_left)]),
    ('G2', [(top, bottom_left), (bottom_left, bottom_right), (bottom_right, top)]),
    ('H1', [(bottom_left, top), (bottom_right, top), (bottom_left, bottom_right), (bottom_right, bottom_left)]),
    ('H2', [(top, bottom_left), (bottom_left, bottom_right), (bottom_right, top)])
]

for idx, (label, edges) in enumerate(configs):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw background
    rect = mpatches.Rectangle((0, 0), 1, 1, linewidth=2, 
                              edgecolor='black', facecolor='#f5f5f5')
    ax.add_patch(rect)
    
    # Draw edges (arrows)
    for start, end in edges:
        arrow = FancyArrowPatch(start, end, 
                               arrowstyle='->', 
                               mutation_scale=20, 
                               linewidth=2.5, 
                               color='#1f3a93',
                               zorder=1)
        ax.add_patch(arrow)
    
    # Draw nodes
    for node_pos in [top, bottom_left, bottom_right]:
        circle = Circle(node_pos, 0.04, color='#c41e3a', zorder=2)
        ax.add_patch(circle)
    
    # Draw label
    label_circle = Circle((0.5, 0.5), 0.12, 
                         color='white', 
                         edgecolor='#2d5016', 
                         linewidth=2.5, 
                         zorder=3)
    ax.add_patch(label_circle)
    ax.text(0.5, 0.5, label, 
           ha='center', va='center', 
           fontsize=12, fontweight='bold', 
           zorder=4)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/148c74213a904740a2b625f24a275ed8.png", bbox_inches="tight")