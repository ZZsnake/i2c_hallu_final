import matplotlib.pyplot as plt
import numpy as np

# Set the figure size as required
plt.figure(figsize=(14, 10))

# Define the nodes (letters) and their positions as vertical prefix-tree layers
nodes = {
    'root': (0.0, 0.0),
    'e': (2.4, -0.35),
    't': (2.4, -1.15),
    'n': (4.9, 0.45),
    'a': (4.9, -0.35),
    'm': (4.9, -1.15),
    'i': (4.9, -1.90),
    'r': (7.6, 2.10),
    'w': (7.6, 1.35),
    'k': (7.6, 0.45),
    'u': (7.6, -0.35),
    's': (7.6, -1.15),
    'd': (7.6, -1.90),
    'o': (7.6, -2.65),
    'g': (7.6, -3.40),
    'q': (10.4, 3.65),
    'h': (10.4, 2.85),
    'j': (10.4, 2.10),
    'f': (10.4, 1.35),
    'y': (10.4, 0.45),
    'z': (10.4, -0.35),
    'v': (10.4, -1.15),
    'b': (10.4, -1.90),
    'p': (10.4, -2.65),
    'c': (10.4, -3.40),
    'x': (10.4, -4.15),
    'l': (10.4, -4.90)
}

# Define the sparse Morse-code prefix-tree edges with their dot/dash labels
edges = [
    ('root', 'e', '•'),
    ('root', 't', '–'),
    ('t', 'n', '•'),
    ('e', 'a', '–'),
    ('t', 'm', '–'),
    ('e', 'i', '•'),
    ('a', 'r', '•'),
    ('a', 'w', '–'),
    ('n', 'k', '–'),
    ('i', 'u', '–'),
    ('i', 's', '•'),
    ('n', 'd', '•'),
    ('m', 'o', '–'),
    ('m', 'g', '•'),
    ('g', 'q', '–'),
    ('s', 'h', '•'),
    ('w', 'j', '–'),
    ('u', 'f', '•'),
    ('k', 'y', '–'),
    ('g', 'z', '•'),
    ('s', 'v', '–'),
    ('d', 'b', '•'),
    ('w', 'p', '•'),
    ('k', 'c', '•'),
    ('d', 'x', '–'),
    ('r', 'l', '•')
]

node_color = '#11118f'
edge_color = '#a94442'
label_color = 'darkgreen'

# Draw directed edges first so nodes sit on top
for start_node, end_node, symbol in edges:
    start_x, start_y = nodes[start_node]
    end_x, end_y = nodes[end_node]

    plt.annotate(
        '',
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops=dict(
            arrowstyle='-|>',
            color=edge_color,
            linewidth=2,
            mutation_scale=14,
            shrinkA=14,
            shrinkB=14
        ),
        zorder=3
    )

    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2
    dx = end_x - start_x
    dy = end_y - start_y
    length = np.hypot(dx, dy)
    if length:
        off_x = -dy / length * 0.08
        off_y = dx / length * 0.08
    else:
        off_x = off_y = 0

    plt.text(
        mid_x,
        mid_y,
        symbol,
        fontsize=15,
        ha='center',
        va='center',
        color=label_color,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='white'),
        zorder=4
    )

# Draw nodes (dark navy-blue circles)
for node, (x, y) in nodes.items():
    if node == 'root':
        plt.scatter(x, y, color=node_color, s=800, edgecolors='face', zorder=5)
    else:
        plt.scatter(x, y, color=node_color, s=800, edgecolors='face', zorder=5)

# Add labels for nodes (except root)
for node, (x, y) in nodes.items():
    if node != 'root':
        plt.text(x, y, node, fontsize=12, ha='center', va='center', weight='bold', color='white', zorder=6)

# Add title
plt.title('Morse Code Prefix Tree Structure', fontsize=16, fontweight='bold', pad=20)

# Set axis limits
plt.xlim(-0.9, 11.3)
plt.ylim(-5.6, 4.4)

# Remove axes
plt.axis('off')

# Adjust layout to prevent clipping
plt.tight_layout()

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/9839a26766c1406fba968d2010bb999a.png")
