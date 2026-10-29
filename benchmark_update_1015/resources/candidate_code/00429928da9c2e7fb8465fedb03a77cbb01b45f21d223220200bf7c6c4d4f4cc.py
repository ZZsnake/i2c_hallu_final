import matplotlib.pyplot as plt
import networkx as nx

# Create a Morse Code Prefix Tree
G = nx.DiGraph()

# Add nodes
letters = [
    'root', 'e', 't', 'i', 'a', 'n', 'm', 's', 'u', 'r', 'w',
    'd', 'k', 'g', 'o', 'h', 'v', 'f', 'l', 'p', 'j', 'b', 'x',
    'c', 'y', 'z', 'q'
]
G.add_nodes_from(letters)

# Add Morse prefix-tree edges with dot/dash labels
edges = [
    ('root', 'e', '•'), ('root', 't', '−'),
    ('e', 'i', '•'), ('e', 'a', '−'),
    ('t', 'n', '•'), ('t', 'm', '−'),
    ('i', 's', '•'), ('i', 'u', '−'),
    ('a', 'r', '•'), ('a', 'w', '−'),
    ('n', 'd', '•'), ('n', 'k', '−'),
    ('m', 'g', '•'), ('m', 'o', '−'),
    ('s', 'h', '•'), ('s', 'v', '−'),
    ('u', 'f', '•'),
    ('r', 'l', '•'),
    ('w', 'p', '•'), ('w', 'j', '−'),
    ('d', 'b', '•'), ('d', 'x', '−'),
    ('k', 'c', '•'), ('k', 'y', '−'),
    ('g', 'z', '•'), ('g', 'q', '−')
]
G.add_edges_from((u, v) for u, v, _ in edges)
edge_labels = {(u, v): label for u, v, label in edges}

# Set node positions in left-to-right Morse-code depth layers
pos = {
    'root': (0.0, -0.25),
    'e': (2.0, 0.25), 't': (2.0, -0.25),
    'n': (4.0, 0.75), 'a': (4.0, 0.25), 'm': (4.0, -0.25), 'i': (4.0, -0.75),
    'r': (6.0, 1.75), 'w': (6.0, 1.25), 'k': (6.0, 0.75), 'u': (6.0, 0.25),
    's': (6.0, -0.25), 'd': (6.0, -0.75), 'o': (6.0, -1.25), 'g': (6.0, -1.75),
    'q': (8.0, 2.75), 'h': (8.0, 2.20), 'j': (8.0, 1.65), 'f': (8.0, 1.10),
    'y': (8.0, 0.55), 'z': (8.0, 0.00), 'v': (8.0, -0.55), 'b': (8.0, -1.10),
    'p': (8.0, -1.65), 'c': (8.0, -2.20), 'x': (8.0, -2.75), 'l': (8.0, -3.30)
}

# Draw the graph
plt.figure(figsize=(14, 10))
labels = {node: ('' if node == 'root' else node) for node in G.nodes()}

nx.draw_networkx_edges(
    G, pos,
    edge_color='#a94442',
    arrows=True,
    arrowstyle='-|>',
    arrowsize=16,
    width=2.0,
    connectionstyle='arc3,rad=0.0'
)
nx.draw_networkx_nodes(
    G, pos,
    node_color='darkblue',
    edgecolors='navy',
    node_size=850
)
nx.draw_networkx_labels(
    G, pos,
    labels=labels,
    font_size=12,
    font_weight='bold',
    font_color='white'
)
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_color='green',
    font_size=14,
    font_weight='bold',
    label_pos=0.5,
    bbox=dict(facecolor='white', edgecolor='none', alpha=1.0, pad=0.0)
)

# Show the plot
plt.title('Morse Code Prefix Tree Structure', fontsize=16, fontweight='bold')
plt.axis('off')
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/9839a26766c1406fba968d2010bb999a.png", bbox_inches="tight")
