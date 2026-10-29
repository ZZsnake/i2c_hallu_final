import matplotlib.pyplot as plt
import networkx as nx

# Create directed graph
G = nx.DiGraph()

# Add nodes
nodes = [0, 1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Add edges based on observed structure
edges = [
    (0, 1),
    (1, 3),
    (3, 0),
    (0, 2),
    (2, 3),
    (2, 4),
    (4, 5)
]
G.add_edges_from(edges)

# Manually specify positions to visually match the image
pos = {
    0: (0.7, 0.3),
    1: (0.75, 0.05),
    3: (0.35, 0.2),
    2: (0.55, 0.45),
    4: (0.75, 0.7),
    5: (0.1, 0.75)
}

# Node sizes: 0 largest, 2 medium, others smaller
node_sizes = {
    0: 1500,
    2: 1200,
    1: 900,
    3: 900,
    4: 900,
    5: 900
}
sizes = [node_sizes[n] for n in G.nodes()]

# Colors
node_color = '#ff9999'  # light pink
edge_color = 'navy'
font_color = 'white'

# Create figure
plt.figure(figsize=(9, 7))

# Draw edges with arrows
nx.draw_networkx_edges(
    G,
    pos,
    edge_color=edge_color,
    arrows=True,
    arrowsize=15,
    arrowstyle='-|>',
    width=1.5
)

# Draw nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=sizes,
    node_color=node_color,
    edgecolors='#cc6666',
    linewidths=1.5
)

# Draw labels
nx.draw_networkx_labels(
    G,
    pos,
    labels={n: str(n) for n in G.nodes()},
    font_color=font_color,
    font_size=10
)

# Title
plt.title("Trust Propagation in Directed Graph", fontsize=14)

# Remove axes
plt.axis('off')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")