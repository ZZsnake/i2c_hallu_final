import matplotlib.pyplot as plt
import networkx as nx

# Create directed graph
G = nx.DiGraph()

# Add nodes
nodes = [0, 1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Add directed edges based on observation
edges = [
    (0, 1),
    (1, 3),
    (3, 0),
    (0, 2),
    (2, 3),
    (2, 4),
    (4, 5),
]
G.add_edges_from(edges)

# Manually specify positions to mimic the layout in the image
pos = nx.spring_layout(G, seed=42)

# Figure and axis
fig, ax = plt.subplots(figsize=(10, 7))

# Draw nodes
node_colors = 'lightcoral'
trust_scores = {0: 1.0, 1: 0.4, 2: 0.8, 3: 0.8, 4: 0.4, 5: 0.4}
node_sizes = [trust_scores[n] * 1200 for n in G.nodes()]

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors='none',
    ax=ax
)

# Draw edges with arrows
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=edges,
    edge_color='#000080',
    width=2,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=18,
    node_size=node_sizes,
    ax=ax
)

# Draw labels
nx.draw_networkx_labels(
    G,
    pos,
    labels={n: str(n) for n in G.nodes()},
    font_size=12,
    font_color='k',
    font_weight='bold',
    ax=ax
)

# Title
ax.set_title("Trust Propagation in Directed Graph", fontsize=14)

# Remove axes
ax.set_axis_off()

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")
