import matplotlib.pyplot as plt
import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes
nodes = list(range(9))
G.add_nodes_from(nodes)

# Define MST edges and other edges with weights
mst_edges = [
    (0, 7, 6), (0, 1, 2), (1, 2, 7), (2, 8, 4),
    (2, 5, 5), (2, 3, 8), (5, 6, 3), (5, 4, 9)
]
other_edges = [(1, 7, 9), (6, 8, 5), (7, 8, 10), (3, 4, 11), (3, 5, 12)]

# Add edges to the graph
G.add_weighted_edges_from(mst_edges + other_edges)

# Position nodes
pos = {
    0: (0.88, 0.70),
    1: (0.43, 0.12),
    2: (-0.08, -0.30),
    3: (-0.40, -0.86),
    4: (-0.72, -1.10),
    5: (-0.68, -0.66),
    6: (-0.95, 0.10),
    7: (0.28, 0.55),
    8: (-0.28, 0.35),
}

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=650, node_color='coral', edgecolors='none')

# Draw MST edges
nx.draw_networkx_edges(G, pos, edgelist=mst_edges, width=3, edge_color='navy')

# Draw non-MST edges
nx.draw_networkx_edges(G, pos, edgelist=other_edges, style='dashed', alpha=0.5, edge_color='gray')

# Draw edge labels
edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, bbox=dict(facecolor='none', edgecolor='none'))

# Draw node labels
node_labels = {node: str(node) for node in nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=15, font_weight='bold')

# Title
plt.title('Graph Theory: Minimum Spanning Tree', fontsize=16)

# Display the plot
plt.axis('off')
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/05869359db314898a9c1967c46236354.png")
