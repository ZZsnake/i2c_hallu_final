import matplotlib.pyplot as plt
import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes
nodes = list(range(9))
G.add_nodes_from(nodes)

# Define MST edges and other edges with weights
mst_edges = [
    (0, 7, 6), (7, 1, 9), (1, 2, 7), (2, 8, 2), 
    (8, 6, 5), (6, 5, 3), (5, 4, 11), (4, 3, 12)
]
other_edges = [(2, 3, 4), (3, 8, 5), (8, 5, 12)]

# Add edges to the graph
G.add_weighted_edges_from(mst_edges + other_edges)

# Position nodes
pos = nx.spring_layout(G, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='orange', edgecolors='black')

# Draw MST edges
nx.draw_networkx_edges(G, pos, edgelist=mst_edges, width=2, edge_color='blue')

# Draw non-MST edges
nx.draw_networkx_edges(G, pos, edgelist=other_edges, style='dashed', alpha=0.5, edge_color='gray')

# Draw edge labels
edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# Draw node labels
node_labels = {node: str(node) for node in nodes}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=15, font_weight='bold')

# Title
plt.title('Graph Theory: Minimum Spanning Tree', fontsize=16)

# Display the plot
plt.axis('off')
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/05869359db314898a9c1967c46236354.png", bbox_inches="tight")