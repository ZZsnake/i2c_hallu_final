import matplotlib.pyplot as plt
import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes
G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])

# Add edges with weights
G.add_edge(0, 7, weight=6)
G.add_edge(0, 1, weight=3)
G.add_edge(1, 2, weight=7)
G.add_edge(1, 7, weight=9)
G.add_edge(2, 3, weight=8)
G.add_edge(2, 5, weight=5)
G.add_edge(3, 4, weight=11)
G.add_edge(3, 5, weight=12)
G.add_edge(4, 5, weight=8)
G.add_edge(5, 6, weight=3)
G.add_edge(6, 8, weight=5)
G.add_edge(8, 2, weight=4)
G.add_edge(7, 8, weight=10)

# Define node positions (based on the image)
pos = {
    0: (5.7, 3.9),
    1: (4.5, 2.85),
    2: (2.93, 1.99),
    3: (2.11, 1.03),
    4: (1.17, 0.55),
    5: (1.28, 1.39),
    6: (0.64, 2.81),
    7: (4.10, 3.63),
    8: (2.54, 3.24)
}

# Draw the graph
plt.figure(figsize=(8, 5))

mst_edges = [(0, 7), (0, 1), (1, 2), (2, 3), (2, 5), (4, 5), (5, 6), (8, 2)]
ordinary_edges = [edge for edge in G.edges() if edge not in mst_edges and (edge[1], edge[0]) not in mst_edges]

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=ordinary_edges,
    edge_color="slategray",
    style="dashed",
    width=1.1,
    alpha=0.75
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=mst_edges,
    edge_color="navy",
    width=3.0
)
nx.draw_networkx_nodes(G, pos, node_size=600, node_color="coral")
nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="black")

# Remove axes
plt.axis("off")

# Set title
plt.title("Graph Theory: Minimum Spanning Tree")

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/05869359db314898a9c1967c46236354.png", bbox_inches="tight")
