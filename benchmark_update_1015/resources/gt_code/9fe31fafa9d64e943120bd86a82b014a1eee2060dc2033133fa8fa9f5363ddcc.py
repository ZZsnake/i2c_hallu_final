import networkx as nx
import matplotlib.pyplot as plt

# Create a weighted graph representing mathematical relationships
G = nx.Graph()
G.add_edges_from([
    (0, 1, {"weight": 3}),
    (0, 7, {"weight": 6}),
    (1, 7, {"weight": 9}),
    (1, 2, {"weight": 7}),
    (2, 8, {"weight": 4}),
    (2, 5, {"weight": 5}),
    (2, 3, {"weight": 8}),
    (3, 4, {"weight": 11}),
    (3, 5, {"weight": 12}),
    (4, 5, {"weight": 8}),
    (5, 6, {"weight": 3}),
    (6, 8, {"weight": 5}),
    (7, 8, {"weight": 10})
])

# Compute minimum spanning tree
T = nx.minimum_spanning_tree(G)

# Visualize the graph structure
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color="coral", node_size=600, alpha=0.9)
nx.draw_networkx_edges(G, pos, edge_color="slategray", style="dashed", alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=14, font_family="DejaVu Sans", font_weight="bold")
nx.draw_networkx_edge_labels(
    G, pos, edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)},
    font_size=10, font_family="monospace"
)
nx.draw_networkx_edges(T, pos, edge_color="navy", width=3, alpha=0.9)

plt.title("Graph Theory: Minimum Spanning Tree", fontsize=16, pad=20)
plt.axis("off")
plt.tight_layout()
plt.show()