import matplotlib.pyplot as plt
import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes
G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])

# Add edges with weights
G.add_edge(0, 7, weight=10)
G.add_edge(0, 1, weight=9)
G.add_edge(1, 2, weight=7)
G.add_edge(2, 3, weight=5)
G.add_edge(2, 5, weight=5)
G.add_edge(3, 4, weight=11)
G.add_edge(3, 5, weight=12)
G.add_edge(4, 5, weight=11)
G.add_edge(5, 6, weight=3)
G.add_edge(6, 8, weight=5)
G.add_edge(8, 2, weight=6)

# Define node positions (based on the image)
pos = {
    0: (1.5, 1.0),
    1: (0.8, 0.3),
    2: (0.2, 0.3),
    3: (-0.5, 0.3),
    4: (-0.8, -0.4),
    5: (-0.2, -0.4),
    6: (-1.5, -0.4),
    7: (1.0, 0.8),
    8: (-1.0, 0.8)
}

# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=800, node_color="orange", font_size=12, font_weight="bold")

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="blue")

# Remove axes
plt.axis("off")

# Set title
plt.title("Graph Theory: Minimum Spanning Tree")

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/05869359db314898a9c1967c46236354.png", bbox_inches="tight")