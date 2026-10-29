import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = [0, 1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Add edges
edges = [(0, 1), (0, 2), (2, 3), (2, 4), (3, 0), (4, 5), (1, 3)]
G.add_edges_from(edges)

# Set node colors and sizes
node_color = 'lightcoral'  # Light red
node_size = [1200, 500, 900, 900, 500, 400]

# Set edge color
edge_color = 'darkblue'

# Draw the graph
plt.figure(figsize=(10, 8))
pos = {
    0: (0.95, -0.45),
    1: (0.70, -0.95),
    2: (0.36, 0.05),
    3: (0.10, -0.58),
    4: (0.50, 0.72),
    5: (-1.00, 0.95),
}
nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=node_size, edge_color=edge_color, arrows=True, font_size=12, font_weight='bold', width=1.5, arrowsize=15, linewidths=1.2)

# Set title
plt.title("Trust Propagation in Directed Graph", fontsize=14)

# Display the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")