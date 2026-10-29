import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_nodes_from([0, 1, 2, 3, 4, 5])

# Add edges
G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 0), (4, 5)])

# Set figure dimensions
plt.figure(figsize=(10, 8))

# Draw the graph
pos = {
    0: (1.05, -0.62),
    1: (0.72, -1.20),
    2: (0.32, -0.05),
    3: (0.04, -0.82),
    4: (0.48, 0.70),
    5: (-1.28, 1.00),
}
node_sizes = [1300, 650, 1000, 1000, 500, 500]
nx.draw(G, pos, with_labels=True, node_color='lightcoral', edge_color='darkblue', node_size=node_sizes, font_size=12, width=1.5, arrowsize=15)

# Set title
plt.title('Trust Propagation in Directed Graph', fontsize=16)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")
