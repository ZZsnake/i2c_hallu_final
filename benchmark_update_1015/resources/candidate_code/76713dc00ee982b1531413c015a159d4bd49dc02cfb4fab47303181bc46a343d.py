import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = [0, 1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Add edges
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (4, 5), (3, 0)]
G.add_edges_from(edges)

# Set figure dimensions
plt.figure(figsize=(10, 8))

# Draw the graph
pos = {
    0: (1.15, -0.35),
    1: (0.88, -0.95),
    2: (0.45, 0.30),
    3: (0.12, -0.55),
    4: (0.60, 1.05),
    5: (-1.25, 1.35),
}  # Position nodes to match the reference layout
node_sizes = [1300, 600, 1000, 1000, 550, 450]
nx.draw(G, pos, with_labels=True, node_color='lightcoral', node_size=node_sizes, edge_color='darkblue', arrows=True, font_size=10, font_weight='bold', width=1.5, arrowsize=15, linewidths=1.2)

# Set title
plt.title('Trust Propagation in Directed Graph', fontsize=14)

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")
