import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = [0, 1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Add edges based on the image
edges = [(0, 2), (0, 1), (1, 3), (2, 3), (2, 4), (3, 0), (4, 5)]
G.add_edges_from(edges)

# Set figure size
plt.figure(figsize=(10, 7))

# Define node positions manually to match the image layout
pos = {
    0: (2.5, -0.85),
    1: (1.75, -2.15),
    2: (0.9, 0.5),
    3: (0.2, -1.25),
    4: (1.3, 2.2),
    5: (-3.0, 2.9)
}

# Draw the graph
node_sizes = [1300, 650, 1050, 950, 500, 450]
nx.draw_networkx_nodes(G, pos, node_color='salmon', node_size=node_sizes, edgecolors='none')
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='navy', width=3/2, arrows=True, arrowstyle='->', arrowsize=10, node_size=node_sizes)
nx.draw_networkx_labels(G, pos, font_color='black', font_size=12, font_weight='bold')

# Add title
plt.title("Trust Propagation in Directed Graph", fontsize=16, pad=20)

# Remove axes
plt.axis('off')

# Adjust layout to fit
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")
