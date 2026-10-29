import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = [0, 1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Add edges based on the image
edges = [(0, 2), (0, 3), (1, 0), (1, 3), (2, 3), (2, 4), (3, 0), (4, 5)]
G.add_edges_from(edges)

# Set figure size
plt.figure(figsize=(9, 7))

# Define node positions manually to match the image layout
pos = {
    0: (0, 0),
    1: (-0.5, -1),
    2: (0.5, 1),
    3: (0, -1),
    4: (1, 2),
    5: (2, 3)
}

# Draw the graph
nx.draw_networkx_nodes(G, pos, node_color='salmon', node_size=600, edgecolors='black', linewidths=1.5)
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='navy', width=2.5, arrows=True, arrowstyle='->', arrowsize=10)
nx.draw_networkx_labels(G, pos, font_color='white', font_size=12, font_weight='bold')

# Add title
plt.title("Trust Propagation in Directed Graph", fontsize=16, pad=20)

# Remove axes
plt.axis('off')

# Adjust layout to fit
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")