import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = [0, 1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Add edges based on the diagram
edges = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 4), (4, 5)]
G.add_edges_from(edges)

# Define positions for nodes to match the layout in the image
pos = {
    0: (2, 2),
    1: (2, 1),
    2: (1, 2),
    3: (1, 1),
    4: (1.5, 3),
    5: (-1, 3)
}

# Set figure size
plt.figure(figsize=(9, 7))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=800, node_color='#FF9999', edgecolors='#CC0000', linewidths=2)

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color='#00008B', arrows=True, arrowstyle='->', arrowsize=20, width=2)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='black')

# Add title
plt.title("Trust Propagation in Directed Graph", fontsize=16, fontweight='bold')

# Remove axes
plt.axis('off')

# Display the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")