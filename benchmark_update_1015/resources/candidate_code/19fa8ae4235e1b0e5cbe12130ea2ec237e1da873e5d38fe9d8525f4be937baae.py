import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes (0 to 5)
nodes = [0, 1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Add edges based on the directed graph in the image
edges = [(0, 2), (3, 0), (0, 1), (1, 3), (2, 3), (2, 4), (4, 5)]
G.add_edges_from(edges)

# Create the plot
plt.figure(figsize=(10, 8))

# Manually position nodes to match the reference layout
pos = {
    0: (1.10, -0.55),
    1: (0.85, -1.15),
    2: (0.48, 0.05),
    3: (0.22, -0.78),
    4: (0.62, 0.84),
    5: (-1.00, 1.10),
}

# Draw nodes with varied sizes matching the reference emphasis
node_sizes = [1400, 650, 1100, 1000, 600, 550]
nx.draw_networkx_nodes(G, pos, node_color='#F08080', node_size=node_sizes, alpha=1.0)

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color='darkblue', arrows=True, arrowsize=20, width=2)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=11, font_weight='normal', font_color='black')

# Set title
plt.title('Trust Propagation in Directed Graph', fontsize=14, fontweight='normal', pad=20)

# Remove axis
plt.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")
