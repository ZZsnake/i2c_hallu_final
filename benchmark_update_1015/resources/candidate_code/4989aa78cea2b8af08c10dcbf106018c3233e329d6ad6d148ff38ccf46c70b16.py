import matplotlib.pyplot as plt
import networkx as nx
# Create a directed graph
G = nx.DiGraph()
# Add nodes
nodes = [0, 1, 2, 3, 4, 5]
G.add_nodes_from(nodes)
# Add edges based on the diagram
edges = [(0, 1), (0, 2), (3, 0), (1, 3), (2, 3), (2, 4), (4, 5)]
G.add_edges_from(edges)
# Define positions for nodes to match the layout in the image
pos = {
    0: (2.25, 0.00),
    1: (1.60, -1.10),
    2: (0.75, 1.00),
    3: (0.10, -0.55),
    4: (1.15, 2.35),
    5: (-3.00, 3.05)
}
# Set figure size
plt.figure(figsize=(10, 8))
# Draw nodes
node_sizes = [1100, 500, 850, 850, 400, 350]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightcoral', edgecolors='none', linewidths=0)
# Draw edges
nx.draw_networkx_edges(G, pos, edge_color='#00008B', arrows=True, arrowsize=15, width=1.5)
# Draw labels
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='black')
# Add title
plt.title("Trust Propagation in Directed Graph", fontsize=14, fontweight='normal')
# Remove axes
plt.axis('off')
# Display the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")