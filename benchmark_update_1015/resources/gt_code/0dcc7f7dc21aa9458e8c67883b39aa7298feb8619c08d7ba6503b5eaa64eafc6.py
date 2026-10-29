import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Create a random graph with different parameters
G = nx.erdos_renyi_graph(12, 0.3, seed=42)
pos = nx.spring_layout(G, seed=42, k=0.8)

# Define node and edge properties
node_sizes = [100 + 50 * i for i in range(len(G))]
edge_weights = [1 + 0.5 * i for i in range(G.number_of_edges())]
edge_colors = np.linspace(0.1, 0.9, G.number_of_edges())
cmap = plt.cm.viridis

# Draw the network
plt.figure(figsize=(10, 8))
nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='darkblue', alpha=0.8)
edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes,
                              edge_color=edge_colors, edge_cmap=cmap,
                              width=1.5, alpha=0.7, arrows=False)

# Add labels and styling
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='black')
plt.title('Network Connectivity Analysis', fontsize=14, fontweight='bold', pad=20)
plt.colorbar(edges, label='Edge Strength', shrink=0.8)

plt.axis('off')
plt.tight_layout()
plt.show()