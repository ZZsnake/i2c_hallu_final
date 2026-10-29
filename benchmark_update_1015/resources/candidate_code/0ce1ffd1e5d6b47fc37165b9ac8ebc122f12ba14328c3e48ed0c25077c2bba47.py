import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# 1. Data: Define nodes, their positions, and the edges with weights
nodes = range(12)
# Node positions are estimated to match the layout in the original image
pos = {
    0: (0.40, 0.60), 1: (0.60, 0.80), 2: (0.45, 0.20),
    3: (0.30, 0.90), 4: (0.40, 0.40), 5: (0.55, 0.25),
    6: (0.80, 0.40), 7: (0.70, 0.50), 8: (0.60, 0.30),
    9: (0.35, 0.10), 10: (0.55, 0.65), 11: (0.10, 0.55)
}

# Edges are defined as (node1, node2, weight), with weights estimated from edge colors
edges_with_weights = [
    (0, 3, 0.2), (0, 4, 0.2), (0, 10, 0.3), (0, 11, 0.3),
    (1, 3, 0.3), (1, 7, 0.5), (1, 10, 0.4),
    (2, 4, 0.5), (2, 9, 0.6),
    (4, 5, 0.6), (4, 7, 0.6), (4, 8, 0.5), (4, 11, 0.6),
    (5, 8, 0.7),
    (6, 7, 0.7), (6, 8, 0.7),
    (7, 8, 0.8), (7, 10, 0.8),
    (8, 10, 0.9)
]

# 2. Create the graph
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_weighted_edges_from(edges_with_weights)

# Extract edge weights for coloring
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

# 3. Set up the plot
# Set figure dimensions
fig, ax = plt.subplots(figsize=(10, 8))

# 4. Draw the network elements
# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='#3A4292', node_size=600, ax=ax)

# Draw edges with colors based on weights
nx.draw_networkx_edges(
    G,
    pos,
    edge_color=edge_weights,
    edge_cmap=plt.cm.viridis,
    edge_vmin=0.1,
    edge_vmax=0.9,
    width=1.5,
    ax=ax
)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_color='white', font_size=8, ax=ax)

# 5. Create and customize the color bar
# Create a ScalarMappable for the color bar
norm = mcolors.Normalize(vmin=0.1, vmax=0.9)
sm = cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])

# Add the color bar to the plot
cbar = plt.colorbar(sm, ax=ax, shrink=0.9)
cbar.set_label('Edge Strength', rotation=270, labelpad=20, fontsize=12)

# 6. Finalize the plot
# Set the title
ax.set_title("Network Connectivity Analysis", fontsize=16, pad=10)

# Remove axes
ax.set_axis_off()

# Adjust layout to prevent title from overlapping
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/6f71e9e4ccd3446192211eba29d72d76.png", bbox_inches="tight")