# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import networkx as nx

# ===================
# Part 2: Data Preparation
# ===================
# Create a directed graph
G = nx.DiGraph()

# Add nodes with their respective colors representing station types
nodes = {
    0: {"color": "lightcoral", "type": "Coastal"},
    1: {"color": "lightseagreen", "type": "Inland"},
    2: {"color": "gold", "type": "Coastal"},
    3: {"color": "orchid", "type": "Mountain"},
    4: {"color": "deepskyblue", "type": "Coastal"},
    5: {"color": "darkorange", "type": "Desert"},
    6: {"color": "forestgreen", "type": "Inland"},
}
for node, attributes in nodes.items():
    G.add_node(node, color=attributes["color"], type=attributes["type"])

# Add edges with labels representing the type of data communication
edges = [
    (0, 1, "Temperature"), 
    (1, 2, "Humidity"), 
    (2, 3, "Wind Speed"), 
    (3, 4, "Precipitation"), 
    (4, 5, "Pressure"), 
    (5, 6, "Visibility")
]
for u, v, label in edges:
    G.add_edge(u, v, label=label)

# Define node positions in a circular layout
pos = nx.circular_layout(G)

title = "Weather Station Data Communication Network"

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
fig, ax = plt.subplots(figsize=(10, 10))

# Draw nodes with color attribute
node_colors = [G.nodes[node]["color"] for node in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)

# Draw edges with labels
nx.draw_networkx_edges(G, pos, arrows=True, connectionstyle='arc3,rad=0.2')
edge_labels = {(u, v): G[u][v]["label"] for u, v in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Add node labels
nx.draw_networkx_labels(G, pos, {node: node for node in G.nodes})

# Add a title
plt.title(title)

# Create a legend for node types
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=type, markersize=10)
    for color, type in set((attributes["color"], attributes["type"]) for attributes in nodes.values())
]
ax.legend(handles=legend_handles, loc='upper right')

# Remove axis
plt.axis("off")

# ===================
# Part 4: Saving Output
# ===================
# Displaying the plot with tight layout to minimize white space
plt.tight_layout()
plt.savefig("graph_8.pdf", bbox_inches="tight")