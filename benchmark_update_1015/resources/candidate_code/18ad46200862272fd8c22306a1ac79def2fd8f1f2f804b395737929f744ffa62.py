import matplotlib.pyplot as plt
import networkx as nx
# Set up the figure and axes with specified dimensions
fig, ax = plt.subplots(figsize=(10, 8))
# Create a directed graph object
G = nx.DiGraph()
# Define the edges of the graph
edges = [(0, 1), (0, 2), (3, 0), (1, 3), (2, 3), (2, 4), (4, 5)]
G.add_edges_from(edges)
# Define the precise positions of each node to match the original image layout
pos = {
    0: (0.78, 0.32),
    1: (0.58, 0.08),
    2: (0.40, 0.50),
    3: (0.22, 0.18),
    4: (0.50, 0.80),
    5: (-0.30, 0.90)
}
# Define visual styling parameters
node_color = 'lightcoral'
node_size = [1238, 557, 956, 932, 430, 387]
edge_color = 'darkblue'
arrow_size = 15
font_color = 'black'
font_size = 10
font_weight = 'bold'
# Draw the network nodes
nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax,
    node_color=node_color,
    node_size=node_size
)
# Draw the network edges with arrows
# The node_size parameter helps shorten the edge to avoid arrow-node overlap
nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    edge_color=edge_color,
    width=1.5,
    arrows=True,
    arrowsize=arrow_size,
    node_size=node_size
)
# Draw the labels on the nodes
nx.draw_networkx_labels(
    G,
    pos,
    ax=ax,
    font_color=font_color,
    font_size=font_size,
    font_weight=font_weight
)
# Set the title of the plot
ax.set_title("Trust Propagation in Directed Graph", fontsize=16, pad=20)
# Turn off the axes to achieve a clean look
plt.axis('off')
# Adjust layout to prevent clipping of the title or graph
plt.tight_layout()
# Display the plot
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")
