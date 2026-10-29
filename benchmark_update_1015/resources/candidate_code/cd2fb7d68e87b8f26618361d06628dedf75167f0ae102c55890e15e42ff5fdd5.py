import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_nodes_from([0, 1, 2, 3, 4, 5])

# Add edges
G.add_edges_from([(0, 1), (0, 2), (0, 3), (2, 3), (3, 4), (5, 4)])

# Define node positions
pos = {
    0: (0, 0),
    1: (1, -1),
    2: (1, 1),
    3: (-1, 0),
    4: (2, 1.5),
    5: (-2, 2)
}

# Set figure dimensions
plt.figure(figsize=(9, 7))

# Draw the graph
nx.draw(G, pos, with_labels=True, node_size=800, node_color="red", arrowsize=20, edge_color="blue")

# Remove axes
plt.axis("off")

# Add title
plt.title("Trust Propagation in Directed Graph")

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")