import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = [0, 1, 2, 3, 4, 5]
G.add_nodes_from(nodes)

# Add edges
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (4, 5)]
G.add_edges_from(edges)

# Set figure dimensions
plt.figure(figsize=(9, 7))

# Draw the graph
pos = nx.spring_layout(G)  # Position nodes using spring layout
nx.draw(G, pos, with_labels=True, node_color='lightcoral', node_size=1000, edge_color='blue', arrows=True, font_size=10, font_weight='bold')

# Set title
plt.title('Trust Propagation in Directed Graph')

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/ec8c10effebe4f39a7b91c09fabf23aa.png", bbox_inches="tight")