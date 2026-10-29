import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 0), (4, 5)])

# Define initial trusted vertices
trusted_vertices = {0, 2}

# Create personalization vector with equal weights for trusted vertices
personalization = {n: 1.0 / len(trusted_vertices) if n in trusted_vertices else 0 for n in G.nodes}

# Compute TrustRank scores using PageRank algorithm
trust_scores = nx.pagerank(G, alpha=0.9, personalization=personalization)

# Generate layout with consistent positioning
layout = nx.spring_layout(G, seed=42)
vertex_sizes = [trust_scores[n] * 4500 for n in G]

plt.figure(figsize=(9, 7))
nx.draw(
    G,
    layout,
    with_labels=True,
    node_size=vertex_sizes,
    node_color="lightcoral",
    edge_color="darkblue",
    font_size=11,
    font_weight="bold",
    width=1.5,
    arrowsize=15,
    linewidths=1.2
)
plt.title("Trust Propagation in Directed Graph", fontsize=14)
plt.show()