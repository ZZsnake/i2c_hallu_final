import networkx as nx
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
relationships = {
    "A1": [],
    "A2": [(1, 2)],
    "B1": [(1, 2), (2, 1)],
    "B2": [(3, 1), (3, 2)],
    "C1": [(1, 3), (2, 3)],
    "C2": [(1, 3), (3, 2)],
    "D1": [(1, 2), (2, 1), (3, 1)],
    "D2": [(1, 2), (2, 1), (1, 3)],
    "E1": [(1, 2), (3, 2), (1, 3)],
    "E2": [(1, 3), (3, 2), (2, 1)],
    "F1": [(1, 2), (2, 1), (3, 1), (1, 3)],
    "F2": [(1, 2), (2, 1), (3, 1), (3, 2)],
    "G1": [(1, 2), (2, 1), (1, 3), (2, 3)],
    "G2": [(1, 2), (2, 1), (1, 3), (3, 2)],
    "H1": [(1, 2), (2, 1), (1, 3), (3, 2), (2, 3)],
    "H2": [(1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)],
}

for (label, edges), ax in zip(relationships.items(), axes.flatten()):
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from(edges)
    nx.draw_networkx(
        G,
        ax=ax,
        with_labels=False,
        node_color=["crimson"],
        node_size=250,
        arrowsize=18,
        width=1.8,
        edge_color="navy",
        pos=nx.planar_layout(G),
    )
    ax.set_xlim(val * 1.3 for val in ax.get_xlim())
    ax.set_ylim(val * 1.3 for val in ax.get_ylim())
    ax.text(
        0,
        0,
        label,
        fontsize=14,
        fontweight="bold",
        horizontalalignment="center",
        bbox={"boxstyle": "circle,pad=0.4", "fc": "none", "ec": "darkgreen", "linewidth": 2},
    )
fig.tight_layout()
plt.show()