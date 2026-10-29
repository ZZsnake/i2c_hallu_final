import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch

# ----- Data -----
G = nx.DiGraph()

nodes = list(range(7))
G.add_nodes_from(nodes)

edges = [
    (0, 1, "Temperature"),
    (1, 2, "Humidity"),
    (2, 3, "Wind Speed"),
    (3, 4, "Precipitation"),
    (4, 5, "Pressure"),
    (5, 6, "Visibility"),
    (6, 0, ""),  # last edge without visible label in original
]
G.add_edges_from([(u, v) for u, v, _ in edges])

# Node colors (approximate)
node_colors = {
    0: "#ff9999",  # light red
    1: "#00b3b3",  # teal
    2: "#ffd700",  # yellow
    3: "#cc66ff",  # purple
    4: "#66ccff",  # light blue
    5: "#ff9900",  # orange
    6: "#228b22",  # green
}

# Legend labels per node
legend_labels = {
    0: "Coastal",
    1: "Coastal",
    2: "Inland",
    3: "Desert",
    4: "Coastal",
    5: "Inland",
    6: "Mountain",
}

# ----- Layout -----
pos = nx.circular_layout(G)  # positions on a unit circle

# ----- Plot -----
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("Weather Station Data Communication Network", fontsize=14)

# Draw nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=[node_colors[n] for n in G.nodes()],
    node_size=1500,
    edgecolors="black",
    linewidths=1.5,
    ax=ax,
)

# Draw node labels (0–6)
nx.draw_networkx_labels(
    G,
    pos,
    labels={n: str(n) for n in G.nodes()},
    font_size=12,
    font_color="black",
    ax=ax,
)

# Custom curved arrows along the circle
def draw_curved_arrow(ax, p1, p2, rad):
    arrow = FancyArrowPatch(
        p1,
        p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1,
        color="black",
    )
    ax.add_patch(arrow)
    return arrow

# Draw edges with slight positive curvature
for (u, v, label) in edges:
    p1, p2 = pos[u], pos[v]
    # curvature radius (same for all to mimic circular shape)
    rad = 0.25
    arrow = draw_curved_arrow(ax, p1, p2, rad)

    if label:
        # Edge midpoint (approximate) for label
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2

        # Offset label slightly outward from the circle
        # vector from center (0,0) to midpoint
        from_center_x, from_center_y = mid_x, mid_y
        scale = 0.18  # offset factor
        lx = mid_x + from_center_x * scale
        ly = mid_y + from_center_y * scale

        ax.text(
            lx,
            ly,
            label,
            fontsize=10,
            ha="center",
            va="center",
            rotation=0,
        )

# ----- Legend -----
# Create dummy patches for legend, one per node
from matplotlib.patches import Circle

legend_patches = []
legend_texts = []
for n in nodes:
    patch = Circle((0, 0), radius=0.2, facecolor=node_colors[n], edgecolor="black")
    legend_patches.append(patch)
    legend_texts.append(legend_labels[n])

class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = (xdescent + width / 2, ydescent + height / 2)
        p = Circle(
            center,
            min(width, height) / 3,
            facecolor=orig_handle.get_facecolor(),
            edgecolor="black",
        )
        p.set_transform(trans)
        return [p]

ax.legend(
    legend_patches,
    legend_texts,
    loc="upper right",
    frameon=True,
    handler_map={Circle: HandlerCircle()},
)

# ----- Aesthetic cleanup -----
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/graph_8.png", bbox_inches="tight")