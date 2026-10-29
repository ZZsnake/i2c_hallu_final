import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# 1. Setup Figure
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.axis('off')

# 2. Data and Configuration
num_nodes = 7
# Radius for the center of the nodes
radius = 1.2
# Visual radius of the node circles themselves
node_size_radius = 0.15

# Node positions (angles in degrees, 0 is at 3 o'clock, positive is CCW)
# These angles are manually tuned to match the visual layout of the original image.
angles_deg = np.array([320, 40, 110, 170, 210, 250, 290])
angles_rad = np.deg2rad(angles_deg)
node_x = radius * np.cos(angles_rad)
node_y = radius * np.sin(angles_rad)

# Node properties (colors, labels, text color)
node_labels = [str(i) for i in range(num_nodes)]
node_colors = {
    0: '#f8b4b4', 1: '#48bca8', 2: '#fde633', 3: '#d682d1',
    4: '#4ab6e0', 5: '#f8a23c', 6: '#449d44'
}
node_text_colors = {
    0: 'white', 1: 'white', 2: 'black', 3: 'white',
    4: 'white', 5: 'black', 6: 'white'
}

# Edge properties (labels for each connection)
edge_labels = {
    (0, 1): 'Temperature', (1, 2): 'Humidity', (2, 3): 'Wind Speed',
    (3, 4): 'Precipitation', (4, 5): 'Pressure', (5, 6): 'Visibility',
    (6, 0): ''
}

# 3. Draw Elements
# Draw edges (arcs) between nodes
for i in range(num_nodes):
    start_node = i
    end_node = (i + 1) % num_nodes
    
    start_pos = (node_x[start_node], node_y[start_node])
    end_pos = (node_x[end_node], node_y[end_node])
    
    # Use FancyArrowPatch to create curved connections
    arrow = FancyArrowPatch(
        posA=start_pos, posB=end_pos,
        arrowstyle='-', color='black',
        connectionstyle="arc3,rad=0.3",
        linewidth=1.5, zorder=1
    )
    ax.add_patch(arrow)

# Draw nodes and their numeric labels
for i in range(num_nodes):
    circle = plt.Circle((node_x[i], node_y[i]), node_size_radius, color=node_colors[i], zorder=2)
    ax.add_artist(circle)
    ax.text(node_x[i], node_y[i], node_labels[i],
            ha='center', va='center', color=node_text_colors[i],
            fontweight='bold', fontsize=12, zorder=3)

# Draw edge labels along the arcs
text_radius = radius + 0.35  # Radius for placing the text
for i in range(num_nodes):
    start_node = i
    end_node = (i + 1) % num_nodes
    label = edge_labels.get((start_node, end_node))
    
    if label:
        start_angle = angles_rad[start_node]
        end_angle = angles_rad[end_node]
        
        # Handle the angle wrap-around for the arc from the last to the first node
        if end_angle < start_angle:
            end_angle += 2 * np.pi
            
        # Calculate the midpoint angle for text placement
        mid_angle = (start_angle + end_angle) / 2
        
        # Position for the text
        text_x = text_radius * np.cos(mid_angle)
        text_y = text_radius * np.sin(mid_angle)
        
        # Rotation for the text
        rotation_deg = np.rad2deg(mid_angle)
        
        # Adjust rotation for text on the left half to keep it upright
        if 90 < rotation_deg < 270:
            rotation_deg += 180
            
        ax.text(text_x, text_y, label,
                ha='center', va='center', rotation=rotation_deg,
                fontsize=12, zorder=3, rotation_mode='anchor')

# 4. Add Title and Legend
ax.set_title("Weather Station Data Communication Network", fontsize=16, pad=20)

# Legend data based on the image
legend_labels = ['Coastal', 'Coastal', 'Inland', 'Desert', 'Coastal', 'Inland', 'Mountain']
# The legend colors are mapped to match the order and labels in the original image
legend_colors = ['#4ab6e0', '#f8b4b4', '#449d44', '#f8a23c', '#fde633', '#4ab6e0', '#d682d1']

# Create custom legend handles
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=label,
           markerfacecolor=color, markersize=10)
    for color, label in zip(legend_colors, legend_labels)
]

# The first legend entry for 'Coastal' has a different color than node 4 ('Inland')
# This is a discrepancy in the original image. We use the color of node 4 for the 'Inland' entry.
# The first 'Coastal' entry color is not used by any node, so we use the color of node 1.
legend_elements[0] = Line2D([0], [0], marker='o', color='w', label='Coastal', markerfacecolor='#48bca8', markersize=10)


ax.legend(handles=legend_elements, loc='upper right', frameon=False, bbox_to_anchor=(0.98, 0.98))

# 5. Final Adjustments
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/graph_8.png", bbox_inches="tight")