import matplotlib.pyplot as plt
import numpy as np

# Set figure size
plt.figure(figsize=(10, 10))

# Define node positions (evenly spaced on a circle)
n_nodes = 7
angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
radius = 1.0
x = radius * np.cos(angles)
y = radius * np.sin(angles)

# Define node colors and labels according to legend
colors = ['#F08080', '#20B2AA', '#FFD700', '#DA70D6', '#00BFFF', '#FF8C00', '#008000']
legend_colors = ['#00BFFF', '#F08080', '#008000', '#FF8C00', '#FFD700', '#20B2AA', '#DA70D6']
labels = ['Coastal', 'Coastal', 'Inland', 'Desert', 'Coastal', 'Inland', 'Mountain']
node_numbers = [0, 1, 2, 3, 4, 5, 6]

# Plot nodes
for i in range(n_nodes):
    plt.scatter(x[i], y[i], s=550, c=colors[i], edgecolors='none', linewidth=0, zorder=5)
    plt.text(x[i], y[i], str(node_numbers[i]), fontsize=12, ha='center', va='center', color='black')

# Define edge labels (weather parameters)
edge_labels = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Pressure', 'Visibility']

# Plot curved edges with labels
for i in range(n_nodes - 1):
    start_angle = angles[i]
    end_angle = angles[i + 1]
    # Create a curved path
    mid_angle = (start_angle + end_angle) / 2
    # Use a slightly larger radius for the curve
    curve_radius = 1.3
    # Calculate control point for the curve
    control_x = curve_radius * np.cos(mid_angle)
    control_y = curve_radius * np.sin(mid_angle)
    # Draw the curve
    plt.annotate("", 
                 xy=(x[i + 1], y[i + 1]), 
                 xytext=(x[i], y[i]), 
                 arrowprops=dict(arrowstyle="->", color='black', lw=1, connectionstyle="arc3,rad=0.2", mutation_scale=12, shrinkA=17, shrinkB=17),
                 zorder=3)
    # Add label along the curve
    label_angle = mid_angle
    label_x = 0.88 * np.cos(label_angle)
    label_y = 0.88 * np.sin(label_angle)
    plt.text(label_x, label_y, edge_labels[i], fontsize=10, ha='center', va='center', rotation=np.degrees(label_angle) - 90, rotation_mode='anchor')

# Add title
plt.title("Weather Station Data Communication Network", fontsize=14, fontweight='bold', pad=20)

# Create legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=labels[i], 
                              markerfacecolor=legend_colors[i], markersize=10, markeredgecolor='none') for i in range(len(labels))]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)

# Remove axes
plt.axis('off')

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/graph_8.png", bbox_inches="tight")
