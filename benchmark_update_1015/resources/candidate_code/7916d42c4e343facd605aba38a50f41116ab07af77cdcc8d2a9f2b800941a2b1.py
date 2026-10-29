import matplotlib.pyplot as plt
import numpy as np

# Dummy data for nodes and their positions
node_labels = ['Coastal', 'Coastal', 'Mountain', 'Coastal', 'Desert', 'Inland', 'Inland']
node_colors = ['#FFA07A', '#00CED1', '#EE82EE', '#FFA07A', '#FFA500', '#2E8B57', '#2E8B57']
edge_labels = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Pressure', 'Visibility']

# Node positions on a circle
theta = np.linspace(0, 2 * np.pi, len(node_labels), endpoint=False)
x = np.cos(theta)
y = np.sin(theta)

# Create figure
plt.figure(figsize=(10, 10))

# Draw nodes
for i in range(len(node_labels)):
    plt.scatter([x[i]], [y[i]], c=[node_colors[i]], s=300, zorder=3)
    plt.text(x[i], y[i], str(i), ha='center', va='center', fontsize=12, color='white')

# Draw edges with labels
for i in range(len(edge_labels)):
    start_angle = theta[i]
    end_angle = theta[(i + 1) % len(node_labels)]
    mid_angle = (start_angle + end_angle) / 2
    dx = x[(i + 1) % len(node_labels)] - x[i]
    dy = y[(i + 1) % len(node_labels)] - y[i]
    
    plt.arrow(x[i], y[i], dx * 0.95, dy * 0.95, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.text((x[i] + x[(i + 1) % len(node_labels)]) / 2, 
             (y[i] + y[(i + 1) % len(node_labels)]) / 2, 
             edge_labels[i], 
             ha='center', va='center', fontsize=10)

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Coastal', markerfacecolor='#FFA07A', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Coastal', markerfacecolor='#00CED1', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Inland', markerfacecolor='#2E8B57', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Desert', markerfacecolor='#FFA500', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Mountain', markerfacecolor='#EE82EE', markersize=10)
]
plt.legend(handles=legend_elements, loc='upper right')

# Title
plt.title('Weather Station Data Communication Network', fontsize=16)

# Hide axes
plt.axis('off')
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/graph_8.png", bbox_inches="tight")