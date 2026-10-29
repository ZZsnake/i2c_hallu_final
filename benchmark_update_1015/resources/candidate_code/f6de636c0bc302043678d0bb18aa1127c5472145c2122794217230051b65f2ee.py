import matplotlib.pyplot as plt
import numpy as np

# Data for node positions
angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
radii = [1.2] * 7  # Adjust radius as needed

# Coordinates for the nodes
x = radii * np.cos(angles)
y = radii * np.sin(angles)

# Node colors based on station type
colors = ['blue', 'blue', 'yellow', 'green', 'purple', 'blue', 'green']

# Labels for the connections
labels = ["Temperature", "Humidity", "Wind Speed", "Precipitation", "Pressure", "Visibility"]

# Create the figure and axes
plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.set_aspect('equal')  # Ensure a circular shape

# Plot the connections
for i in range(6):
    ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color='gray', linewidth=1)
    
    # Add connection labels
    angle = (angles[i] + angles[i+1]) / 2
    x_label = 1.3 * np.cos(angle)
    y_label = 1.3 * np.sin(angle)

    ax.text(x_label, y_label, labels[i], ha='center', va='center', rotation=np.degrees(angle))

# Connect last node to the first node
ax.plot([x[6], x[0]], [y[6], y[0]], color='gray', linewidth=1)
ax.text(x[0]+0.1,y[0]-0.1,"Visibility", ha='center', va='center')

# Plot the nodes
for i in range(7):
    ax.plot(x[i], y[i], 'o', markersize=10, color=colors[i])
    ax.text(x[i], y[i], str(i), ha='center', va='center', color='white')

# Remove axes and ticks
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Set plot limits to accommodate the circle
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/graph_8.png", bbox_inches="tight")