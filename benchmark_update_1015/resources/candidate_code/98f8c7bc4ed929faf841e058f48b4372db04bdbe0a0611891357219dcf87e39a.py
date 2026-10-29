import matplotlib.pyplot as plt
import numpy as np

# Set figure size
plt.figure(figsize=(6, 6))

# Create a polar plot to generate concentric circles
theta = np.linspace(0, 2*np.pi, 1000)
radii = [2, 4, 6, 8, 10]

# Plot each circle
colors = ['blue', 'blue', 'blue', 'teal', 'teal']
for i, r in enumerate(radii):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plt.plot(x, y, color=colors[i], linewidth=2)

# Add labels at specific positions on each circle
labels = [2, 4, 6, 8, 10]
for i, r in enumerate(labels):
    # Place label at the top of each circle (theta = 0)
    angle = 0
    x_label = r * np.cos(angle)
    y_label = r * np.sin(angle)
    plt.text(x_label, y_label, str(r), ha='center', va='center', fontsize=12, color=colors[i])

# Set axis limits and aspect ratio
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.gca().set_aspect('equal', adjustable='box')

# Remove axes for a cleaner look
plt.axis('off')

# Add grid for reference (optional, but helps visualize the axes)
plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

# Add axes labels (x and y) at the bottom and left
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)

# Show the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/contour_4.png", bbox_inches="tight")