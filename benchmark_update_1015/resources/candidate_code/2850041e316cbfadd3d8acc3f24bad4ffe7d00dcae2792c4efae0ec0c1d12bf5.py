import numpy as np
import matplotlib.pyplot as plt

# Circle centered at O(2, 3) passing through P(5, 6)
theta = np.linspace(0, 2 * np.pi, 600)
r = np.sqrt((5 - 2)**2 + (6 - 3)**2)
x = 2 + r * np.cos(theta)
y = 3 + r * np.sin(theta)

# Plot the curve
plt.figure(figsize=(8, 8))
plt.plot(x, y, color='blue', linewidth=1.5)

# Mark the points O(2, 3) and P(5, 6)
plt.scatter(2, 3, color='black', zorder=3)
plt.scatter(5, 6, color='black', zorder=3)

# Add point labels
plt.text(1.45, 3.05, 'O (2,3)', fontsize=10)
plt.text(5.0, 6.05, 'P (5,6)', fontsize=10)

# Connect the points with a dashed line
plt.plot([2, 5], [3, 6], color='black', linestyle='--', linewidth=2)

# Set the grid
plt.grid(True, linestyle='-', color='0.7', alpha=1.0)

# Set the axis limits
plt.xlim(0, 7)
plt.ylim(0, 7)
plt.xticks(np.arange(0, 7, 1))
plt.yticks(np.arange(0, 7, 1))

# Add coordinate-axis arrows
plt.arrow(0, 0, 5.2, 0, head_width=0.18, head_length=0.18, fc='black', ec='black', length_includes_head=True, zorder=4)
plt.arrow(0, 0, 0, 6.2, head_width=0.18, head_length=0.18, fc='black', ec='black', length_includes_head=True, zorder=4)

# Set the labels
plt.xlabel('x', fontsize=10)
plt.ylabel('y', fontsize=10)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/AsymptoteGraphicPipeline_Analytic_geometry_3-30.png", bbox_inches="tight")
