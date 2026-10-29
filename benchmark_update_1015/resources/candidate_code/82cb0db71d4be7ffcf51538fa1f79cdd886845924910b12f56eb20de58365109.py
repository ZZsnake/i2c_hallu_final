import matplotlib.pyplot as plt
import numpy as np

# Set the figure dimensions
plt.figure(figsize=(8, 8))

# Generate data for the circle centered at O(2,3) passing through P(5,6)
center_x, center_y = 2, 3
radius = np.sqrt((5 - center_x) ** 2 + (6 - center_y) ** 2)
theta = np.linspace(0, 2 * np.pi, 600)
x = center_x + radius * np.cos(theta)
y = center_y + radius * np.sin(theta)

# Plot the circle
plt.plot(x, y, 'b-', linewidth=1.3)

# Mark the points O(2,3) and P(5,6)
plt.scatter([2, 5], [3, 6], color='black', zorder=5)

# Annotate the points
plt.annotate('O (2,3)', (2, 3), textcoords="offset points", xytext=(-2, 2), ha='right')
plt.annotate('P (5,6)', (5, 6), textcoords="offset points", xytext=(0, 2), ha='left')

# Draw a dashed line connecting the points
plt.plot([2, 5], [3, 6], 'k--', linewidth=1.5)

# Set axis limits and integer tick scale
plt.xlim(0, 7)
plt.ylim(0, 7)
plt.xticks(np.arange(0, 7, 1))
plt.yticks(np.arange(0, 7, 1))

# Add arrowed coordinate axes starting at the origin
plt.arrow(0, 0, 5.2, 0, length_includes_head=True, head_width=0.18, head_length=0.18,
          fc='black', ec='black', linewidth=1.2, zorder=4)
plt.arrow(0, 0, 0, 6.2, length_includes_head=True, head_width=0.18, head_length=0.18,
          fc='black', ec='black', linewidth=1.2, zorder=4)

# Add axis labels
plt.xlabel('x')
plt.ylabel('y')

# Add grid
plt.grid(True)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/AsymptoteGraphicPipeline_Analytic_geometry_3-30.png", bbox_inches="tight")