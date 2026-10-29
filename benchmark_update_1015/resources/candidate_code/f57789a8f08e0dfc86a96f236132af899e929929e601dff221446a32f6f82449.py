import numpy as np
import matplotlib.pyplot as plt

# Generate dummy data for the curve
x = np.linspace(0, 6, 400)
y = 0.5 * x**2 + 1  # Example quadratic function

# Plot the curve
plt.figure(figsize=(8, 8))
plt.plot(x, y, color='blue', linewidth=2)

# Mark the points O(2, 3) and P(5, 6)
plt.scatter(2, 3, color='black', label='O (2, 3)')
plt.scatter(5, 6, color='black', label='P (5, 6)')

# Connect the points with a dashed line
plt.plot([2, 5], [3, 6], color='black', linestyle='--', linewidth=2)

# Set the grid
plt.grid(True, linestyle='--', alpha=0.7)

# Set the axis limits
plt.xlim(0, 6)
plt.ylim(0, 6)

# Set the labels
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)

# Add a legend
plt.legend()

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/AsymptoteGraphicPipeline_Analytic_geometry_3-30.png", bbox_inches="tight")