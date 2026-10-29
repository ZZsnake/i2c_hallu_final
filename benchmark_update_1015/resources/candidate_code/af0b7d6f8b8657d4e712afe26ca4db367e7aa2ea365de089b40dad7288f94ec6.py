import matplotlib.pyplot as plt
import numpy as np

# Set the figure dimensions
plt.figure(figsize=(8, 8))

# Generate dummy data for the curve (parabolic function)
x = np.linspace(-10, 10, 400)
y = -0.1 * x**2 + 6

# Plot the curve
plt.plot(x, y, 'b-', linewidth=2)

# Mark the points O(2,3) and P(5,6)
plt.scatter([2, 5], [3, 6], color='black', zorder=5)

# Annotate the points
plt.annotate('O(2,3)', (2, 3), textcoords="offset points", xytext=(0,10), ha='center')
plt.annotate('P(5,6)', (5, 6), textcoords="offset points", xytext=(0,10), ha='center')

# Draw a dashed line connecting the points
plt.plot([2, 5], [3, 6], 'k--', linewidth=1.5)

# Set axis limits
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# Add grid
plt.grid(True)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/AsymptoteGraphicPipeline_Analytic_geometry_3-30.png", bbox_inches="tight")