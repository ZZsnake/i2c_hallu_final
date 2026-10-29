import matplotlib.pyplot as plt
import numpy as np

# Create a figure with specified dimensions
plt.figure(figsize=(6, 6))

# Generate x values
x = np.linspace(-2, 2, 400)
# Generate y values for the parabola y = x^2 / 2
y = x**2 / 2

# Plot the parabola
plt.plot(x, y, color='#1f77b4')

# Mark and label the points A, B, and C
points = [(0, 0), (1, 0.5), (2, 2)]
labels = ['A (0,0)', 'B (1,0.5)', 'C (2,2)']

for (x_val, y_val), label in zip(points, labels):
    plt.plot(x_val, y_val, 'ko')  # black dot
    plt.text(x_val, y_val, label, fontsize=10, ha='left', va='bottom' if x_val == 0 else 'top')

# Set the grid
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

# Draw coordinate axes through the origin
plt.axvline(0, color='black', linewidth=0.6)
plt.axhline(0, color='black', linewidth=0.6)

# Set the axis limits
plt.xlim(-2.5, 2.5)
plt.ylim(-0.5, 2.5)
plt.xticks(np.arange(-2.5, 2.6, 0.5))
plt.yticks(np.arange(0, 2.6, 0.5))

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/AsymptoteGraphicPipeline_Coordinate_Geometry_1-152.png", bbox_inches="tight")
