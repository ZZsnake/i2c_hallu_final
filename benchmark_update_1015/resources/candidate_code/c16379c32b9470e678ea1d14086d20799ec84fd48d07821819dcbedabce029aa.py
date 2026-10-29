import matplotlib.pyplot as plt
import numpy as np

# Create a figure with specified dimensions
plt.figure(figsize=(6, 6))

# Generate x values
x = np.linspace(-2.5, 2.5, 400)
# Generate y values for the parabola y = x^2 / 2
y = x**2 / 2

# Plot the parabola
plt.plot(x, y, 'b-')

# Mark and label the points A, B, and C
points = [(-2, 2), (0, 0), (2, 2)]
labels = ['A(-2,2)', 'B(0,0)', 'C(2,2)']

for (x_val, y_val), label in zip(points, labels):
    plt.plot(x_val, y_val, 'ko')  # black dot
    plt.text(x_val, y_val, label, fontsize=12, ha='right')

# Set the grid
plt.grid(True)

# Set the axis limits
plt.xlim(-2.5, 2.5)
plt.ylim(0, 2.5)

# Set the axis labels
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/AsymptoteGraphicPipeline_Coordinate_Geometry_1-152.png", bbox_inches="tight")