import matplotlib.pyplot as plt
import numpy as np

# Define the function
def f(x):
    return 0.5 * x**2

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the function
x = np.linspace(-2, 2, 400)
y = f(x)
ax.plot(x, y)

# Plot and label points
points = [(0, 0), (1, 0.5), (2, 2)]
labels = ["A (0,0)", "B (1,0.5)", "C (2,2)"]
positions = ['lower right', 'upper left', 'upper left']

for point, label, pos in zip(points, labels, positions):
    ax.plot(point[0], point[1], 'ko')
    ax.text(point[0], point[1], label, ha='left', va='bottom' if pos == 'lower right' else 'top')

# Set up axes
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.set_xticks(np.arange(-2.5, 3, 0.5))
ax.set_yticks(np.arange(0, 3, 0.5))
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save the figure
plt.savefig('/home/zhoujiawei/work/image2codeBench/image2code/combine_data/graphic_py_dataset_output/images/AsymptoteGraphicPipeline_Coordinate_Geometry_1-152.png')
plt.close()