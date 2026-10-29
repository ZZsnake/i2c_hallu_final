import matplotlib.pyplot as plt
import numpy as np

# Data for the parabola
x = np.linspace(-2, 2, 100)
y = 0.5 * (x - 0.5)**2 + 0.25  # Equation of the parabola

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the parabola
ax.plot(x, y, color='blue')

# Mark the points
ax.plot(0, 0, 'o', color='black')
ax.plot(1, 0.5, 'o', color='black')
ax.plot(2, 2, 'o', color='black')

# Add labels to the points
ax.text(0, -0.1, 'A (0,0)', ha='center')
ax.text(1, 0.6, 'B (1,0.5)', ha='center')
ax.text(2, 2.1, 'C (2,2)', ha='center')

# Set axis limits
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([0, 2.5])

# Add grid
ax.grid(True)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/AsymptoteGraphicPipeline_Coordinate_Geometry_1-152.png", bbox_inches="tight")