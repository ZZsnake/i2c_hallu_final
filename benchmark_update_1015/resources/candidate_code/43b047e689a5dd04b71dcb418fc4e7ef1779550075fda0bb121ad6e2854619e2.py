import matplotlib.pyplot as plt
import numpy as np

# Data for the parabola
x = np.linspace(-2, 2, 100)
y = 0.5 * x**2  # Equation of the parabola

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the parabola
ax.plot(x, y)

# Mark the points
ax.plot(0, 0, 'o', color='black')
ax.plot(1, 0.5, 'o', color='black')
ax.plot(2, 2, 'o', color='black')

# Add labels to the points
ax.text(0.02, 0.02, 'A (0,0)', ha='left', va='bottom')
ax.text(1.02, 0.48, 'B (1,0.5)', ha='left', va='center')
ax.text(2.02, 1.98, 'C (2,2)', ha='left', va='center')

# Set axis limits
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-0.5, 2.5])
ax.set_xticks(np.arange(-2.5, 2.51, 0.5))
ax.set_yticks(np.arange(0, 2.51, 0.5))

# Add grid
ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# Add central reference axes
ax.axvline(0, color='black', linewidth=0.6)
ax.axhline(0, color='black', linewidth=0.6)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/AsymptoteGraphicPipeline_Coordinate_Geometry_1-152.png", bbox_inches="tight")
