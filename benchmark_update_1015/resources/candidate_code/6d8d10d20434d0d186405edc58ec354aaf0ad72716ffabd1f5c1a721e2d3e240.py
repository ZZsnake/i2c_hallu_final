import numpy as np
import matplotlib.pyplot as plt

# Data for the parabola y = 0.5 * x^2
x = np.linspace(-2, 2, 400)
y = 0.5 * x**2

# Key points
points_x = [0, 1, 2]
points_y = [0, 0.5, 2]
labels = ["A (0,0)", "B (1,0.5)", "C (2,2)"]
text_offsets = [(0.05, 0.03), (0.05, -0.05), (0.05, -0.05)]

# Create figure
plt.figure(figsize=(6, 6))

# Plot the parabola
plt.plot(x, y, color='C0')

# Plot the key points
plt.scatter(points_x, points_y, color='black', zorder=3)

# Annotate the points
for (px, py, label, (dx, dy)) in zip(points_x, points_y, labels, text_offsets):
    plt.text(px + dx, py + dy, label, color='black')

# Axes lines at x=0 and y=0
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Set limits
plt.xlim(-2.5, 2.5)
plt.ylim(-0.5, 2.5)

# Grid, ticks, and layout
plt.grid(True, color='0.7', linestyle='--', linewidth=0.5)
plt.xticks(np.arange(-2.5, 2.51, 0.5))
plt.yticks(np.arange(-0, 3, 0.5))
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/AsymptoteGraphicPipeline_Coordinate_Geometry_1-152.png", bbox_inches="tight")
