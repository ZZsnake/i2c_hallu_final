import matplotlib.pyplot as plt
import numpy as np

# Set figure dimensions
plt.figure(figsize=(6, 5))

# Create grid coordinates
x = np.linspace(-2.0, 2.0, 20)
y = np.linspace(-2.0, 2.0, 20)
X, Y = np.meshgrid(x, y)

# Define magnetic field components
B_x = -Y
B_y = X

# Compute magnitude for color encoding
magnitude = np.sqrt(B_x**2 + B_y**2)

# Create the vector field plot
plt.quiver(X, Y, B_x, B_y, magnitude, cmap='viridis', angles='xy', scale_units='xy', scale=25)

# Set axis labels
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Magnetic Field $\vec{B}$ in Tesla units")
plt.gca().set_aspect('equal')

# Add gridlines
plt.grid(True, linestyle='--')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/quiver_5.png", bbox_inches="tight")
