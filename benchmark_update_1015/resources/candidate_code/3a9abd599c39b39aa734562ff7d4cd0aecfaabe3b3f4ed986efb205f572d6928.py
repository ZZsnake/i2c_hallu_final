import numpy as np
import matplotlib.pyplot as plt

# Define the grid
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Define the magnetic field components (example function)
Bx = -Y
By = X
B_magnitude = np.sqrt(Bx**2 + By**2)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(6, 5))

# Plot the vector field
ax.quiver(X, Y, Bx, By, B_magnitude, cmap='viridis', angles='xy')

# Set the title
ax.set_title('Magnetic Field $\\vec{B}$ in Tesla units')

# Set the labels
ax.set_xlabel('x')
ax.set_ylabel('y')

# Set the limits
ax.set_xlim([-2.2, 2.2])
ax.set_ylim([-2.2, 2.2])
ax.set_aspect('equal', adjustable='box')

# Add grid
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/quiver_5.png", bbox_inches="tight")
