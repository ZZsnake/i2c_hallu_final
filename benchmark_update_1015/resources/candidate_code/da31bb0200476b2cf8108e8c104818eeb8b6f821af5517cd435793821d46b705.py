import numpy as np
import matplotlib.pyplot as plt

# Set the figure size
plt.figure(figsize=(6, 5))

# Create a grid of x and y values
x = np.linspace(-2.0, 2.0, 20)
y = np.linspace(-2.0, 2.0, 20)
X, Y = np.meshgrid(x, y)

# Define the vector field components
U = -Y
V = X

# Magnitude of the vector field
N = np.sqrt(U**2 + V**2)

# Plot the vector field
plt.quiver(X, Y, U, V, N, cmap=plt.cm.viridis)

# Set the title and labels
plt.title('Magnetic Field $\\vec{B}$ in Tesla units')
plt.xlabel('x')
plt.ylabel('y')

# Set the axis limits
plt.xlim(-2.2, 2.2)
plt.ylim(-2.2, 2.2)

# Add gridlines
plt.grid(True, linestyle='--', color='gray', alpha=0.5)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/quiver_5.png", bbox_inches="tight")
