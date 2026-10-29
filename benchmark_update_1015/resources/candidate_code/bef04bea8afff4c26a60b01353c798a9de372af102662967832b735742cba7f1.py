# ===================
# Part 1: Importing Libraries
# ===================
import numpy as np
import matplotlib.pyplot as plt

# ===================
# Part 2: Data Preparation
# ===================
np.random.seed(0)
# Define the vector field function for wind flow
def compute_vector_field(X, Y):
    # Simulated wind flow pattern
    U = -1 + np.sin(np.pi * X) * np.cos(np.pi * Y)
    V = 1 + np.cos(np.pi * X) * np.sin(np.pi * Y)
    return U, V

# Create a finer grid of points
x = np.linspace(-5.0, 5.0, 20)
y = np.linspace(-5.0, 5.0, 20)
X, Y = np.meshgrid(x, y)

# Compute the vector field
U, V = compute_vector_field(X, Y)
xlabel = "x (m)"
ylabel = "y (m)"
title = "Groundwater Flow Patterns Under Agricultural Land"
colorbar_title = "Flow Rate (m³/s)"

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Use a more contrasting color scheme
colors = np.sqrt(U**2 + V**2)
quiver = ax.quiver(X, Y, U, V, colors, cmap="plasma")

# Add several streamlines to the vector field plot
strm = ax.streamplot(X, Y, U, V, color='black', linewidth=0.5, density=1.2)

# Set labels and title
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_title(title)

# Show grid, set grid style
ax.grid(True, linestyle="--", alpha=0.7)

# Adjust the aspect ratio
ax.set_aspect("equal")

# Add color bar to indicate the magnitude of the vectors
cbar = plt.colorbar(quiver, ax=ax)
cbar.set_label(colorbar_title)

# ===================
# Part 4: Saving Output
# ===================
# Adjust layout
plt.tight_layout()

# Display the plot
plt.savefig('quiver_19.pdf', bbox_inches='tight')
