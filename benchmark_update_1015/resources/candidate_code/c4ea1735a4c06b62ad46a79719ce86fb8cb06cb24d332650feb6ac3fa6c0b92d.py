# ===================
# Part 1: Importing Libraries
# ===================
import numpy as np
import matplotlib.pyplot as plt

# ===================
# Part 2: Data Preparation
# ===================
np.random.seed(0)
# Define the vector field function for wind patterns
def compute_wind_vector_field(X, Y):
    U = np.sin(np.pi * X) * np.cos(np.pi * Y)
    V = -np.cos(np.pi * X) * np.sin(np.pi * Y)
    return U, V

# Create a grid of points
x = np.linspace(-50.0, 50.0, 20)
y = np.linspace(-50.0, 50.0, 20)
X, Y = np.meshgrid(x, y)

# Compute the vector field for water currents
U, V = compute_wind_vector_field(X, Y)
speed = np.sqrt(U**2 + V**2)

xlabel = "X Coordinate (m)"
ylabel = "Y Coordinate (m)"
title = "Simulated Water Currents in a Lake"
cbar_label = 'Water Speed'

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
Q = ax.quiver(X, Y, U, V, speed, cmap='YlGnBu', alpha=0.8)

# Set labels and title
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_title(title)

# Show grid
ax.grid(True, linestyle="--", alpha=0.6)

# Add a color bar
cbar = fig.colorbar(Q, ax=ax)
cbar.set_label(cbar_label)

# ===================
# Part 4: Saving Output
# ===================
# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('quiver_15.pdf', bbox_inches='tight')
