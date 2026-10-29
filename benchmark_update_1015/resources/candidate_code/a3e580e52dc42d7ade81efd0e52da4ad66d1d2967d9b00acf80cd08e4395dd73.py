import matplotlib.pyplot as plt
import numpy as np

# Create dummy data for the vector field
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 4, 20)
X, Y = np.meshgrid(x, y)

# Define the flow field components (example)
u = 1.5 + 0.2 * Y  # x-component of the flow
v = -1.0 - 0.1 * X  # y-component of the flow

# Normalize the magnitude for color mapping
magnitude = np.sqrt(u**2 + v**2)
normalized_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the vector field with color mapping
quiver = ax.quiver(X, Y, u, v, normalized_magnitude, cmap='viridis', scale=50)

# Add a colorbar
cbar = fig.colorbar(quiver, ax=ax, label='Flow Rate (m²/s)')

# Set axis labels
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

# Set axis limits
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 4])

# Add grid lines
ax.grid(True, linestyle=':')

# Set title
ax.set_title('Groundwater Flow Patterns Under Agricultural Land')

# Show the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/quiver_19.png", bbox_inches="tight")