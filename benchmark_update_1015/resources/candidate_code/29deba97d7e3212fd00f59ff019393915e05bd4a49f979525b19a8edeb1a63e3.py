import numpy as np
import matplotlib.pyplot as plt

# Create dummy data
x = np.linspace(-15, 15, 400)
y = np.linspace(-15, 15, 400)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 9))

# Plot the scalar field distribution
contour = ax.contourf(X, Y, Z, levels=np.linspace(-1.5, 9, 30), cmap='viridis')

# Plot the field lines
ax.contour(X, Y, Z, levels=np.linspace(1, 9, 10), colors='red', linestyles='dashed')
ax.contour(X, Y, Z, levels=np.linspace(1, 9, 10), colors='green', linestyles='dotted', alpha=0.7)

# Add the color bar
cbar = fig.colorbar(contour, ax=ax, label='Field Intensity')
cbar.set_ticks(np.arange(-1.5, 9.1, 1.5))

# Set the title and labels
ax.set_title('Scalar Field Distribution')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# Add the legend
ax.legend(handles=[plt.Line2D([0], [0], color='red', linestyle='dashed', label='Field B'),
                   plt.Line2D([0], [0], color='green', linestyle='dotted', label='Field A')],
          loc='upper right')

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/contour_15.png", bbox_inches="tight")