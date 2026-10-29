import numpy as np
import matplotlib.pyplot as plt

# Create figure with specified dimensions
fig, ax = plt.subplots(figsize=(6, 6))

# Create coordinate grid
x = np.linspace(-5, 10, 500)
y = np.linspace(-5, 10, 500)
X, Y = np.meshgrid(x, y)

# Create concentric circular contour function centered at the origin
r = np.sqrt(X**2 + Y**2)
Z = r

# Define contour levels
levels = [2, 4, 6, 8, 10, 12]

# Create contour plot with color gradient
contour = ax.contour(X, Y, Z, levels=levels, cmap='winter', linewidths=1.5)

# Add contour labels
ax.clabel(contour, inline=True, fontsize=10, fmt='%d')

# Set axis limits
ax.set_xlim(-5, 10)
ax.set_ylim(-5, 10)

# Set aspect ratio to equal for square appearance
ax.set_aspect('equal')

# Add tick marks
ax.set_xticks(range(-4, 11, 2))
ax.set_yticks(range(-4, 11, 2))

# Display the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/contour_4.png", bbox_inches="tight")
