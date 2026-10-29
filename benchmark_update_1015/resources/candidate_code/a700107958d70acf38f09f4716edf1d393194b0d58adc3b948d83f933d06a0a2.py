import numpy as np
import matplotlib.pyplot as plt

# Generate dummy data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)

# Create the contour plot
plt.figure(figsize=(6, 6))
contour = plt.contour(X, Y, Z, levels=[2, 4, 6, 8, 10, 12], colors='k')
plt.clabel(contour, inline=True, fontsize=10)

# Color gradient
plt.contourf(X, Y, Z, levels=[2, 4, 6, 8, 10, 12], cmap='coolwarm')

# Set axis limits and labels
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel('X')
plt.ylabel('Y')

# Show the plot
plt.colorbar()
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/contour_5.png", bbox_inches="tight")