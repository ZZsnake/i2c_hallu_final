import numpy as np
import matplotlib.pyplot as plt

# Generate dummy data
x = np.linspace(-5, 10, 100)
y = np.linspace(-5, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)

# Create the contour plot
plt.figure(figsize=(7, 7))

# Color gradient
plt.contourf(X, Y, Z, levels=[0, 2, 4, 6, 8, 10, 12, 14, 16], cmap='coolwarm')

contour = plt.contour(X, Y, Z, levels=[2, 4, 6, 8, 10, 12], colors='k', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=10)

# Set axis limits
plt.xlim(-5, 10)
plt.ylim(-5, 10)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/contour_5.png", bbox_inches="tight")
