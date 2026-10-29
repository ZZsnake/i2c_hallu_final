import numpy as np
import matplotlib.pyplot as plt

# Create grid data
x = np.linspace(-5, 10, 300)
y = np.linspace(-5, 10, 300)
X, Y = np.meshgrid(x, y)

# Radially symmetric function
Z = np.sqrt(X**2 + Y**2)

# Contour levels similar to the figure
levels = np.arange(2, 14, 2)  # 2,4,6,8,10,12

# Create figure
plt.figure(figsize=(6, 6))

# Draw contours with a single colormap (similar to default look)
cs = plt.contour(X, Y, Z, levels=levels, cmap='viridis')

# Label contours inline
plt.clabel(cs, inline=True, fontsize=10, fmt='%d')

# Axis limits to match visual range
plt.xlim(-5, 10)
plt.ylim(-5, 10)

# Keep default ticks and style, no title/legend/grid
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/contour_4.png", bbox_inches="tight")