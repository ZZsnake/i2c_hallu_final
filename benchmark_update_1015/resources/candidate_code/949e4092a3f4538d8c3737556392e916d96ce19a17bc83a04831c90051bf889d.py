import numpy as np
import matplotlib.pyplot as plt

# Create grid
x = np.linspace(-5, 10, 300)
y = np.linspace(-5, 10, 300)
X, Y = np.meshgrid(x, y)

# Radial function centered at the origin
R = np.sqrt(X**2 + Y**2)
Z = R  # simple increasing function of radius

# Contour levels similar to those seen (2,4,6,8,10,12,...)
levels = np.arange(2, 14, 2)

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))

# Filled contours with coolwarm-like palette
cf = ax.contourf(X, Y, Z, levels=levels, cmap='coolwarm', extend='both')

# Contour lines
cs = ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.8)

# Contour labels
ax.clabel(cs, inline=True, fontsize=10, fmt='%d')

# Axes limits to resemble the image
ax.set_xlim(-5, 10)
ax.set_ylim(-5, 10)

# Equal aspect for circular contours
ax.set_aspect('equal')

# Remove axis labels for a clean look (ticks remain)
ax.set_xlabel('')
ax.set_ylabel('')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/contour_5.png", bbox_inches="tight")
