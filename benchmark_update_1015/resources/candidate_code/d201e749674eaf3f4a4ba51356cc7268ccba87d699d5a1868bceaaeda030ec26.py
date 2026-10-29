import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# Create custom colormap from green to pink
cmap = LinearSegmentedColormap.from_list("green_pink", ["green", "pink"])

# Generate data
x = np.linspace(-2, 2, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)

# Top plot data (log scale with two Gaussian peaks)
Z1 = np.exp(-(X**2 + Y**2)) + 0.1 * np.exp(-((X-1.2)**2 + (Y-0.7)**2)/0.1)

# Bottom plot data (linear scale with negative Gaussian)
Z2 = -6 * np.exp(-(X**2 + Y**2)/0.5)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Top plot with logarithmic color scale
c1 = ax1.pcolormesh(X, Y, Z1, cmap=cmap, norm=LogNorm(vmin=1e-1, vmax=1e0), shading='auto')
ax1.set_title('Logarithmic Scale')
cbar1 = plt.colorbar(c1, ax=ax1, orientation='vertical')
cbar1.set_ticks([1e-1, 1e0])
cbar1.set_ticklabels(['$10^{-1}$', '$10^{0}$'])

# Bottom plot with linear color scale
c2 = ax2.pcolormesh(X, Y, Z2, cmap=cmap, vmin=-6, vmax=6, shading='auto')
ax2.set_title('Linear Scale')
cbar2 = plt.colorbar(c2, ax=ax2, orientation='vertical')
cbar2.set_ticks([-6, 0, 6])

# Set axis limits and layout
ax1.set_xlim(-2, 2)
ax1.set_ylim(-1.5, 1.5)
ax2.set_xlim(-2, 2)
ax2.set_ylim(-1.5, 1.5)
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/7fdf098029ed48589f19cc01b4be5140.png", bbox_inches="tight")