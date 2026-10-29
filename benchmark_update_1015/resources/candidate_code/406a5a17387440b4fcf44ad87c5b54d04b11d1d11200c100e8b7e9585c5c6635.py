import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm

# Create dummy data
x = np.linspace(-2.8, 2.8, 300)
y = np.linspace(-1.8, 1.8, 200)
X, Y = np.meshgrid(x, y)

# Signed two-lobed field used for both panels
field = (
    7.0 * np.exp(-(((X + 0.35) / 0.42) ** 2 + ((Y + 0.42) / 0.35) ** 2))
    + 2.4 * np.exp(-(((X + 0.75) / 1.25) ** 2 + ((Y + 0.35) / 1.00) ** 2))
    + 1.2 * np.exp(-(((X - 0.55) / 0.55) ** 2 + ((Y + 0.65) / 0.55) ** 2))
    - 4.5 * np.exp(-(((X - 0.50) / 0.26) ** 2 + ((Y - 0.50) / 0.22) ** 2))
)

# Logarithmic scale data
log_data = field

# Linear scale data
linear_data = field

# Diverging pink-white-green colormap
cmap = LinearSegmentedColormap.from_list(
    'pink_white_green',
    ['#c0007a', '#e65aac', '#f7f7f7', '#b7e36a', '#1b7f00'],
    N=256
)

# Set up the figure dimensions
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

extent = [-2.8, 2.8, -1.8, 1.8]

# Plot logarithmic scale heatmap
im1 = axs[0].imshow(
    log_data,
    extent=extent,
    origin='lower',
    cmap=cmap,
    norm=SymLogNorm(linthresh=0.1, linscale=1.0, vmin=-7, vmax=7),
    aspect='auto',
    interpolation='bilinear'
)
axs[0].text(0.09, 0.88, 'Logarithmic Scale', transform=axs[0].transAxes,
            fontweight='bold', ha='left', va='center')
fig.colorbar(im1, ax=axs[0], orientation='vertical', extend='both', ticks=[-1, -0.1, 0, 0.1, 1])

# Plot linear scale heatmap
im2 = axs[1].imshow(
    linear_data,
    extent=extent,
    origin='lower',
    cmap=cmap,
    norm=Normalize(vmin=-7, vmax=7),
    aspect='auto',
    interpolation='bilinear'
)
axs[1].text(0.09, 0.88, 'Linear Scale', transform=axs[1].transAxes,
            fontweight='bold', ha='left', va='center')
fig.colorbar(im2, ax=axs[1], orientation='vertical', extend='both', ticks=[-6, -4, -2, 0, 2, 4, 6])

for ax in axs:
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-1.8, 1.8)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/7fdf098029ed48589f19cc01b4be5140.png", bbox_inches="tight")
