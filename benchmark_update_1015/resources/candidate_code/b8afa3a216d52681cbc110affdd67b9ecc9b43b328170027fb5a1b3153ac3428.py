import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

# Create dummy data
alpha = np.linspace(-2.5, 2.5, 400)
beta = np.linspace(-1.8, 1.8, 400)
A, B = np.meshgrid(alpha, beta)
Z = np.exp(-A**2 - B**2) + 40 * np.exp(-64 * A**2 - 64 * B**2)
Z = np.ma.masked_where((A < -2.22) & (B < -1.58), Z)

# Create the plot
fig, ax = plt.subplots(figsize=(9, 6.5))
levels = np.logspace(-5, 2, 8)
contour = ax.contourf(A, B, Z, levels=levels, norm=LogNorm(vmin=1e-5, vmax=1e2), cmap='YlOrRd_r')

# Add color bar
cbar = plt.colorbar(contour, ax=ax, ticks=levels)
cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(10))
cbar.set_label('Log Magnitude', fontweight='bold')

# Set labels and title
ax.set_xlabel('Parameter α', fontweight='bold')
ax.set_ylabel('Parameter β', fontweight='bold')
ax.set_title('Exponential Function Surface: f(α,β) = e^{-α²-β²} + 40e^{-64α²-64β²}')

# Set limits
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.8, 1.8)

# Add gridlines
ax.grid(True, linestyle='--', color='0.7', alpha=0.45)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/8372281327ad4cc0b7266b0363c68e1a.png", bbox_inches="tight")
