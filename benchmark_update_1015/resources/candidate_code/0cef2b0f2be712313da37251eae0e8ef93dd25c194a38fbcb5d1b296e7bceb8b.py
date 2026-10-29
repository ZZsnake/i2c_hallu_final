import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Generate data
alpha = np.linspace(-2.5, 2.5, 100)
beta = np.linspace(-1.8, 1.8, 100)
alpha, beta = np.meshgrid(alpha, beta)
f = np.exp(-alpha**2 - beta**2) + 40 * np.exp(-64 * (alpha**2 + beta**2))
f = np.ma.masked_where((alpha < -2.2) & (beta < -1.6), f)

# Create plot
plt.figure(figsize=(9, 7))
levels = np.logspace(-5, 2, 8)
contour = plt.contourf(alpha, beta, f, levels=levels,
                      norm=LogNorm(vmin=1e-5, vmax=1e2),
                      cmap='YlOrRd_r', alpha=0.85)
cbar = plt.colorbar(contour)
cbar.set_label('Log Magnitude', fontsize=12, fontweight='bold')
cbar.set_ticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])

# Labels and title
plt.xlabel('Parameter α', fontsize=12, fontweight='bold')
plt.ylabel('Parameter β', fontsize=12, fontweight='bold')
plt.title('Exponential Function Surface: f(α,β) = e^{-α²-β²} + 40e^{-64α²-64β²}', fontsize=14)
plt.grid(True, linestyle='--', color='0.7', alpha=0.5, linewidth=0.7)
plt.xlim(-2.5, 2.5)
plt.ylim(-1.8, 1.8)
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/8372281327ad4cc0b7266b0363c68e1a.png", bbox_inches="tight")
