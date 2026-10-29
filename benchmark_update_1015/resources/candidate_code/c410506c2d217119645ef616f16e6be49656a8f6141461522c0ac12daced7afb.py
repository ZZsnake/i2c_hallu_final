import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Generate data
alpha = np.linspace(-2, 2, 100)
beta = np.linspace(-2, 2, 100)
alpha, beta = np.meshgrid(alpha, beta)
f = np.exp(-alpha**2 - beta**2) + 40 * np.exp(-64 * (alpha**2 + beta**2))

# Create plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(alpha, beta, f, levels=100, 
                      norm=LogNorm(vmin=1e-5, vmax=1e2), 
                      cmap='YlOrRd_r')
cbar = plt.colorbar(contour)
cbar.set_label('Log Magnitude')
cbar.set_ticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])

# Labels and title
plt.xlabel(r'Parameter $\alpha$')
plt.ylabel(r'Parameter $\beta$')
plt.title(r'Exponential Function Surface: f($\alpha,\beta$) = $e^{-\alpha^2-\beta^2} + 40e^{-64\alpha^2-64\beta^2}$')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/8372281327ad4cc0b7266b0363c68e1a.png", bbox_inches="tight")