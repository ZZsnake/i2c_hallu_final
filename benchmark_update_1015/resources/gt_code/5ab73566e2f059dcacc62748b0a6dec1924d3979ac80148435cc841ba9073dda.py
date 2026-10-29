import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from matplotlib import cm, ticker

N = 120
x = np.linspace(-2.5, 2.5, N)
y = np.linspace(-1.8, 1.8, N)

X, Y = np.meshgrid(x, y)

Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X * 8)**2 - (Y * 8)**2)
z = Z1 + 40 * Z2

z[:7, :7] = -1
z = ma.masked_where(z <= 0, z)

fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contourf(X, Y, z, locator=ticker.LogLocator(), cmap=cm.YlOrRd_r, alpha=0.85)

ax.set_xlabel('Parameter α', fontsize=12, fontweight='bold')
ax.set_ylabel('Parameter β', fontsize=12, fontweight='bold')
ax.set_title('Exponential Function Surface: f(α,β) = e^{-α²-β²} + 40e^{-64α²-64β²}', fontsize=14)

cbar = fig.colorbar(cs)
cbar.set_label('Log Magnitude', rotation=270, labelpad=20, fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
plt.tight_layout()
plt.show()