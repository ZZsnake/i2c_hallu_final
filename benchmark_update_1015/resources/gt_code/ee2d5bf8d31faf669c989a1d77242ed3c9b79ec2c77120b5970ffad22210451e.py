import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-8, 8, 120)
sigmoid = 1 / (1 + np.exp(-0.7 * x))

fig, ax = plt.subplots(figsize=(9, 6))
ax.axhline(y=0, color='navy', linestyle='-.', alpha=0.8)
ax.axhline(y=0.5, color='maroon', linestyle=(0, (3, 3)), linewidth=1.5)
ax.axhline(y=1.0, color='navy', linestyle='-.', alpha=0.8)
ax.axvline(color='darkgreen', alpha=0.6)
ax.axline((0, 0.5), slope=0.18, color='purple', linestyle=(0, (4, 2)))
ax.plot(x, sigmoid, linewidth=3, color='crimson', 
        label=r"$f(x) = \frac{1}{1 + e^{-\alpha x}}$")
ax.set(xlim=(-8, 8), xlabel="Input Variable x", ylabel="Activation f(x)")
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, linestyle=':', alpha=0.4)
plt.show()