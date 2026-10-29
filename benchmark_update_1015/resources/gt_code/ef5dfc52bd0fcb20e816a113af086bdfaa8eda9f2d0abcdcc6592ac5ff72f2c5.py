import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5, 100)
y = 0.5 * x**2 + 2 * x - 1

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, color='darkblue', linewidth=2.5, linestyle='--', marker='s', 
        markersize=4, markeredgecolor='black', markerfacecolor='red')

ax.set_xlabel('Time (s)', fontsize=12, fontfamily='serif')
ax.set_ylabel('Velocity (m/s)', fontsize=12, fontfamily='serif')
ax.set_title('Quadratic Velocity Function: v(t) = 0.5t² + 2t - 1', 
             fontsize=14, fontweight='bold')

ax.grid(True, linestyle=':', alpha=0.7, color='gray')
ax.tick_params(axis='y', rotation=55)
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()