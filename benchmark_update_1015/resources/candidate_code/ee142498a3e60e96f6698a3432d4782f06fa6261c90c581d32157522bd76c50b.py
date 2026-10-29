import numpy as np
import matplotlib.pyplot as plt

# Data for sigmoid
x = np.linspace(-8, 8, 400)
alpha = 1.0
f = 1 / (1 + np.exp(-alpha * x))

# Tangent at x0 = 0
x0 = 0.0
f0 = 1 / (1 + np.exp(-alpha * x0))  # 0.5
slope = alpha * f0 * (1 - f0)       # derivative of sigmoid at x0
tangent = f0 + slope * (x - x0)

# Create figure
plt.figure(figsize=(9, 6))

# Plot main sigmoid curve
plt.plot(x, f, color='#d62728', linewidth=3, label=r'$f(x) = \frac{1}{1 + e^{-\alpha x}}$')

# Plot tangent line
plt.plot(x, tangent, linestyle='--', color='purple', linewidth=2)

# Horizontal line at f(0) = 0.5
plt.axhline(0.5, color='red', linestyle='--', linewidth=1.5)

# Asymptotes at y = 0 and y = 1
plt.axhline(0.0, color='#1f3b7f', linestyle='-.', linewidth=1.5)
plt.axhline(1.0, color='#1f3b7f', linestyle='-.', linewidth=1.5)

# Vertical line at x = 0
plt.axvline(0.0, color='green', linewidth=2)

# Grid
plt.grid(True, linestyle='--', color='lightgray', alpha=0.7)

# Axis labels
plt.xlabel('Input Variable x')
plt.ylabel('Activation f(x)')

# Axis limits and ticks
plt.xlim(-8, 8)
plt.ylim(0, 1.05)
plt.yticks(np.linspace(0, 1.0, 6))

# Legend in lower right
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/b3ec5580320149faaa38e96004215c3f.png", bbox_inches="tight")