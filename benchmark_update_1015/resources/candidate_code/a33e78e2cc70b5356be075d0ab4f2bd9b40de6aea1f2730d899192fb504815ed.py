import numpy as np
import matplotlib.pyplot as plt

# Set figure size
plt.figure(figsize=(9, 6))

# Define x range
x = np.linspace(-8, 8, 1000)

# Sigmoid function
f_x = 1 / (1 + np.exp(-x))

# Plot sigmoid function (red solid line)
plt.plot(x, f_x, color='crimson', linewidth=2.5, label=r'$f(x) = \frac{1}{1 + e^{-x}}$')

# Plot linear approximation (purple dashed line)
# This line passes through (0, 0.5) and has slope ~0.25 (approximate derivative at x=0)
# Derivative of sigmoid at x=0 is 0.25, so line: y = 0.25*x + 0.5
linear_approx = 0.25 * x + 0.5
plt.plot(x, linear_approx, color='purple', linestyle='--', linewidth=2.0)

# Horizontal dashed lines at y=0.0 and y=1.0 (blue)
plt.axhline(y=0.0, color='darkblue', linestyle='--', linewidth=1.5)
plt.axhline(y=1.0, color='darkblue', linestyle='--', linewidth=1.5)

# Horizontal dashed line at y=0.5 (brown)
plt.axhline(y=0.5, color='saddlebrown', linestyle='--', linewidth=1.5)

# Vertical dashed line at x=0 (green)
plt.axvline(x=0, color='green', linestyle='-', linewidth=1.5)

# Labels and title
plt.xlabel('Input Variable x', fontsize=12, fontweight='bold')
plt.ylabel('Activation f(x)', fontsize=12, fontweight='bold')

# Grid
plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray')

# Set axis limits
plt.xlim(-8, 8)
plt.ylim(0.0, 1.0)

# Legend
legend = plt.legend(loc='lower right', frameon=True, framealpha=1.0, edgecolor='black', fontsize=12)
legend.get_frame().set_linewidth(1.0)

# Adjust layout to prevent clipping
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/b3ec5580320149faaa38e96004215c3f.png", bbox_inches="tight")