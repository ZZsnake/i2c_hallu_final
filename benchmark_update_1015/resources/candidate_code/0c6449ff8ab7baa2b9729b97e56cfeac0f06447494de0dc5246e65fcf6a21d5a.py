import numpy as np
import matplotlib.pyplot as plt

# Set figure size
plt.figure(figsize=(8, 6))

# Define x range
x = np.linspace(-8, 8, 1000)

# Sigmoid function
alpha = 0.7
f_x = 1 / (1 + np.exp(-alpha * x))

# Plot sigmoid function (red solid line)
plt.plot(x, f_x, color='crimson', linewidth=3.0, label=r'$f(x) = \frac{1}{1 + e^{-\alpha x}}$')

# Plot linear approximation (purple dashed line)
# This line passes through (0, 0.5) and has slope alpha/4 (approximate derivative at x=0)
# Derivative of sigmoid at x=0 is alpha/4, so line: y = (alpha/4)*x + 0.5
linear_approx = (alpha / 4) * x + 0.5
plt.plot(x, linear_approx, color='purple', linestyle='--', linewidth=2.0)

# Horizontal dash-dot lines at y=0.0 and y=1.0 (blue)
plt.axhline(y=0.0, color='darkblue', linestyle='-.', linewidth=1.5)
plt.axhline(y=1.0, color='darkblue', linestyle='-.', linewidth=1.5)

# Horizontal dashed line at y=0.5 (brown)
plt.axhline(y=0.5, color='maroon', linestyle='--', linewidth=1.5)

# Vertical dashed line at x=0 (green)
plt.axvline(x=0, color='green', linestyle='-', linewidth=1.5)

# Labels and title
plt.xlabel('Input Variable x')
plt.ylabel('Activation f(x)')

# Grid
plt.grid(True, which='major', linestyle=':', linewidth=0.5, color='lightgray')

# Set axis limits
plt.xlim(-8, 8)
plt.ylim(-0.05, 1.05)

# Legend
legend = plt.legend(loc='lower right', fontsize=12)
legend.get_frame().set_linewidth(1.0)

# Adjust layout to prevent clipping
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/b3ec5580320149faaa38e96004215c3f.png", bbox_inches="tight")
