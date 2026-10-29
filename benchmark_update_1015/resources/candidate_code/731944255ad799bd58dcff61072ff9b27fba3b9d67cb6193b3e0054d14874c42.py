import numpy as np
import matplotlib.pyplot as plt

# Generate data for sigmoid function
x = np.linspace(-8, 8, 1000)
y = 1 / (1 + np.exp(-x))

# Create figure with specified dimensions
fig, ax = plt.subplots(figsize=(9, 6))

# Plot the sigmoid curve
ax.plot(x, y, color='crimson', linewidth=2.5, label=r'$f(x) = \frac{1}{1+e^{-x}}$')

# Add horizontal reference lines
ax.axhline(y=0.5, color='brown', linestyle='--', linewidth=1.5)
ax.axhline(y=0.0, color='darkblue', linestyle='-.', linewidth=1.2)
ax.axhline(y=1.0, color='darkblue', linestyle='-.', linewidth=1.2)

# Add vertical reference line at x=0
ax.axvline(x=0, color='darkgreen', linestyle='-', linewidth=1.5)

# Set axis labels
ax.set_xlabel('Input Variable x', fontsize=11)
ax.set_ylabel('Activation f(x)', fontsize=11)

# Set axis limits
ax.set_xlim(-8, 8)
ax.set_ylim(0, 1.0)

# Add grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add legend
ax.legend(loc='lower right', fontsize=11, frameon=True)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/b3ec5580320149faaa38e96004215c3f.png", bbox_inches="tight")