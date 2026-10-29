import numpy as np
import matplotlib.pyplot as plt

# Set figure dimensions
plt.figure(figsize=(9, 6))

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate x values
x = np.linspace(-8, 8, 400)

# Calculate corresponding y values
y = sigmoid(x)

# Plot the sigmoid function
plt.plot(x, y, 'r-', label=r'$f(x) = \frac{1}{1+e^{-\alpha x}}$')

# Add horizontal dashed lines
plt.axhline(y=0.4, color='k', linestyle='--')
plt.axhline(y=0.6, color='k', linestyle='--')

# Add vertical dashed line at x=0
plt.axvline(x=0, color='g', linestyle='--')

# Set axis limits
plt.xlim(-8, 8)
plt.ylim(0, 1)

# Label axes
plt.xlabel('Input Variable x')
plt.ylabel('Activation f(x)')

# Add legend
plt.legend(loc='lower right')

# Show grid
plt.grid(True)

# Display the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/b3ec5580320149faaa38e96004215c3f.png", bbox_inches="tight")