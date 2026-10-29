import numpy as np
import matplotlib.pyplot as plt

# Create dummy data
alpha = np.linspace(-2, 2, 400)
beta = np.linspace(-1.5, 1.5, 400)
A, B = np.meshgrid(alpha, beta)
Z = np.exp(-A**2 - B**2) + 40 * np.exp(-64 * A**2 - 64 * B**2)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(A, B, Z, levels=np.logspace(-5, 2, 100), cmap='YlOrRd')

# Add color bar
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Log Magnitude')

# Set labels and title
ax.set_xlabel('Parameter α')
ax.set_ylabel('Parameter β')
ax.set_title('Exponential Function Surface: f(α,β) = e^{-α²-β²} + 40e^{-64α²-64β²}')

# Set limits
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 1.5)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/8372281327ad4cc0b7266b0363c68e1a.png", bbox_inches="tight")