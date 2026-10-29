import numpy as np
import matplotlib.pyplot as plt

# Generate dummy data
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
X, Y = np.meshgrid(x, y)

# Define vector field components
U = -Y + 0.1 * X * Y
V = X + 0.1 * Y * X

# Create the plot
plt.figure(figsize=(5, 4))
plt.quiver(X, Y, U, V, color='#3171ad')

# Set plot properties
plt.title('Vector Field: -F + ρ∇FF (Small ρ)', fontsize=12)
plt.xlabel('x', fontsize=10, fontfamily='sans-serif')
plt.ylabel('y', fontsize=10, fontfamily='sans-serif')
plt.xlim(-1.10, 1.10)
plt.ylim(-1.10, 1.10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(np.arange(-1.00, 1.01, 0.25))
plt.yticks(np.arange(-1.00, 1.01, 0.25))

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/quiver_4.png", bbox_inches="tight")
