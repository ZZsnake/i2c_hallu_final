import numpy as np
import matplotlib.pyplot as plt

# Generate dummy data
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
X, Y = np.meshgrid(x, y)

# Define vector field components
U = -Y + 0.1 * X * Y
V = X + 0.1 * Y * X

# Create the plot
plt.figure(figsize=(5, 4))
plt.quiver(X, Y, U, V, color='blue')

# Set plot properties
plt.title('Vector Field: -F + ρ∇FF (Small ρ)', fontsize=12, fontfamily='serif')
plt.xlabel('x', fontsize=10, fontfamily='sans-serif')
plt.ylabel('y', fontsize=10, fontfamily='sans-serif')
plt.xlim(-1.00, 1.00)
plt.ylim(-1.00, 1.00)
plt.grid(True)
plt.xticks(np.arange(-1.00, 1.01, 0.25))
plt.yticks(np.arange(-1.00, 1.01, 0.25))

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/quiver_4.png", bbox_inches="tight")