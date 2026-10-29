import numpy as np
import matplotlib.pyplot as plt

# Set figure size
plt.figure(figsize=(5, 4))

# Define grid
x = np.linspace(-1.0, 1.0, 10)
y = np.linspace(-1.0, 1.0, 10)
X, Y = np.meshgrid(x, y)

# Define vector field: -F + ρ∇FF (Small ρ)
# For simplicity, we model a pure rotational field
# Let's assume F = (y, -x) for rotation
# We'll use ρ = 0.1 for "small ρ"
rho = 0.1

# Define vector components
U = -Y + rho * X
V = X + rho * Y

# Plot vector field
plt.quiver(X, Y, U, V, color='#1f77b4', angles='xy', scale_units='xy', scale=7.0)

# Add grid
plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5)

# Set axis labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field: -F + ρ∇FF (Small ρ)')

# Set axis limits
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)

# Show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/quiver_4.png", bbox_inches="tight")
