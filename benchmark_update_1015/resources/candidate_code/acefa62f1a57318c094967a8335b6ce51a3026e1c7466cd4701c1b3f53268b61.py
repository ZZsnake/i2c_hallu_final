import numpy as np
import matplotlib.pyplot as plt

# Set figure size
plt.figure(figsize=(5, 4))

# Define grid
x = np.linspace(-1.0, 1.0, 9)
y = np.linspace(-1.0, 1.0, 9)
X, Y = np.meshgrid(x, y)

# Define vector field: -F + ρ∇FF (Small ρ)
# For simplicity, we model a rotational field with a small radial component
# Let's assume F = (y, -x) for rotation, and ρ∇FF is a small radial inward force
# We'll use ρ = 0.1 for "small ρ"
rho = 0.1

# Define vector components
U = -Y + rho * X  # -F_x + ρ * ∂F_x/∂x (approximated as radial component)
V = X + rho * Y   # -F_y + ρ * ∂F_y/∂y (approximated as radial component)

# Plot vector field
plt.quiver(X, Y, U, V, color='blue', angles='xy', scale_units='xy', scale=1.0)

# Add grid
plt.grid(True, color='gray', linestyle='-', linewidth=0.5)

# Set axis labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field: -F + ρ∇FF (Small ρ)')

# Set axis limits
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)

# Show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/quiver_4.png", bbox_inches="tight")