import numpy as np
import matplotlib.pyplot as plt

# Set figure size
plt.figure(figsize=(12, 9))
ax = plt.gca()
ax.set_facecolor('0.75')

# Define grid
x = np.linspace(-15, 15, 300)
y = np.linspace(-15, 15, 300)
X, Y = np.meshgrid(x, y)

# Compute radial distance
R = np.sqrt(X**2 + Y**2)
R_safe = np.where(R == 0, 1e-10, R)

# Define scalar field intensity (example: a radial function with some variation)
# Field intensity increases with radius, with a peak at center and some oscillation
Z = 9.0 * np.exp(-R/5) * (1 + 0.5 * np.sin(R/2))  # Example field intensity

# Create contour plot for background field intensity
contour_levels = np.linspace(-1.5, 9.0, 8)
contour = plt.contourf(X, Y, Z, levels=contour_levels, cmap='viridis', alpha=0.35)

# Add colorbar
cbar = plt.colorbar(contour, ax=plt.gca(), shrink=0.8, pad=0.05)
cbar.set_ticks(contour_levels)
cbar.set_label('Field Intensity', rotation=90, labelpad=15)

# Define Field A and Field B from radial scalar fields
field_A = np.sin(R_safe) / R_safe
field_A[R == 0] = 1.0
field_B = np.cos(R_safe) / R_safe

levels_A = np.linspace(-0.2, 0.8, 10)  # Field A: green dotted
levels_B = np.linspace(-0.2, 0.8, 10)  # Field B: red dash-dot

# Plot Field A (green dotted)
plt.contour(X, Y, field_A, levels=levels_A, colors='green', linestyles=':', linewidths=1.0, alpha=0.8)

# Plot Field B (red dash-dot)
plt.contour(X, Y, field_B, levels=levels_B, colors='red', linestyles='-.', linewidths=1.5, alpha=0.8)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2, label='Field A'),
    plt.Line2D([0], [0], color='red', linestyle='-.', linewidth=2, label='Field B')
]
plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=12)

# Add title and axis labels
plt.title('Scalar Field Distribution', fontsize=16, pad=20)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)

# Set axis limits
plt.xlim(-15, 15)
plt.ylim(-15, 15)
ax.set_aspect('equal', adjustable='box')

# Add grid
plt.grid(True, alpha=0.3)

# Adjust layout to prevent clipping
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/contour_15.png", bbox_inches="tight")
