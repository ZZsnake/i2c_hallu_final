import matplotlib.pyplot as plt
import numpy as np

# Generate data for the scatter plot
theta = np.linspace(0.2, 1.5, 12)
X, Y = np.meshgrid(theta, theta)
x_vals, y_vals = X.ravel(), Y.ravel()

# Create figure and plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x_vals, y_vals, c=x_vals*y_vals, 
                    cmap='viridis', marker='D', s=85, alpha=0.8, edgecolors='black', linewidth=0.5)

# Add labels and title
ax.set_xlabel('Variable α', fontsize=12, fontfamily='serif')
ax.set_ylabel('Variable β', fontsize=12, fontfamily='serif')
ax.set_title('Mathematical Relationship: f(α,β) = α·β', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Product Value', rotation=270, labelpad=15)

# Customize grid and ticks
ax.grid(True, linestyle='--', alpha=0.3)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()