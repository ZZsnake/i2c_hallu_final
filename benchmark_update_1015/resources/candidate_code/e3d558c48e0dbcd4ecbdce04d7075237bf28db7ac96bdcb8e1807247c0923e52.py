import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Create data grid
alpha = np.linspace(0.2, 1.5, 12)
beta = np.linspace(0.2, 1.5, 12)
Alpha, Beta = np.meshgrid(alpha, beta)

# Calculate product f(α,β) = α·β
product = Alpha * Beta

# Create figure with specified dimensions
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with diamond markers
scatter = ax.scatter(Alpha.flatten(), Beta.flatten(), 
                     c=product.flatten(), 
                     cmap='viridis', 
                     marker='D', 
                     s=85, 
                     edgecolors='black', 
                     linewidths=0.5,
                     alpha=0.8)

# Set labels and title
ax.set_xlabel('Variable α', fontsize=12, fontfamily='serif')
ax.set_ylabel('Variable β', fontsize=12, fontfamily='serif')
ax.set_title('Mathematical Relationship: f(α,β) = α·β', fontsize=14, fontweight='bold')

# Set axis limits
ax.set_xlim(0.135, 1.565)
ax.set_ylim(0.135, 1.565)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Product Value', fontsize=11)

# Add dashed gray gridlines
ax.set_axisbelow(True)
ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Display plot
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/516b1589df5b4185be0b1b914ca896f9.png", bbox_inches="tight")
