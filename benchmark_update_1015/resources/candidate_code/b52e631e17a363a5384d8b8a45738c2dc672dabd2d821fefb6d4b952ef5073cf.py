import matplotlib.pyplot as plt
import numpy as np

# Create figure with specified dimensions
plt.figure(figsize=(8, 6))

# Generate dummy data for alpha and beta
alpha = np.linspace(0.2, 1.5, 12)
beta = np.linspace(0.2, 1.5, 12)
Alpha, Beta = np.meshgrid(alpha, beta)

# Calculate product value for color mapping
product_value = Alpha * Beta

# Create a scatter plot with diamond markers
scatter = plt.scatter(Alpha.ravel(), Beta.ravel(), 
                      c=product_value.ravel(), 
                      cmap='viridis', 
                      marker='D', 
                      edgecolors='black', 
                      linewidths=0.5,
                      alpha=0.75,
                      s=80)

# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_label('Product Value', rotation=270, labelpad=20)

# Set axis labels
plt.xlabel('Variable α', fontsize=12, fontfamily='serif')
plt.ylabel('Variable β', fontsize=12, fontfamily='serif')

# Set title
plt.title('Mathematical Relationship: f(α,β) = α·β', fontsize=14, fontweight='bold')

# Set axis limits
plt.xlim(0.135, 1.565)
plt.ylim(0.135, 1.565)

# Set grid
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

# Set tick labels
plt.xticks(np.arange(0.2, 1.6, 0.2))
plt.yticks(np.arange(0.2, 1.6, 0.2))

# Adjust layout to prevent clipping
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/516b1589df5b4185be0b1b914ca896f9.png", bbox_inches="tight")
