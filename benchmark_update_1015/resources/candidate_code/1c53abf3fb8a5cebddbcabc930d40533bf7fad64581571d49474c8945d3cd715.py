import matplotlib.pyplot as plt
import numpy as np

# Create the data
alpha = np.linspace(0.2, 1.5, 12)
beta = np.linspace(0.2, 1.5, 12)
alpha, beta = np.meshgrid(alpha, beta)
product = alpha * beta

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 6))

# Create the heatmap
im = ax.scatter(alpha, beta, c=product, cmap='viridis', marker='d', s=80, edgecolors='black', linewidths=0.5)

# Set the axis labels
ax.set_xlabel("Variable α")
ax.set_ylabel("Variable β")

# Add the colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Product Value")

# Set the axis limits
ax.set_xlim(0.15, 1.57)
ax.set_ylim(0.13, 1.57)

# Add grid lines
ax.grid(True, linestyle='--', alpha=0.5)

# Set the title
ax.set_title("Mathematical Relationship: f(α,β) = α·β", fontweight='bold')

# Show the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/516b1589df5b4185be0b1b914ca896f9.png", bbox_inches="tight")