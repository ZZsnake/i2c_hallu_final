import matplotlib.pyplot as plt
import squarify

# Data
labels = ['Graphic Design - 40%', 'Interior Design - 25%', 'Web Design - 15%', 
          'Fashion Design - 10%', 'Product Design - 5%', 'Illustration - 5%']
sizes = [40, 25, 15, 10, 5, 5]
colors = ['#FFD700', '#FF6347', '#4682B4', '#32CD32', '#FF69B4', '#BA55D3']

# Create figure with specified dimensions
plt.figure(figsize=(10, 8))

# Create treemap
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, bar_kwargs={'edgecolor': 'white', 'linewidth': 2}, text_kwargs={'fontsize':12, 'color':'black'})

# Title
plt.title('Art and Design Categories Distribution', fontsize=18, fontweight='bold')

# Remove axes
plt.axis('off')

# Adjust layout to prevent clipping
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/tree_15.png", bbox_inches="tight")
