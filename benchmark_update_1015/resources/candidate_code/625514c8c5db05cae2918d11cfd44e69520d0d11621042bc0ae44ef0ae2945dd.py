import matplotlib.pyplot as plt
import squarify

# Data
labels = ['Louis Vuitton\n38.27%', 'Gucci\n24.36%', 'Chanel\n18.04%', 'Hermès\n9.12%', 'Rolex\n6.84%', 'Others\n3.37%']
sizes = [38.27, 24.36, 18.04, 9.12, 6.84, 3.37]
colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a']

# Create figure with specified dimensions
plt.figure(figsize=(12, 8))

# Create treemap
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    alpha=0.7,
    text_kwargs={'fontsize': 18},
    bar_kwargs={'edgecolor': 'black', 'linewidth': 1}
)

# Add title
plt.title('Market Share of Leading Luxury Brands', fontsize=22, fontweight='normal', pad=2)

# Remove axes
plt.axis('off')

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/tree_13.png", bbox_inches="tight")
