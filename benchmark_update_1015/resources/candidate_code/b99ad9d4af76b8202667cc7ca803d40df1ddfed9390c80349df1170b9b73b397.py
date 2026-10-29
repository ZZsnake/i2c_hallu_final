import matplotlib.pyplot as plt
import squarify

# Data for the treemap
labels = ['Tesla', 'BYD', 'Volkswagen', 'Ford', 'Nio', 'Others']
sizes = [35, 25, 15, 10, 10, 5]
colors = ['#0047AB', '#2E3B6A', '#6A4C93', '#C74C9D', '#B36B9B', '#FF6B6B']

# Create figure with specified dimensions
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

# Create the treemap
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, text_kwargs={'fontsize': 14, 'color': 'white', 'fontweight': 'bold'}, pad=True, ax=ax)

# Set title
ax.set_title('Car Market Share', fontsize=20, fontweight='bold', pad=20)

# Remove axes
ax.axis('off')

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/tree_8.png", bbox_inches="tight")