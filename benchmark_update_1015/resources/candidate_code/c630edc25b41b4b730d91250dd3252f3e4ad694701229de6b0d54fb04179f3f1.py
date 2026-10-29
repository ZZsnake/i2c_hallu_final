import matplotlib.pyplot as plt
import squarify

# Data for the treemap
data = {
    'Chanel': 40,
    'Dior': 20,
    'Gucci': 15,
    'Yves Saint Laurent': 10,
    'Tom Ford': 8,
    'Jo Malone': 7
}

# Colors for each brand
colors = ['#ffe5cc', '#ffb6c1', '#ff8f88', '#9b91b8', '#aec983', '#f7d6d6']

# Create the treemap
plt.figure(figsize=(12, 8))
labels = [f"{label}\n{value}%" for label, value in data.items()]
squarify.plot(
    sizes=list(data.values()),
    label=labels,
    color=colors,
    alpha=1,
    edgecolor='black',
    linewidth=1,
    text_kwargs={'fontsize': 18, 'fontweight': 'bold', 'color': 'black'}
)

# Remove axes
plt.axis('off')

plt.tight_layout()
# Set title
plt.title("Market Share of Leading Perfume Brand", fontsize=22, fontweight='bold')

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/tree_14.png", bbox_inches="tight")
