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
colors = ['#f08080', '#f08080', '#90ee90', '#ba55d3', '#e9967a', '#ffb347']

# Create the treemap
plt.figure(figsize=(12, 8))
squarify.plot(sizes=list(data.values()), label=list(data.keys()), color=colors, alpha=.8)

# Add labels with percentages
for i, label in enumerate(data.keys()):
    plt.text(0.5, 0.5, f"{label}\n{data[label]}%", ha='center', va='center', color='white', fontsize=10)

# Remove axes
plt.axis('off')

# Set title
plt.title("Market Share of Leading Perfume Brand", fontsize=16)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/tree_14.png", bbox_inches="tight")