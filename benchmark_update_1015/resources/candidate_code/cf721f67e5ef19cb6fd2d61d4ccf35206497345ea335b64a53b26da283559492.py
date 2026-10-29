import matplotlib.pyplot as plt
import squarify

# Data based on the image
data = {
    'Chanel': 40,
    'Dior': 20,
    'Gucci': 15,
    'Yves Saint Laurent': 10,
    'Tom Ford': 8,
    'Jo Malone': 7
}

# Sort data by value in descending order for the squarify algorithm
sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)

# Extract sizes, labels, and corresponding colors
sizes = [item[1] for item in sorted_data]
labels = [f"{item[0]}\n{item[1]}%" for item in sorted_data]

# Color palette matching the image
colors_map = {
    'Chanel': '#FFDDC1',
    'Dior': '#FDA2A8',
    'Gucci': '#FF6F61',
    'Yves Saint Laurent': '#6B5B95',
    'Tom Ford': '#88B04B',
    'Jo Malone': '#F7CAC9'
}
colors = [colors_map[item[0]] for item in sorted_data]

# Set the figure dimensions
plt.figure(figsize=(12, 8))

# Create the treemap
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    edgecolor='black',
    linewidth=1,
    text_kwargs={'fontsize': 18, 'fontweight': 'bold', 'fontname': 'sans-serif'}
)

# Add the title
plt.title('Market Share of Leading Perfume Brand', fontsize=20, fontweight='bold', pad=6, fontname='sans-serif')

# Remove axes
plt.axis('off')

# Ensure the layout is tight
plt.tight_layout()

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/tree_14.png", bbox_inches="tight")
