import matplotlib.pyplot as plt
import squarify  # pip install squarify

# Data
labels = [
    "Chanel\n40%",
    "Dior\n20%",
    "Gucci\n15%",
    "Yves Saint Laurent\n10%",
    "Tom Ford\n8%",
    "Jo Malone\n7%",
]
sizes = [40, 20, 15, 10, 8, 7]

# Colors selected to visually match the reference image
colors = [
    "#FFE5CC",  # Chanel - light peach
    "#FFCCCC",  # Dior - light pink
    "#FF9999",  # Gucci - coral
    "#B3A2C7",  # Yves Saint Laurent - lavender
    "#B2D68C",  # Tom Ford - light olive
    "#F7D7E3",  # Jo Malone - very light pink
]

# Create figure
plt.figure(figsize=(12, 8))

# Draw treemap
# Normalize sizes to sum to 100 and use a little padding between rectangles
normed_sizes = squarify.normalize_sizes(sizes, 100, 100)
rects = squarify.squarify(normed_sizes, 0, 0, 100, 100)

# Plot rectangles
for rect, color, label in zip(rects, colors, labels):
    x, y, dx, dy = rect['x'], rect['y'], rect['dx'], rect['dy']
    plt.gca().add_patch(
        plt.Rectangle(
            (x, y),
            dx,
            dy,
            facecolor=color,
            edgecolor='white',
            linewidth=1
        )
    )
    # Centered text
    plt.text(
        x + dx / 2,
        y + dy / 2,
        label,
        ha='center',
        va='center',
        fontsize=14,
        fontweight='bold',
        color='black'
    )

# Set limits and remove axes
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.axis('off')

# Title
plt.title(
    "Market Share of Leading Perfume Brand",
    fontsize=22,
    fontweight='bold',
    pad=20
)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/tree_14.png", bbox_inches="tight")