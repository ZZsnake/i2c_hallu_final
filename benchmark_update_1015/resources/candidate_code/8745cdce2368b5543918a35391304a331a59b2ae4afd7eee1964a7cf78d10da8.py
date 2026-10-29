import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Data
companies = ['BYD', 'Tesla', 'Volkswagen', 'Ford', 'Nio', 'Others']
market_shares = [25, 35, 15, 10, 10, 5]
colors = {
    'BYD': '#345383',
    'Tesla': '#004b63',
    'Volkswagen': '#6a5598',
    'Ford': '#d34f86',
    'Nio': '#a1519b',
    'Others': '#fb5968'
}

# Create a figure with specified dimensions
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Treemap layout with white gutters between rectangular tiles
W, H = 100, 100
gap = 2.0
left_w = 59.0
right_x = left_w + gap
right_w = W - right_x

# Left column: BYD over Tesla, proportional to their combined share
left_available_h = H - gap
byd_h = left_available_h * 25 / (25 + 35)
tesla_h = left_available_h - byd_h

tiles = [
    ('Tesla', 35, 0, 0, left_w, tesla_h),
    ('BYD', 25, 0, tesla_h + gap, left_w, byd_h),
]

# Right column: top row Ford/Others, then Nio, then Volkswagen
right_available_h = H - 2 * gap
top_h = right_available_h * (10 + 5) / (10 + 5 + 10 + 15)
nio_h = right_available_h * 10 / (10 + 5 + 10 + 15)
volkswagen_h = right_available_h - top_h - nio_h

volkswagen_y = 0
nio_y = volkswagen_h + gap
top_y = nio_y + nio_h + gap

right_available_w = right_w - gap
ford_w = right_available_w * 10 / (10 + 4.5)
others_w = right_available_w - ford_w

tiles.extend([
    ('Volkswagen', 15, right_x, volkswagen_y, right_w, volkswagen_h),
    ('Nio', 10, right_x, nio_y, right_w, nio_h),
    ('Ford', 10, right_x, top_y, ford_w, top_h),
    ('Others', 5, right_x + ford_w + gap, top_y, others_w, top_h),
])

# Draw treemap tiles and centered labels
for name, share, x, y, w, h in tiles:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=colors[name], edgecolor='none'))
    ax.text(
        x + w / 2,
        y + h / 2,
        f'{name}\n{share}%',
        ha='center',
        va='center',
        color='white',
        fontsize=20
    )

# Set title and remove axes/frame
ax.set_title('Car Market Share', fontsize=24, pad=12)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Adjust layout
plt.tight_layout(pad=2.0)

# Show plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/tree_8.png", bbox_inches="tight")
