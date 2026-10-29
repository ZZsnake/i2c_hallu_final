import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Data
brands = ['Wilmar International', 'Cargill', 'ADM', 'Bunge', 'COFCO', 'Others']
market_shares = [25, 30, 20, 10, 8, 7]
colors = {
    'Wilmar International': '#FF6347',
    'Cargill': '#5AA378',
    'ADM': '#6FA0C4',
    'Bunge': '#E3BE4D',
    'COFCO': '#9C50DF',
    'Others': '#FF69B4'
}

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Treemap drawing area
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Layout constants
outer_x, outer_y = 1.0, 1.0
outer_w, outer_h = 98.0, 98.0
gap = 2.0

# Split into left and right groups proportional to their totals
left_total = 25 + 30
right_total = 20 + 10 + 8 + 7
usable_w = outer_w - gap
left_w = usable_w * left_total / (left_total + right_total)
right_w = usable_w - left_w
left_x = outer_x
right_x = outer_x + left_w + gap

# Left column: Wilmar over Cargill
left_usable_h = outer_h - gap
wilmar_h = left_usable_h * 25 / left_total
cargill_h = left_usable_h * 30 / left_total
rects = [
    ('Wilmar International', 25, left_x, outer_y + cargill_h + gap, left_w, wilmar_h),
    ('Cargill', 30, left_x, outer_y, left_w, cargill_h),
]

# Right column: Others on top, Bunge/COFCO in middle, ADM at bottom
right_usable_h = outer_h - 2 * gap
others_h = right_usable_h * 7 / right_total
middle_h = right_usable_h * (10 + 8) / right_total
adm_h = right_usable_h * 20 / right_total

rects.append(('Others', 7, right_x, outer_y + adm_h + gap + middle_h + gap, right_w, others_h))

middle_y = outer_y + adm_h + gap
middle_usable_w = right_w - gap
bunge_w = middle_usable_w * 10 / (10 + 8)
cofco_w = middle_usable_w * 8 / (10 + 8)
rects.extend([
    ('Bunge', 10, right_x, middle_y, bunge_w, middle_h),
    ('COFCO', 8, right_x + bunge_w + gap, middle_y, cofco_w, middle_h),
    ('ADM', 20, right_x, outer_y, right_w, adm_h),
])

# Draw treemap tiles and labels
for brand, share, x, y, w, h in rects:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=colors[brand], edgecolor='black', linewidth=1.0))
    ax.text(
        x + w / 2,
        y + h / 2,
        f'{brand}\n{share}%',
        ha='center',
        va='center',
        color='white',
        fontsize=14,
        fontweight='bold'
    )

# Title
ax.set_title('Market Share of Edible Oil Brands', fontsize=16, fontweight='bold', pad=8)

# Adjust layout
plt.tight_layout(pad=0.3)

# Show plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/tree_20.png", bbox_inches="tight")
