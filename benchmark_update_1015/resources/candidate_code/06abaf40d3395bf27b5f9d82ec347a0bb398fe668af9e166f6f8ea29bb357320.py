import matplotlib.pyplot as plt
import numpy as np

# Data
brands = ["Louis Vuitton", "Gucci", "Chanel", "Hermès", "Rolex", "Others"]
shares = [38.27, 24.36, 18.04, 9.12, 6.84, 3.37]
colors = ["#4682B4", "#ADD8E6", "#FF8C00", "#FFDAB9", "#3CB371", "#90EE90"]

# Calculate total width and height for the treemap
total_width = 12
total_height = 8

# Define positions and sizes for each rectangle
# Left column: Louis Vuitton (bottom), Gucci (top)
lv_width = total_width * 0.5
lv_height = total_height * (38.27 / 100)
gucci_height = total_height * (24.36 / 100)

# Right column: Chanel, Hermès, Rolex, Others (bottom to top)
right_width = total_width * 0.5
chanel_height = total_height * (18.04 / 100)
hermes_height = total_height * (9.12 / 100)
rolex_height = total_height * (6.84 / 100)
others_height = total_height * (3.37 / 100)

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, total_width)
ax.set_ylim(0, total_height)
ax.axis('off')  # Hide axes

# Draw rectangles
rects = []

# Left column
rect_lv = plt.Rectangle((0, 0), lv_width, lv_height, color=colors[0], linewidth=1, edgecolor='black')
rect_gucci = plt.Rectangle((0, lv_height), lv_width, gucci_height, color=colors[1], linewidth=1, edgecolor='black')
ax.add_patch(rect_lv)
ax.add_patch(rect_gucci)

# Right column
rect_chanel = plt.Rectangle((total_width * 0.5, 0), right_width, chanel_height, color=colors[2], linewidth=1, edgecolor='black')
rect_hermes = plt.Rectangle((total_width * 0.5, chanel_height), right_width, hermes_height, color=colors[3], linewidth=1, edgecolor='black')
rect_rolex = plt.Rectangle((total_width * 0.5, chanel_height + hermes_height), right_width, rolex_height, color=colors[4], linewidth=1, edgecolor='black')
rect_others = plt.Rectangle((total_width * 0.5, chanel_height + hermes_height + rolex_height), right_width, others_height, color=colors[5], linewidth=1, edgecolor='black')
ax.add_patch(rect_chanel)
ax.add_patch(rect_hermes)
ax.add_patch(rect_rolex)
ax.add_patch(rect_others)

# Add text inside each rectangle
text_props = {'ha': 'center', 'va': 'center', 'fontsize': 12, 'color': 'black', 'fontweight': 'normal'}

# Left column
ax.text(lv_width / 2, lv_height / 2, f"{brands[0]}\n{shares[0]:.2f}%", **text_props)
ax.text(lv_width / 2, lv_height + gucci_height / 2, f"{brands[1]}\n{shares[1]:.2f}%", **text_props)

# Right column
ax.text(total_width * 0.5 + right_width / 2, chanel_height / 2, f"{brands[2]}\n{shares[2]:.2f}%", **text_props)
ax.text(total_width * 0.5 + right_width / 2, chanel_height + hermes_height / 2, f"{brands[3]}\n{shares[3]:.2f}%", **text_props)
ax.text(total_width * 0.5 + right_width / 2, chanel_height + hermes_height + rolex_height / 2, f"{brands[4]}\n{shares[4]:.2f}%", **text_props)
ax.text(total_width * 0.5 + right_width / 2, chanel_height + hermes_height + rolex_height + others_height / 2, f"{brands[5]}\n{shares[5]:.2f}%", **text_props)

# Add title
plt.title("Market Share of Leading Luxury Brands", fontsize=16, fontweight='bold', pad=20)

# Save or show
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/tree_13.png", bbox_inches="tight")