import matplotlib.pyplot as plt
import numpy as np

# Data
brands = ["Louis Vuitton", "Gucci", "Chanel", "Hermès", "Rolex", "Others"]
shares = [38.27, 24.36, 18.04, 9.12, 6.84, 3.37]
colors = ["#1F77B4", "#AEC7E8", "#FF7F0E", "#FFBB78", "#2CA02C", "#98DF8A"]

# Calculate total width and height for the treemap
total_width = 12
total_height = 8

# Define positions and sizes for each rectangle
# Left column: Louis Vuitton (bottom), Gucci (top)
left_total = shares[0] + shares[1]
right_total = shares[2] + shares[3] + shares[4] + shares[5]
lv_width = total_width * (left_total / sum(shares))
right_width = total_width - lv_width
right_x = lv_width

lv_height = total_height * (shares[0] / left_total)
gucci_height = total_height * (shares[1] / left_total)

# Right column: Chanel, Hermès, and top row with Rolex and Others side-by-side
chanel_height = total_height * (shares[2] / right_total)
hermes_height = total_height * (shares[3] / right_total)
top_height = total_height * ((shares[4] + shares[5]) / right_total)
rolex_width = right_width * (shares[4] / (shares[4] + shares[5]))
others_width = right_width - rolex_width

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, total_width)
ax.set_ylim(0, total_height)
ax.axis('off')  # Hide axes

# Draw rectangles
rects = []

# Left column
rect_lv = plt.Rectangle((0, 0), lv_width, lv_height, facecolor=colors[0], linewidth=1, edgecolor='black')
rect_gucci = plt.Rectangle((0, lv_height), lv_width, gucci_height, facecolor=colors[1], linewidth=1, edgecolor='black')
ax.add_patch(rect_lv)
ax.add_patch(rect_gucci)

# Right column
rect_chanel = plt.Rectangle((right_x, 0), right_width, chanel_height, facecolor=colors[2], linewidth=1, edgecolor='black')
rect_hermes = plt.Rectangle((right_x, chanel_height), right_width, hermes_height, facecolor=colors[3], linewidth=1, edgecolor='black')
rect_rolex = plt.Rectangle((right_x, chanel_height + hermes_height), rolex_width, top_height, facecolor=colors[4], linewidth=1, edgecolor='black')
rect_others = plt.Rectangle((right_x + rolex_width, chanel_height + hermes_height), others_width, top_height, facecolor=colors[5], linewidth=1, edgecolor='black')
ax.add_patch(rect_chanel)
ax.add_patch(rect_hermes)
ax.add_patch(rect_rolex)
ax.add_patch(rect_others)

# Add text inside each rectangle
text_props = {'ha': 'center', 'va': 'center', 'fontsize': 18, 'color': 'black', 'fontweight': 'normal'}

# Left column
ax.text(lv_width / 2, lv_height / 2, f"{brands[0]}\n{shares[0]:.2f}%", **text_props)
ax.text(lv_width / 2, lv_height + gucci_height / 2, f"{brands[1]}\n{shares[1]:.2f}%", **text_props)

# Right column
ax.text(right_x + right_width / 2, chanel_height / 2, f"{brands[2]}\n{shares[2]:.2f}%", **text_props)
ax.text(right_x + right_width / 2, chanel_height + hermes_height / 2, f"{brands[3]}\n{shares[3]:.2f}%", **text_props)
ax.text(right_x + rolex_width / 2, chanel_height + hermes_height + top_height / 2, f"{brands[4]}\n{shares[4]:.2f}%", **text_props)
ax.text(right_x + rolex_width + others_width / 2, chanel_height + hermes_height + top_height / 2, f"{brands[5]}\n{shares[5]:.2f}%", **text_props)

# Add title
plt.title("Market Share of Leading Luxury Brands", fontsize=22)

# Save or show
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/tree_13.png", bbox_inches="tight")
