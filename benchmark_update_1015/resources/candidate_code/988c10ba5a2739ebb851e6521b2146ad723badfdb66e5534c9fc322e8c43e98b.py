import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams["font.family"] = "sans-serif"
# Data for the treemap
categories = [
    {"name": "Graphic Design", "percent": 40, "color": "#ffde2e"},
    {"name": "Interior Design", "percent": 25, "color": "#ff7b6b"},
    {"name": "Web Design", "percent": 15, "color": "#6d9ec3"},
    {"name": "Fashion Design", "percent": 10, "color": "#56d653"},
    {"name": "Product Design", "percent": 5, "color": "#ee75b6"},
    {"name": "Illustration", "percent": 5, "color": "#bf6bd4"},
]

# Total width and height for the treemap
total_width = 10
total_height = 8

# Calculate total area
total_area = total_width * total_height

# Define layout: left side (two large blocks), right side (four smaller blocks)
# Left side: Graphic Design (40%) and Interior Design (25%)
# Right side: Product Design (5%) and Illustration (5%) on top, Fashion Design (10%), Web Design (15%)

# Left column occupies the combined 65% area of Graphic and Interior Design
left_width = total_width * 0.65
right_width = total_width - left_width

# Heights are derived from true treemap areas so the full plotting area is occupied
graphic_height = (total_area * 0.40) / left_width
interior_height = (total_area * 0.25) / left_width

# Right column heights derived from true areas
web_height = (total_area * 0.15) / right_width
fashion_height = (total_area * 0.10) / right_width
top_height = (total_area * (0.05 + 0.05)) / right_width
product_width = (total_area * 0.05) / top_height
illustration_width = (total_area * 0.05) / top_height

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, total_width)
ax.set_ylim(0, total_height)
ax.set_axis_off()  # Hide axes

border_style = {"edgecolor": "white", "linewidth": 2}

# Draw left column rectangles
# Graphic Design (bottom-left)
graphic_rect = patches.Rectangle(
    (0, 0), left_width, graphic_height, facecolor="#ffde2e", **border_style
)
ax.add_patch(graphic_rect)
ax.text(
    left_width / 2,
    graphic_height / 2,
    "Graphic Design - 40%",
    ha="center",
    va="center",
    fontsize=12,
    color="black"
)

# Interior Design (top-left)
interior_rect = patches.Rectangle(
    (0, total_height - interior_height), left_width, interior_height, facecolor="#ff7b6b", **border_style
)
ax.add_patch(interior_rect)
ax.text(
    left_width / 2,
    total_height - interior_height / 2,
    "Interior Design - 25%",
    ha="center",
    va="center",
    fontsize=12,
    color="black"
)

# Draw right column rectangles
right_x = total_width - right_width

# Product Design (top-right, left half of top row)
product_rect = patches.Rectangle(
    (right_x, total_height - top_height), product_width, top_height, facecolor="#ee75b6", **border_style
)
ax.add_patch(product_rect)
ax.text(
    right_x + product_width / 2,
    total_height - top_height / 2,
    "Product Design - 5%",
    ha="center",
    va="center",
    fontsize=12,
    color="black"
)

# Illustration (top-right, right half of top row)
illustration_rect = patches.Rectangle(
    (right_x + product_width, total_height - top_height), illustration_width, top_height, facecolor="#bf6bd4", **border_style
)
ax.add_patch(illustration_rect)
ax.text(
    right_x + product_width + illustration_width / 2,
    total_height - top_height / 2,
    "Illustration - 5%",
    ha="center",
    va="center",
    fontsize=12,
    color="black"
)

# Fashion Design (middle-right)
fashion_rect = patches.Rectangle(
    (right_x, web_height), right_width, fashion_height, facecolor="#56d653", **border_style
)
ax.add_patch(fashion_rect)
ax.text(
    right_x + right_width / 2,
    web_height + fashion_height / 2,
    "Fashion Design - 10%",
    ha="center",
    va="center",
    fontsize=12,
    color="black"
)

# Web Design (bottom-right)
web_rect = patches.Rectangle(
    (right_x, 0), right_width, web_height, facecolor="#6d9ec3", **border_style
)
ax.add_patch(web_rect)
ax.text(
    right_x + right_width / 2,
    web_height / 2,
    "Web Design - 15%",
    ha="center",
    va="center",
    fontsize=12,
    color="black"
)

# Add title
plt.title("Art and Design Categories Distribution", fontsize=16, fontweight="bold", pad=6)

# Show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/tree_15.png", bbox_inches="tight")
