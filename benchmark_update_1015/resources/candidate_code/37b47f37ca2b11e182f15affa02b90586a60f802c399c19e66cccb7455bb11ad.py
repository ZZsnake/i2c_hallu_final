import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Data for the treemap
categories = [
    {"name": "Graphic Design", "percent": 40, "color": "yellow"},
    {"name": "Interior Design", "percent": 25, "color": "coral"},
    {"name": "Web Design", "percent": 15, "color": "skyblue"},
    {"name": "Fashion Design", "percent": 10, "color": "limegreen"},
    {"name": "Product Design", "percent": 5, "color": "pink"},
    {"name": "Illustration", "percent": 5, "color": "orchid"},
]

# Total width and height for the treemap
total_width = 10
total_height = 8

# Calculate total area
total_area = total_width * total_height

# Define layout: left side (two large blocks), right side (four smaller blocks stacked)
# Left side: Graphic Design (40%) and Interior Design (25%)
# Right side: Product Design (5%), Illustration (5%), Fashion Design (10%), Web Design (15%)

# Left column: Graphic Design (bottom) and Interior Design (top)
left_width = total_width * 0.6  # 60% of width for left column
right_width = total_width * 0.4  # 40% for right column

# Left column heights
graphic_height = total_height * 0.4  # 40% of total height
interior_height = total_height * 0.25  # 25% of total height

# Right column: total height is 100%, divided into 4 parts
right_total_height = total_height
right_parts = [5, 5, 10, 15]  # percentages for right column
right_heights = [h * total_height / 100 for h in right_parts]

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, total_width)
ax.set_ylim(0, total_height)
ax.set_axis_off()  # Hide axes

# Draw left column rectangles
# Graphic Design (bottom-left)
graphic_rect = patches.Rectangle(
    (0, 0), left_width, graphic_height, facecolor="yellow", edgecolor="none"
)
ax.add_patch(graphic_rect)
ax.text(
    left_width / 2,
    graphic_height / 2,
    "Graphic Design - 40%",
    ha="center",
    va="center",
    fontsize=10,
    color="black",
    fontweight="bold"
)

# Interior Design (top-left)
interior_rect = patches.Rectangle(
    (0, total_height - interior_height), left_width, interior_height, facecolor="coral", edgecolor="none"
)
ax.add_patch(interior_rect)
ax.text(
    left_width / 2,
    total_height - interior_height / 2,
    "Interior Design - 25%",
    ha="center",
    va="center",
    fontsize=10,
    color="black",
    fontweight="bold"
)

# Draw right column rectangles (stacked vertically)
right_x = total_width - right_width

# Product Design (top-right)
product_rect = patches.Rectangle(
    (right_x, total_height - right_heights[0]), right_width, right_heights[0], facecolor="pink", edgecolor="none"
)
ax.add_patch(product_rect)
ax.text(
    right_x + right_width / 2,
    total_height - right_heights[0] / 2,
    "Product Design - 5%",
    ha="center",
    va="center",
    fontsize=10,
    color="black",
    fontweight="bold"
)

# Illustration (next below Product Design)
illustration_rect = patches.Rectangle(
    (right_x, total_height - right_heights[0] - right_heights[1]), right_width, right_heights[1], facecolor="orchid", edgecolor="none"
)
ax.add_patch(illustration_rect)
ax.text(
    right_x + right_width / 2,
    total_height - right_heights[0] - right_heights[1] / 2,
    "Illustration - 5%",
    ha="center",
    va="center",
    fontsize=10,
    color="black",
    fontweight="bold"
)

# Fashion Design (next below Illustration)
fashion_rect = patches.Rectangle(
    (right_x, total_height - right_heights[0] - right_heights[1] - right_heights[2]), right_width, right_heights[2], facecolor="limegreen", edgecolor="none"
)
ax.add_patch(fashion_rect)
ax.text(
    right_x + right_width / 2,
    total_height - right_heights[0] - right_heights[1] - right_heights[2] / 2,
    "Fashion Design - 10%",
    ha="center",
    va="center",
    fontsize=10,
    color="black",
    fontweight="bold"
)

# Web Design (bottom-right)
web_rect = patches.Rectangle(
    (right_x, 0), right_width, right_heights[3], facecolor="skyblue", edgecolor="none"
)
ax.add_patch(web_rect)
ax.text(
    right_x + right_width / 2,
    right_heights[3] / 2,
    "Web Design - 15%",
    ha="center",
    va="center",
    fontsize=10,
    color="black",
    fontweight="bold"
)

# Add title
plt.title("Art and Design Categories Distribution", fontsize=16, fontweight="bold", pad=20)

# Show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/tree_15.png", bbox_inches="tight")