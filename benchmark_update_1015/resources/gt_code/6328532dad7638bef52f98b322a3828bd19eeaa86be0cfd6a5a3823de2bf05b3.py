# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import squarify

# ===================
# Part 2: Data Preparation
# ===================
# Data
sizes = [30, 25, 20, 10, 8, 7]
labels = [
    "Cargill\n30%",
    "Wilmar International\n25%",
    "ADM\n20%",
    "Bunge\n10%",
    "COFCO\n8%",
    "Others\n7%",
]
title = "Market Share of Edible Oil Brands"
# ===================
# Part 3: Plot Configuration and Rendering
# ===================
colors = ["#2E8B57", "#FF6347", "#4682B4", "#DAA520", "#8A2BE2", "#FF69B4"]
# Create a figure with the specified size
fig = plt.figure(figsize=(10, 8))

# Create a treemap
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    alpha=0.8,
    text_kwargs={"fontsize": 14, "color": "white", "weight": "bold"},
    pad=True,
    ec="black",
)

# Remove axes
plt.axis("off")

# Add title
plt.title(title, fontsize=16, weight="bold")

# ===================
# Part 4: Saving Output
# ===================
# Show plot with tight layout and save to file
plt.tight_layout()
plt.savefig('tree_20.pdf', bbox_inches='tight')