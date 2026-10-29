# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import squarify

# ===================
# Part 2: Data Preparation
# ===================
# New data for cloud service provider usage
sizes = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
labels = [
    "Tesla\n35%",
    "BYD\n25%",
    "Volkswagen\n15%",
    "Nio\n10%",
    "Ford\n10%",
    "Others\n5%",
]
title = "Car Market Share"

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
colors = ["#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087", "#f95d6a"]
# Create a figure with the specified size
fig = plt.figure(figsize=(14, 10))

# Create a treemap
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    text_kwargs={"fontsize": 20, "color": "white"},
    pad=0.3,
)

# Add a title
plt.title(title, fontsize=24)

# Remove axes
plt.axis("off")

# ===================
# Part 4: Saving Output
# ===================
# Displaying the plot with tight layout to minimize white space
plt.tight_layout()
# Save the plot
plt.savefig("tree_8.pdf", bbox_inches="tight")