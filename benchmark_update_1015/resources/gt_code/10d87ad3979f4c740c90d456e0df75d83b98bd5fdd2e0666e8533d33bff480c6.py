# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import squarify

# ===================
# Part 2: Data Preparation
# ===================
# Data
sizes = [38.27, 24.36, 18.04, 9.12, 6.84, 3.37]
labels = [
    "Louis Vuitton\n38.27%",
    "Gucci\n24.36%",
    "Chanel\n18.04%",
    "Hermès\n9.12%",
    "Rolex\n6.84%",
    "Others\n3.37%",
]
title = 'Market Share of Leading Luxury Brands'

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
colors = ["#1F77B4", "#AEC7E8", "#FF7F0E", "#FFBB78", "#2CA02C", "#98DF8A"]
# Create a figure with the specified size
fig = plt.figure(figsize=(12, 8))

# Create a treemap
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    alpha=0.7,
    text_kwargs={"fontsize": 18},
    ec="black",
)

# Remove axes
plt.axis("off")

# Add title
plt.title(title, fontsize=22)

# ===================
# Part 4: Saving Output
# ===================
# Show plot with tight layout
plt.tight_layout()
plt.savefig('tree_13.pdf', bbox_inches='tight')