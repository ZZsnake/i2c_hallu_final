# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import squarify

# ===================
# Part 2: Data Preparation
# ===================
# Data
sizes = [40, 20, 15, 10, 8, 7]
labels = [
    "Chanel\n40%",
    "Dior\n20%",
    "Gucci\n15%",
    "Yves Saint Laurent\n10%",
    "Tom Ford\n8%",
    "Jo Malone\n7%",
]
title ='Market Share of Leading Perfume Brand'

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create a figure with the specified size
colors = ["#FFDDC1", "#FDA2A8", "#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9"]
fig = plt.figure(figsize=(12, 8))

# Create a treemap
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    alpha=0.7,
    text_kwargs={"fontsize": 18, "weight": 'bold'},
    ec="black",
)

# Add a title
plt.title(title, fontsize=22, weight='bold')

# Remove axes
plt.axis("off")

# ===================
# Part 4: Saving Output
# ===================
# Show plot with tight layout
plt.tight_layout()
plt.savefig('tree_14.pdf', bbox_inches='tight')