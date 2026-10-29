# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import squarify

# ===================
# Part 2: Data Preparation
# ===================
# Data - Art and Design Categories
sizes = [40, 25, 15, 10, 5, 5]
labels = ["Graphic Design - 40%", "Interior Design - 25%", "Web Design - 15%", "Fashion Design - 10%", "Product Design - 5%", "Illustration - 5%"]
title="Art and Design Categories Distribution"


# ===================
# Part 3: Plot Configuration and Rendering
# ===================
colors = ["#FFD700", "#FF6347", "#4682B4", "#32CD32", "#FF69B4", "#BA55D3"]
# Create a figure with the specified size
fig = plt.figure(figsize=(10, 8))

# Create a treemap
squarify.plot(
    sizes=sizes,
    label=labels,
    color=colors,
    alpha=0.8,
    text_kwargs={"fontsize": 12, "color": "black"},
    edgecolor="white",  # Improved border visibility
    linewidth=2 # Added edge line width for better visuals
)

# Set plot title
plt.title(title, fontsize=18, color='black', weight='bold')

# Remove axes
plt.axis("off")

# ===================
# Part 4: Saving Output
# ===================
# Show plot with tight layout and save to file
plt.tight_layout()
plt.savefig('tree_15.pdf', bbox_inches='tight')