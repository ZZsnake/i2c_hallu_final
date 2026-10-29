# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import numpy as np


# ===================
# Part 2: Data Preparation
# ===================
np.random.seed(0)
# Sample data to create another type of distribution contour lines
x = np.linspace(-15, 15, 150)
y = np.linspace(-15, 15, 150)
X, Y = np.meshgrid(x, y)
Z1 = np.sin(np.sqrt(X**2 + Y**2)) / np.sqrt(X**2 + Y**2)
Z2 = np.cos(np.sqrt(X**2 + Y**2)) / np.sqrt(X**2 + Y**2)
legend_labels = ["Field A", "Field B"]
xlabel = "X-axis"
ylabel = "Y-axis"
title = "Scalar Field Distribution"
cbar_label = 'Field Intensity'

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create the plot
fig, ax = plt.subplots(figsize=(12, 9))

# Contour lines for Field A (cool colors) and Field B (warm colors)
contour_field_a = ax.contour(X, Y, Z1, colors="green", linestyles='dotted', linewidths=1)
contour_field_b = ax.contour(X, Y, Z2, colors="red", linestyles='dashdot', linewidths=2)

# Labels for x and y axes
ax.set_xlabel(xlabel, fontsize=15)
ax.set_ylabel(ylabel, fontsize=15)
ax.set_title(title, fontsize=18)

# Adding a legend manually
h1, _ = contour_field_a.legend_elements()
h2, _ = contour_field_b.legend_elements()
ax.legend([h1[0], h2[0]], legend_labels, fontsize=13)

# Set the aspect of the plot
ax.set_aspect("equal")
ax.grid(True)
ax.set_facecolor("#e0e0e0")
ax.set_ylim(-15, 15)
ax.set_xlim(-15, 15)

# Add color bar to represent field intensity
field_intensity = ax.contourf(X, Y, Z1 + Z2, alpha=0.3, cmap='viridis')
cbar = fig.colorbar(field_intensity, ax=ax)
cbar.set_label(cbar_label, fontsize=13)

# ===================
# Part 4: Saving Output
# ===================
# Show the plot with tight layout to minimize white space
plt.tight_layout()
plt.savefig("contour_15.pdf", bbox_inches="tight")
