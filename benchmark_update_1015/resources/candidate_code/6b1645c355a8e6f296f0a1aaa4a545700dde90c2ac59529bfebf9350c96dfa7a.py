import numpy as np
import matplotlib.pyplot as plt

# Data and categories
categories = [
    "Storage", "Material", "Labeling", "Nutrition",
    "Purity", "Allergen", "Pollution", "Compliance", "Recall"
]

# Approximate scores (0-10 scale)
nestle_values = [8, 8.5, 6, 6.5, 6, 9.5, 9.5, 8, 8]
muji_values   = [6.5, 7, 8.7, 7.5, 9, 9, 8.7, 9.3, 9]

# Close the polygons by repeating first value at end
nestle = np.array(nestle_values + [nestle_values[0]])
muji = np.array(muji_values + [muji_values[0]])

# Angles for each axis
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close loop

# Create figure
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Adjust starting angle and direction to roughly match layout
ax.set_theta_offset(np.pi / 2)     # start at top
ax.set_theta_direction(-1)         # go clockwise

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)

# Set radial limits and ticks
ax.set_rlabel_position(0)
radial_ticks = [2, 4, 6, 8, 10]
plt.yticks(radial_ticks, [str(t) for t in radial_ticks], color="grey", size=9)
plt.ylim(0, 10)

# Grid style
ax.grid(color='lightgrey', linestyle='-', linewidth=0.7)

# Colors
nestle_color = '#ff7f50'  # coral-like
muji_color = '#6b8e23'    # olive green

# Plot Nestle
ax.plot(angles, nestle, color=nestle_color, linewidth=2, linestyle='-', marker='o',
        markersize=4, label='Nestle')
ax.fill(angles, nestle, color=nestle_color, alpha=0.25)

# Plot MUJI
ax.plot(angles, muji, color=muji_color, linewidth=2, linestyle='-', marker='s',
        markersize=4, label='MUJI')
ax.fill(angles, muji, color=muji_color, alpha=0.25)

# Title
plt.title("MUJI vs Nestle in Food Safety", fontsize=16, pad=20)

# Legend at bottom center
legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15),
                    ncol=2, frameon=False)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/radar_15.png", bbox_inches="tight")