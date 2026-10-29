import matplotlib.pyplot as plt
import numpy as np

# Dummy data based on the image
categories = ['Labeling', 'Material', 'Storage', 'Recall', 'Compliance', 'Pollution', 'Allergen', 'Purity', 'Nutrition']
MUJI_scores = [0.9, 0.85, 0.8, 0.75, 0.9, 0.85, 0.7, 0.95, 0.8]
Nestle_scores = [0.8, 0.75, 0.7, 0.65, 0.8, 0.75, 0.75, 0.85, 0.7]

# Normalize angles for radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
MUJI_scores += MUJI_scores[:1]
Nestle_scores += Nestle_scores[:1]
angles += angles[:1]

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
plt.ylim(0, 1)

# Plot data
ax.plot(angles, MUJI_scores, linewidth=1, linestyle='solid', label="MUJI")
ax.fill(angles, MUJI_scores, 'b', alpha=0.1)

ax.plot(angles, Nestle_scores, linewidth=1, linestyle='solid', label="Nestle", color='r')
ax.fill(angles, Nestle_scores, 'r', alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Add title
plt.title("MUJI vs Nestle in Food Safety")

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/radar_15.png", bbox_inches="tight")