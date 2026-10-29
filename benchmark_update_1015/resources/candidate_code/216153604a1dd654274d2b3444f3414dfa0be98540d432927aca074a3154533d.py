import matplotlib.pyplot as plt
import numpy as np

# Dummy data based on the image
categories = ['Labeling', 'Material', 'Storage', 'Recall', 'Compliance', 'Pollution', 'Allergen', 'Purity', 'Nutrition']
MUJI_scores = [0.87, 0.68, 0.64, 0.90, 0.95, 0.86, 0.88, 0.92, 0.77]
Nestle_scores = [0.63, 0.83, 0.80, 0.80, 0.83, 0.96, 0.98, 0.68, 0.68]

# Normalize angles for radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
MUJI_scores += MUJI_scores[:1]
Nestle_scores += Nestle_scores[:1]
angles += angles[:1]

# Initialize the plot
fig, ax = plt.subplots(figsize=(7.4, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2 - np.deg2rad(10))
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["", "", "", ""], color="grey", size=7)
plt.ylim(0, 1)

# Plot data
muji_line, = ax.plot(angles, MUJI_scores, linewidth=1, linestyle='solid', label="MUJI", color='darkolivegreen', marker='s')
ax.fill(angles, MUJI_scores, color='darkolivegreen', alpha=0.2)

nestle_line, = ax.plot(angles, Nestle_scores, linewidth=1, linestyle='solid', label="Nestle", color='tomato', marker='o')
ax.fill(angles, Nestle_scores, color='tomato', alpha=0.2)

# Add legend
plt.legend(handles=[nestle_line, muji_line], loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

# Add title
plt.title("MUJI vs Nestle in Food Safety", size=14)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/radar_15.png", bbox_inches="tight")
