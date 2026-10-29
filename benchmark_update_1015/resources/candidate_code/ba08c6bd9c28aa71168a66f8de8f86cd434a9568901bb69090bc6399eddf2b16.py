import numpy as np
import matplotlib.pyplot as plt

# Define the categories and the performance metrics for each company
categories = ['Storage', 'Material', 'Labeling', 'Nutrition', 'Purity', 'Allergen', 'Pollution', 'Compliance', 'Recall']
nestle_performance = [80, 85, 65, 70, 68, 98, 97, 83, 80]
muji_performance = [65, 68, 88, 78, 92, 90, 88, 95, 92]

# Number of variables
num_vars = len(categories)

# Compute angle for each category
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

# The radar chart is a circular plot
values_nestle = nestle_performance + nestle_performance[:1]
values_muji = muji_performance + muji_performance[:1]

# Initialize the radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Draw one line per company
ax.fill(angles, values_nestle, color='#ff6347', alpha=0.25)
ax.fill(angles, values_muji, color='#556b2f', alpha=0.25)

# Draw the outline for each company
ax.plot(angles, values_nestle, color='#ff6347', linewidth=2, label='Nestle', marker='o')
ax.plot(angles, values_muji, color='#556b2f', linewidth=2, label='MUJI', marker='s')

# Fill the center area
ax.fill_between(angles, values_nestle, values_muji, color='gray', alpha=0.2)

# Labels for each category
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80])
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Add a title
plt.title('MUJI vs Nestle in Food Safety', size=15, color='black', y=1.1)

# Add a legend
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/radar_15.png", bbox_inches="tight")
