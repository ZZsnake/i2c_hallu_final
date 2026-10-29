import matplotlib.pyplot as plt

# Data
labels = ['Tech A 40%', 'Tech B 20%', 'Tech C 25%', 'Tech D 15%']
sizes = [40, 20, 25, 15]
colors = ['#FF6347', '#4682B4', '#3CB371', '#FFA500']  # Red, Blue, Green, Orange
explode = (0, 0, 0, 0)  # keep all slices contiguous

# Plot
fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
    shadow=True, startangle=140, pctdistance=0.8,
    wedgeprops=dict(width=0.3)
)

# Add hatch patterns to the donut slices
hatches = ['/', '\\', '|', '-']
for wedge, hatch in zip(wedges, hatches):
    wedge.set_hatch(hatch)

# Draw a circle at the center to make it a donut chart
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')  
plt.title('Technology Adoption Distribution')
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/pie_18.png")
