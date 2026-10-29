import matplotlib.pyplot as plt

# Data
labels = ['Tech A', 'Tech B', 'Tech C', 'Tech D']
sizes = [40, 20, 25, 15]
colors = ['#FF6347', '#1E90FF', '#32CD32', '#FFA500']  # Red, Blue, Green, Orange
explode = (0.1, 0, 0, 0)  # explode the 1st slice (Tech A)

# Plot
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=140, wedgeprops=dict(width=0.3))

# Draw a circle at the center to make it a donut chart
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')  
plt.title('Technology Adoption Distribution')
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/pie_18.png", bbox_inches="tight")