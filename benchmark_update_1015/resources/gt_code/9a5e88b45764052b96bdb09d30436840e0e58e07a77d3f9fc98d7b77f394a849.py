# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt

# ===================
# Part 2: Data Preparation
# ===================
labels = ["Tech A 40%", "Tech B 20%", "Tech C 25%", "Tech D 15%"]
sizes = [40, 20, 25, 15]
title = "Technology Adoption Distribution"

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
autopct_format = '%1.1f%%'
startangle = 140
pctdistance = 0.85
centre_circle_radius = 0.70
fig, ax = plt.subplots(figsize=(7, 7))
colors = ['#FF6347', '#4682B4', '#3CB371', '#FFA500']  # Tomato, Steel Blue, Medium Sea Green, Orange
hatches = ['/', '\\', '|', '-']
ax.pie(sizes, labels=labels, colors=colors, autopct=autopct_format, startangle=startangle, 
       hatch=hatches, shadow=True, pctdistance=pctdistance)
plt.title(title)

# Draw center circle for a donut effect
centre_circle = plt.Circle((0, 0), centre_circle_radius, fc='white')
fig.gca().add_artist(centre_circle)

# Ensure that pie is drawn as a circle.
ax.axis('equal')  

# ===================
# Part 4: Saving Output
# ===================
plt.tight_layout()
plt.savefig('pie_18.pdf', bbox_inches='tight')