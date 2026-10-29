import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Tech A 40%', 'Tech B 20%', 'Tech C 25%', 'Tech D 15%']
sizes = [40, 20, 25, 15]
colors = ['#FF6347', '#4682B4', '#3CB371', '#FFA500']  # Red-orange, Steel blue, Green, Orange
hatches = ['/', '\\', '|', '-']

# Create figure with specified size
plt.figure(figsize=(7, 7))

# Create donut chart
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=144,
    counterclock=True,
    shadow=True,
    wedgeprops=dict(width=0.3, edgecolor='black'),
    textprops={'fontsize': 10}
)

# Apply per-slice hatch patterns
for wedge, hatch in zip(wedges, hatches):
    wedge.set_hatch(hatch)

# Customize the autopct labels (internal percentages)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(10)

# Add title
plt.title('Technology Adoption Distribution', fontsize=14, pad=20)

# Adjust layout to prevent clipping
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/pie_18.png", bbox_inches="tight")
