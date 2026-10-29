import matplotlib.pyplot as plt
import numpy as np

# Data (years and approximate regional populations in millions)
years = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2015])

africa   = np.array([230, 280, 360, 470, 630, 820, 1040, 1270])
americas = np.array([950, 610, 550, 670, 820, 500, 540, 600])
asia     = np.array([1400, 1700, 2100, 1800, 1200, 1700, 2100, 2600])
europe   = np.array([280, 310, 320, 330, 340, 350, 360, 370])
oceania  = np.array([180, 220, 260, 230, 210, 230, 260, 280])

regions = [africa, americas, asia, europe, oceania]

# Colors chosen to visually match the reference
colors = [
    "#b6e3a8",  # africa - light green
    "#ffe3a3",  # americas - light yellow
    "#f28e8c",  # asia - salmon
    "#f6c49f",  # europe - light orange
    "#c4c8ff",  # oceania - light lavender/blue
]

labels = ["africa", "americas", "asia", "europe", "oceania"]

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Stacked area chart
ax.stackplot(years, regions, labels=labels, colors=colors, alpha=0.7)

# Title and labels
ax.set_title("World population")
ax.set_xlabel("Year")
ax.set_ylabel("Number of people (millions)")

# Y-axis limit to reach 6000 as in the image
ax.set_ylim(0, 6000)

# X-axis ticks
ax.set_xticks(years)

# Legend at top center with semi-transparent white background
legend = ax.legend(
    loc="upper center",
    ncol=5,
    framealpha=0.8,
)

legend.get_frame().set_facecolor("white")

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/area_5.png", bbox_inches="tight")