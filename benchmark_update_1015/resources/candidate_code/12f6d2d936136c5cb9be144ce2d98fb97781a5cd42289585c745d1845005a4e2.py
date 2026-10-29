import matplotlib.pyplot as plt
import numpy as np

# Data (approximate to the image)
years = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018])

africa   = np.array([230, 280, 370, 480, 630, 810, 1050, 1280])
americas = np.array([920, 620, 550, 720, 850, 440, 520, 620])
asia     = np.array([1430, 1700, 2100, 1650, 1200, 1710, 2100, 2550])
europe   = np.array([220, 230, 280, 300, 320, 300, 250, 300])
oceania  = np.array([200, 300, 350, 350, 275, 270, 430, 300])

data = np.vstack([africa, americas, asia, europe, oceania])

colors = [
    '#b8e8b2',  # africa - light green
    '#ffe6a3',  # americas - light yellow
    '#df7568',  # asia - salmon/red
    '#fb9e7f',  # europe - light orange
    '#adb6df',  # oceania - light lavender/blue
]

labels = ['africa', 'americas', 'asia', 'europe', 'oceania']

plt.figure(figsize=(8, 6))

# Stacked area chart
plt.stackplot(years, data, labels=labels, colors=colors)

# Title and labels
plt.title('World population', y=1.08)
plt.xlabel('Year')
plt.ylabel('Number of people (millions)')

# Legend at top center
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=5, frameon=False)

# Adjust limits to resemble the original
plt.xlim(1950, 2018)
plt.ylim(0, 6000)
plt.tick_params(axis="both", which="both", length=0)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/area_5.png", bbox_inches="tight")
