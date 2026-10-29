import matplotlib.pyplot as plt
import numpy as np

# Data (approximate to the image)
years = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2015])

africa   = np.array([230, 280, 370, 480, 630, 810, 1030, 1280])
americas = np.array([900, 620, 560, 700, 850, 460, 520, 620])
asia     = np.array([1400, 1700, 2100, 1800, 1600, 1700, 2200, 2600])
europe   = np.array([300, 320, 340, 350, 360, 380, 390, 410])
oceania  = np.array([120, 180, 310, 290, 240, 260, 310, 320])

data = np.vstack([africa, americas, asia, europe, oceania])

colors = [
    '#c4e8b6',  # africa - light green
    '#ffe9a3',  # americas - light yellow
    '#f39987',  # asia - salmon/red
    '#f6b993',  # europe - light orange
    '#c6c6f3',  # oceania - light lavender/blue
]

labels = ['africa', 'americas', 'asia', 'europe', 'oceania']

plt.figure(figsize=(8, 6))

# Stacked area chart
plt.stackplot(years, data, labels=labels, colors=colors, alpha=0.7)

# Title and labels
plt.title('World population')
plt.xlabel('Year')
plt.ylabel('Number of people (millions)')

# Legend at top center
plt.legend(loc='upper center', ncol=5, framealpha=0.9)

# Adjust limits to resemble the original
plt.xlim(years[0], years[-1])
plt.ylim(0, 6000)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/area_5.png", bbox_inches="tight")