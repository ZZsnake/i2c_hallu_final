import matplotlib.pyplot as plt
import numpy as np

# Data (approximate to match the visual)
years = np.array([1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])

africa   = np.array([0.6, 0.7, 0.95, 1.15, 1.35, 1.55, 1.8, 2.1])
americas = np.array([1.2, 1.25, 1.28, 1.3, 1.33, 1.35, 1.38, 1.4])
asia     = np.array([3.2, 3.3, 3.4, 3.55, 3.8, 4.0, 4.2, 4.5])
europe   = np.array([0.7, 0.7, 0.69, 0.68, 0.67, 0.65, 0.63, 0.6])
oceania  = np.array([0.03, 0.035, 0.04, 0.042, 0.045, 0.047, 0.048, 0.05])

# Colors approximating the original shades
colors = [
    "#ffb6c1",  # Africa - light pink
    "#ff9ca8",  # Americas - slightly deeper pink
    "#ff7f7f",  # Asia - medium pink
    "#ff4c4c",  # Europe - stronger red
    "#ff0000",  # Oceania - deep red
]

labels = ["Africa", "Americas", "Asia", "Europe", "Oceania"]

fig, ax = plt.subplots(figsize=(10, 7))

# Stacked area chart
ax.stackplot(
    years,
    africa,
    americas,
    asia,
    europe,
    oceania,
    labels=labels,
    colors=colors,
    alpha=0.8
)

# Title and labels
ax.set_title("World Population Growth by Continent")
ax.set_xlabel("Year")
ax.set_ylabel("Population (Billions)")

# Axis limits and ticks to resemble the original
ax.set_xlim(1990, 2025)
ax.set_ylim(0, 10)
ax.set_xticks(years)

# Legend in upper right inside axes
legend = ax.legend(loc="upper right")
legend.get_frame().set_edgecolor("0.8")
legend.get_frame().set_facecolor("white")

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/area_18.png", bbox_inches="tight")