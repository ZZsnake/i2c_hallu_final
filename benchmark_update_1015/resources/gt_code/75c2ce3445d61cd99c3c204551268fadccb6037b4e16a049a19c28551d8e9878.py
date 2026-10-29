# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import numpy as np

# ===================
# Part 2: Data Preparation
# ===================
np.random.seed(0)
year = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]
population_by_continent = {
    "Africa": [0.6, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.1],
    "Americas": [1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55],
    "Asia": [3.2, 3.3, 3.4, 3.6, 3.8, 4.0, 4.2, 4.5],
    "Europe": [0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79],
    "Oceania": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
}

# Extracted variables
legend_labels = list(population_by_continent.keys())
xlabel_value = "Year"
ylabel_value = "Population (Billions)"
title_value = "World Population Growth by Continent"

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
xlim_values = (1990, 2025)
ylim_values = (0, 10)
legend_loc = "upper left"
legend_reverse = False
legend_frameon = True
legend_ncol = 1
legend_bbox_to_anchor = (1, 1)
title_y_position = 1.05
colors = ["#ffb3b3", "#ff9999", "#ff6666", "#ff4d4d", "#ff0000"]

fig, ax = plt.subplots(figsize=(10, 7))
ax.stackplot(
    year,
    population_by_continent.values(),
    labels=legend_labels,
    alpha=0.8,
    colors=colors,
)
ax.legend(
    loc=legend_loc,
    reverse=legend_reverse,
    frameon=legend_frameon,
    ncol=legend_ncol,
    bbox_to_anchor=legend_bbox_to_anchor,
)
ax.set_xlim(*xlim_values)
ax.set_ylim(*ylim_values)
ax.set_title(title_value, y=title_y_position)
ax.set_xlabel(xlabel_value)
ax.set_ylabel(ylabel_value)
ax.tick_params(axis="both", which="both", length=0)

# ===================
# Part 4: Saving Output
# ===================
plt.tight_layout()
plt.savefig("area_18.pdf", bbox_inches="tight")