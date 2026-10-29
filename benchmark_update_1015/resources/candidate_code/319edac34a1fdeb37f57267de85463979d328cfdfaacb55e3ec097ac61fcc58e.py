# ===================
# Part 1: Importing Libraries
# ===================
import numpy as np; np.random.seed(0)

import matplotlib.pyplot as plt
from math import pi

# ===================
# Part 2: Data Preparation
# ===================
# Define the data for each line
axis_labels = np.array(
    [
        "line",
        "heatmap",
        "line_num",
        "candlestick",
        "3D-bar",
        "rose",
        "multi-axes",
        "bubble",
        "radar",
        "area",
        "pie",
        "funnel",
        "histogram",
        "bar_num",
        "box",
        "treemap",
    ]
)
axis_count = len(axis_labels)

series1 = np.array([3, 3.5, 4.2, 4, 4.8, 5, 4, 3.2, 4, 4.3, 3.6, 3.8, 4.4, 4.2, 3.6, 4])
series2 = np.array(
    [3.5, 3.2, 3.6, 3.5, 3.7, 3.1, 3.4, 4, 4.5, 3.7, 4, 2.8, 3, 4, 2.9, 3]
)
series3 = np.array(
    [2, 2.4, 2.3, 2.4, 2.5, 2.4, 2.3, 2.2, 1.4, 2.5, 1.3, 1.4, 2.2, 2.3, 1.4, 1.5]
)
series_labels=["QWen-VL", "SPHINX-V2", "ChartLlama"]
yticks=[1, 2, 3, 4, 5]
ytickslabel=["1", "2", "3", "4", "5"]
ylim=[0, 5]


def close_series(series_values):
    return np.concatenate((series_values, [series_values[0]]))

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Compute angle for each axis
angles = [n / float(axis_count) * 2 * pi for n in range(axis_count)]
series1 = close_series(series1)
series2 = close_series(series2)
series3 = close_series(series3)
angles += angles[:1]

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], axis_labels)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks(yticks, ytickslabel, color="black", size=7)
plt.ylim(ylim)

# Plot data
ax.plot(
    angles,
    series1,
    linewidth=1,
    linestyle="solid",
    label=series_labels[0],
    color="#971d2b",
    marker="o",
)
ax.fill(angles, series1, "#971d2b", alpha=0.1)

ax.plot(
    angles,
    series2,
    linewidth=1,
    linestyle="dashed",
    label=series_labels[1],
    color="#6f98c3",
    marker="s",
)
ax.fill(angles, series2, "#6f98c3", alpha=0.1)

ax.plot(
    angles,
    series3,
    linewidth=1,
    linestyle="dotted",
    label=series_labels[2],
    color="#f4c17d",
    marker="D",
)
ax.fill(angles, series3, "#f4c17d", alpha=0.1)

# Add legend
plt.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1))

# ===================
# Part 4: Saving Output
# ===================
# Adjust layout for better fit and save the plot
plt.tight_layout()
plt.savefig('radar_7.pdf', bbox_inches='tight')
