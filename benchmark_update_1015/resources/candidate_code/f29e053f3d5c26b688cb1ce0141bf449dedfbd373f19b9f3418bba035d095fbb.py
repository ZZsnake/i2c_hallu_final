# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


# ===================
# Part 2: Data Preparation
# ===================
# Data representing social statistics for three different cities
city_names = ["Detroit", "Philadelphia", "Baltimore"]
metric_names = [
    "Crime Rate",
    "Happiness Index",
    "Social Security Coverage",
    "Political Participation",
]
performance_values = np.array(
    [
        [50, 70, 90, 80],
        [60, 80, 85, 75],
        [40, 75, 95, 85],
    ]
)
error_values = np.array(
    [
        [5, 6, 9, 6],
        [6, 7, 8, 7],
        [8, 9, 6, 8],
    ]
)
y_limits = [30, 100]
y_axis_label = "Percentage"

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Figure size to match a 3x1 subplot layout
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
# Colors, choosing a different palette to differentiate the plots
colors = ["#8e44ad", "#3498db", "#e74c3c", "#f1c40f"]

# Plotting bars
for i, ax in enumerate(axes):
    for j, metric in enumerate(metric_names):
        ax.bar(
            j,
            performance_values[i, j],
            width=0.8,
            color=colors[j],
            yerr=error_values[i, j],
            capsize=0,
            label=metric if i == 0 else "",
        )

    # Setting x-axis labels, y-axis limits, and titles
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45)
    ax.set_ylim(y_limits)
    ax.set_xlabel(f"({chr(97+i)}) {city_names[i]}")
    ax.set_ylabel(y_axis_label)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

# Adding a legend outside of the plot on top
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(metric_names))

# ===================
# Part 4: Saving Output
# ===================
# Adjusting layout to prevent overlap and ensure labels are visible
plt.tight_layout()
plt.savefig("errorbar_19.pdf", bbox_inches="tight")
