import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ["Crime Rate", "Happiness Index",
              "Social Security Coverage", "Political Participation"]
x = np.arange(len(categories))

# Values and error bars (approximate from the figure)
detroit_vals = [50, 70, 90, 80]
detroit_errs = [5, 6, 8, 7]

philly_vals = [60, 80, 85, 75]
philly_errs = [5, 6, 7, 6]

balt_vals = [40, 75, 95, 85]
balt_errs = [6, 7, 6, 6]

# Colors (approximate)
colors = {
    "Crime Rate": "#9b59b6",          # purple
    "Happiness Index": "#3498db",     # blue
    "Social Security Coverage": "#e74c3c",  # red
    "Political Participation": "#f1c40f"    # yellow
}

# Create figure and axes
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# Common y-axis settings
for ax in axes:
    ax.set_ylim(30, 100)
    ax.set_ylabel("Percentage")
    ax.set_yticks(np.arange(30, 101, 10))
    ax.grid(axis='y', linestyle='-', color='0.8', alpha=0.7)
    ax.set_axisbelow(True)

# Helper to plot one city
def plot_city(ax, values, errs):
    bars = []
    for i, (cat, val, err) in enumerate(zip(categories, values, errs)):
        bar = ax.bar(
            x[i],
            val,
            yerr=err,
            color=colors[cat],
            capsize=5,
            ecolor='black',
            edgecolor=colors[cat]
        )
        bars.append(bar[0])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0)
    return bars

# Plot each city
bars_detroit = plot_city(axes[0], detroit_vals, detroit_errs)
bars_philly = plot_city(axes[1], philly_vals, philly_errs)
bars_balt = plot_city(axes[2], balt_vals, balt_errs)

# Subplot labels under each axes
axes[0].set_title("(a) Detroit", y=-0.25)
axes[1].set_title("(b) Philadelphia", y=-0.25)
axes[2].set_title("(c) Baltimore", y=-0.25)

# Build legend using handles from first subplot
legend_labels = ["Crime Rate", "Happiness Index",
                 "Social Security Coverage", "Political Participation"]
legend_handles = [
    bars_detroit[0], bars_detroit[1], bars_detroit[2], bars_detroit[3]
]

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.98)
)

plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/errorbar_19.png", bbox_inches="tight")