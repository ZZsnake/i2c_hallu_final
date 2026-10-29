import matplotlib.pyplot as plt
import numpy as np

# Data
appliances = ['Refrigerator', 'Washing Machine', 'Dishwasher']
index = np.arange(len(appliances))
bar_width = 0.25

standard = [0.15, -0.10, 0.05]
advanced = [0.25, -0.05, 0.15]
optimal = [0.30, 0.05, 0.20]

# Standard deviations (for error bars)
std_standard = [0.02, 0.03, 0.02]
std_advanced = [0.02, 0.03, 0.02]
std_optimal = [0.02, 0.03, 0.02]

# Colors
color_standard = 'lightcoral'
color_advanced = 'lightgreen'
color_optimal = 'lightblue'

fig, ax = plt.subplots(figsize=(8, 5))

# Positions
x_standard = index - bar_width
x_advanced = index
x_optimal = index + bar_width

# Plot bars with error bars and hatching
bars_standard = ax.bar(
    x_standard, standard, bar_width,
    yerr=std_standard, capsize=5,
    color=color_standard, edgecolor='black', linewidth=0,
    hatch='//', label='Standard Model'
)

bars_advanced = ax.bar(
    x_advanced, advanced, bar_width,
    yerr=std_advanced, capsize=5,
    color=color_advanced, edgecolor='black', linewidth=0,
    hatch='\\', label='Advanced Model'
)

bars_optimal = ax.bar(
    x_optimal, optimal, bar_width,
    yerr=std_optimal, capsize=5,
    color=color_optimal, edgecolor='black', linewidth=0,
    hatch='--', label='Optimal Model'
)

# Title and labels
ax.set_title('Energy Efficiency Ratings by Appliance Model')
ax.set_xlabel('')
ax.set_ylabel('Energy Efficiency')

# X-ticks
ax.set_xticks(index)
ax.set_xticklabels(appliances)

# Add value labels on each bar
def add_labels(bars, errors):
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        if height >= 0:
            y = height + err + 0.005
            va = 'bottom'
        else:
            y = height - err - 0.005
            va = 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f'{height:.2f}',
            ha='center',
            va=va
        )

add_labels(bars_standard, std_standard)
add_labels(bars_advanced, std_advanced)
add_labels(bars_optimal, std_optimal)

# 修复：y=0 黑色基准线，严格贴合边框、无溢出、语法合法
ax.axhline(y=0, color='gray', linestyle='-')

# Legend
ax.legend(loc='upper right')

# Adjust y-limits to encompass labels and error bars
ax.set_ylim(-0.15, 0.35)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/errorbar_22.png", bbox_inches="tight")