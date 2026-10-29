import matplotlib.pyplot as plt
import numpy as np

# Data
appliances = ['Refrigerator', 'Washing Machine', 'Dishwasher']
x = np.arange(len(appliances))
width = 0.25

standard = [0.15, -0.10, 0.05]
advanced = [0.25, -0.05, 0.15]
optimal  = [0.30,  0.05, 0.20]

# Error bars
std_err = [0.01, 0.015, 0.015]
adv_err = [0.015, 0.015, 0.015]
opt_err = [0.02, 0.02, 0.02]

fig, ax = plt.subplots(figsize=(8, 5))

# Bars
bars1 = ax.bar(x - width, standard, width,
               yerr=std_err, capsize=5,
               label='Standard Model',
               color='#ff9999', edgecolor='black', hatch='//')
bars2 = ax.bar(x, advanced, width,
               yerr=adv_err, capsize=5,
               label='Advanced Model',
               color='#99ff99', edgecolor='black', hatch='\\')
bars3 = ax.bar(x + width, optimal, width,
               yerr=opt_err, capsize=5,
               label='Optimal Model',
               color='#99ccff', edgecolor='black', hatch='--')

# Title and labels
ax.set_title('Energy Efficiency Ratings by Appliance Model')
ax.set_xlabel('')
ax.set_ylabel('Energy Efficiency')

ax.set_xticks(x)
ax.set_xticklabels(appliances)

# Y-limits similar to figure
ax.set_ylim(-0.15, 0.35)

# Legend
ax.legend(loc='upper right', frameon=True)

# Annotate bars with values
def annotate_bars(bars, values):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        # For negative bars, place text slightly below the top; for positive, above
        if height >= 0:
            y = height + 0.01 + 0.02  # above error bar and cap
        else:
            y = height - 0.02 - 0.03
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f'{val:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top')

annotate_bars(bars1, standard)
annotate_bars(bars2, advanced)
annotate_bars(bars3, optimal)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/errorbar_22.png", bbox_inches="tight")