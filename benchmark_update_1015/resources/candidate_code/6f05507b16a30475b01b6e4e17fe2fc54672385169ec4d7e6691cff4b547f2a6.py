import matplotlib.pyplot as plt
import numpy as np

# Data
appliances = ['Refrigerator', 'Washing Machine', 'Dishwasher']
standard = [0.15, -0.10, 0.05]
advanced = [0.25, -0.05, 0.15]
optimal = [0.30, 0.05, 0.20]
error = [[0.02, 0.03, 0.02], [0.02, 0.03, 0.02], [0.02, 0.03, 0.02]]

x = np.arange(len(appliances))  # the label locations
width = 0.25  # the width of the bars

# Plotting the data
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width, standard, width, label='Standard Model', color='lightcoral', hatch='//')
bars2 = ax.bar(x, advanced, width, label='Advanced Model', color='lightgreen', hatch='\\')
bars3 = ax.bar(x + width, optimal, width, label='Optimal Model', color='lightblue', hatch='--')

# Adding error bars
ax.errorbar(x - width, standard, yerr=error[0], fmt='none', capsize=5, color='black')
ax.errorbar(x, advanced, yerr=error[1], fmt='none', capsize=5, color='black')
ax.errorbar(x + width, optimal, yerr=error[2], fmt='none', capsize=5, color='black')

# Adding labels
ax.set_ylabel('Energy Efficiency')
ax.set_title('Energy Efficiency Ratings by Appliance Model')
ax.set_xticks(x)
ax.set_xticklabels(appliances)
ax.axhline(0, color='gray', linewidth=1)
ax.legend()

# Annotating data points
for bars, errs in zip([bars1, bars2, bars3], error):
    for bar, err in zip(bars, errs):
        height = bar.get_height()
        if height >= 0:
            y = height + err
            offset = 3
            va = 'bottom'
        else:
            y = height - err
            offset = -3
            va = 'top'
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha='center', va=va)
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/errorbar_22.png", bbox_inches="tight")
