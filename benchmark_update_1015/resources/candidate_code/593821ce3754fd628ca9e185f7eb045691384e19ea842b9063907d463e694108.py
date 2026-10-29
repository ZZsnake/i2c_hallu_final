import matplotlib.pyplot as plt
import numpy as np

# Data
appliances = ['Refrigerator', 'Washing Machine', 'Dishwasher']
standard = [0.15, -0.10, 0.05]
advanced = [0.25, -0.05, 0.15]
optimal = [0.30, 0.05, 0.20]
error = [[0.02, 0.02, 0.02], [0.02, 0.02, 0.02], [0.02, 0.02, 0.02]]

x = np.arange(len(appliances))  # the label locations
width = 0.25  # the width of the bars

# Plotting the data
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width, standard, width, label='Standard Model', color='red', hatch='//')
bars2 = ax.bar(x, advanced, width, label='Advanced Model', color='green', hatch='//')
bars3 = ax.bar(x + width, optimal, width, label='Optimal Model', color='blue', hatch='--')

# Adding error bars
ax.errorbar(x - width, standard, yerr=error[0], fmt='none', capsize=5, color='black')
ax.errorbar(x, advanced, yerr=error[1], fmt='none', capsize=5, color='black')
ax.errorbar(x + width, optimal, yerr=error[2], fmt='none', capsize=5, color='black')

# Adding labels
ax.set_xlabel('Appliance')
ax.set_ylabel('Energy Efficiency')
ax.set_title('Energy Efficiency Ratings by Appliance Model')
ax.set_xticks(x)
ax.set_xticklabels(appliances)
ax.legend()

# Annotating data points
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/errorbar_22.png", bbox_inches="tight")