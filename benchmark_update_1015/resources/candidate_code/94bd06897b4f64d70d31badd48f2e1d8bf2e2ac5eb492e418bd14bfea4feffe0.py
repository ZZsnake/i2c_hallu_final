import matplotlib.pyplot as plt
import numpy as np

# Data
cities = ['Detroit', 'Philadelphia', 'Baltimore']
metrics = ['Crime Rate', 'Happiness Index', 'Social Security Coverage', 'Political Participation']
values = {
    'Detroit': [50, 70, 90, 80],
    'Philadelphia': [60, 80, 85, 75],
    'Baltimore': [40, 75, 95, 85]
}
errors = {
    'Detroit': [5, 5, 5, 5],
    'Philadelphia': [5, 5, 5, 5],
    'Baltimore': [5, 5, 5, 5]
}
colors = ['purple', 'blue', 'red', 'yellow']

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True, sharey=True)

for i, city in enumerate(cities):
    axs[i].bar(metrics, values[city], yerr=errors[city], color=colors, capsize=5)
    axs[i].set_title(city)
    axs[i].set_ylim(30, 100)

# Common labels
fig.text(0.5, 0.04, 'Metrics', ha='center')
fig.text(0.04, 0.5, 'Percentage', va='center', rotation='vertical')

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/errorbar_19.png", bbox_inches="tight")