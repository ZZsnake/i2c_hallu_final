import matplotlib.pyplot as plt
import numpy as np

# Create dummy data matching visual patterns
cities = ['Detroit', 'Philadelphia', 'Baltimore']
categories = ['Crime Rate', 'Happiness Index', 'Social Security Coverage', 'Political Participation']
colors = ['#800080', '#1f77b4', '#d62728', '#ff7f0e']
data = {
    'Detroit': [50, 70, 90, 80],
    'Philadelphia': [60, 80, 85, 75],
    'Baltimore': [40, 75, 95, 85]
}
errors = {
    'Detroit': [5, 4, 6, 5],
    'Philadelphia': [4, 5, 5, 4],
    'Baltimore': [5, 6, 7, 5]
}

# Create figure and subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
x = np.arange(len(categories))
bar_width = 0.8

# Plot each city's data
for i, (city, ax) in enumerate(zip(cities, axes)):
    ax.bar(x, data[city], yerr=errors[city], capsize=5, color=colors, 
           edgecolor='black', width=bar_width)
    ax.set_ylim(0, 100)
    ax.set_title(f'({chr(97+i)}) {city}', pad=10)
    
    # Set x-axis labels only for bottom subplot
    if i == 2:
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=0)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(['']*4)
    
    ax.set_ylabel('Percentage')

# Create legend
handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(4)]
labels = categories
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.95))

# Adjust layout and display
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/errorbar_19.png", bbox_inches="tight")