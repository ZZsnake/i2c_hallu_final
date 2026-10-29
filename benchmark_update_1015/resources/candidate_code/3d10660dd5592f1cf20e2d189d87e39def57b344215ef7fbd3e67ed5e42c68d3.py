import matplotlib.pyplot as plt
import numpy as np

# Dummy data creation
data = {
    'Detroit': {'Crime Rate': 50, 'Happiness Index': 70, 'Social Security Coverage': 90, 'Political Participation': 80},
    'Philadelphia': {'Crime Rate': 60, 'Happiness Index': 80, 'Social Security Coverage': 85, 'Political Participation': 75},
    'Baltimore': {'Crime Rate': 40, 'Happiness Index': 75, 'Social Security Coverage': 95, 'Political Participation': 85}
}

categories = ['Crime Rate', 'Happiness Index', 'Social Security Coverage', 'Political Participation']
colors = ['#8A2BE2', '#00BFFF', '#FF4500', '#FFFF00']
error_bars = {
    'Detroit': [5, 6, 9, 6],
    'Philadelphia': [6, 7, 8, 7],
    'Baltimore': [8, 9, 5, 8]
}

fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

for i, (city, values) in enumerate(data.items()):
    x_pos = np.arange(len(categories))
    bars = axs[i].bar(x_pos, [values[cat] for cat in categories], color=colors, yerr=error_bars[city], ecolor='black')
    
    axs[i].set_xlabel(f'({chr(97 + i)}) {city}')
    axs[i].set_xticks(x_pos)
    axs[i].set_xticklabels(categories, rotation=45)
    axs[i].set_ylim(30, 100)
    axs[i].set_ylabel('Percentage')
    axs[i].set_axisbelow(True)
    axs[i].grid(True, axis='y', linestyle='-', alpha=0.7)

fig.legend(bars, categories, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/errorbar_19.png", bbox_inches="tight")
