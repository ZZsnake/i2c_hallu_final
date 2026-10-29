import matplotlib.pyplot as plt
import numpy as np

# Data based on visual inspection
judges = ['GPT-3.5-Turbo', 'Claude 2.1', 'GPT-4-Turbo', 'Human']
protocols = ['Expert', 'Blind', 'Consultancy', 'Interactive Consultancy', 'Debate', 'Interactive Debate']

# Accuracy data (estimated from chart)
data = {
    'Expert': [100, 105, 110, 113],
    'Blind': [50, 55, 60, 65],
    'Consultancy': [60, 65, 70, 75],
    'Debate': [80, 85, 90, 95],
    'Interactive Consultancy': [70, 75, 80, 85],
    'Interactive Debate': [90, 95, 100, 107]
}

# Error bars (estimated from chart)
errors = {
    'Expert': [5, 5, 5, 3],
    'Blind': [5, 5, 5, 3],
    'Consultancy': [5, 5, 5, 5],
    'Debate': [5, 5, 5, 5],
    'Interactive Consultancy': [5, 5, 5, 5],
    'Interactive Debate': [5, 5, 10, 5]
}

# Colors and patterns
colors = {
    'Expert': 'white',
    'Blind': '#4A6FA5',
    'Consultancy': '#D97D3E',
    'Interactive Consultancy': '#5A9A8A',
    'Debate': '#A85A3A',
    'Interactive Debate': '#9A7AA5'
}
hatches = {
    'Expert': '///',
    'Blind': '',
    'Consultancy': '',
    'Interactive Consultancy': '',
    'Debate': '',
    'Interactive Debate': ''
}
edgecolors = {protocol: 'black' for protocol in protocols}

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Bar positions
x = np.arange(len(judges))
width = 0.15
expert_width = 0.75

# Plot the wide Expert bars behind each judge group
expert_index = protocols.index('Expert')
ax.bar(x + (expert_index + 3) * width, data['Expert'], expert_width,
       label='Expert',
       color=colors['Expert'],
       edgecolor=edgecolors['Expert'],
       hatch=hatches['Expert'],
       linewidth=1.0,
       yerr=errors['Expert'],
       error_kw={'ecolor': 'black', 'capsize': 5, 'elinewidth': 1.5, 'capthick': 1.5},
       zorder=1)

# Plot colored protocol bars in front of the Expert bars
for i, protocol in enumerate(protocols):
    if protocol == 'Expert':
        continue
    values = data[protocol]
    errs = errors[protocol]
    ax.bar(x + i * width, values, width,
           label=protocol,
           color=colors[protocol],
           edgecolor=edgecolors[protocol],
           hatch=hatches[protocol],
           linewidth=1.0,
           yerr=errs,
           error_kw={'ecolor': 'black', 'capsize': 3, 'elinewidth': 1.5, 'capthick': 1.5},
           zorder=2)

# Customize the plot
ax.set_ylabel('Judge Accuracy (%)', fontsize=11)
ax.set_xlabel('Judge', fontsize=11)
ax.set_xticks(x + width * (len(protocols) - 1) / 2)
ax.set_xticklabels(judges)
ax.set_ylim(0, 120)
ax.set_yticks(np.arange(0, 121, 20))

# Legend
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, frameon=True,
                   edgecolor='black', fontsize=9, title='Protocol')
legend.get_title().set_fontsize(10)

# Add border
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/errorbar_2.png", bbox_inches="tight")
