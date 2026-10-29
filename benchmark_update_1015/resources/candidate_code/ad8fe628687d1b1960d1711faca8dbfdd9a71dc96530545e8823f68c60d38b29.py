import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['GPT-3.5-Turbo', 'Claude 2.1', 'GPT-4-Turbo', 'Human']
protocols = ['Expert', 'Blind', 'Consultancy', 'Debate', 'Interactive Consultancy', 'Interactive Debate']
accuracy = {
    'GPT-3.5-Turbo': [60, 70, 80, 90, 95, 100],
    'Claude 2.1': [55, 65, 75, 85, 95, 100],
    'GPT-4-Turbo': [60, 70, 80, 90, 95, 100],
    'Human': [80, 85, 90, 95, 100, 105]
}

# Colors
colors = {'Expert': 'grey', 'Blind': 'blue', 'Consultancy': 'orange', 'Debate': 'green', 
          'Interactive Consultancy': 'purple', 'Interactive Debate': 'pink'}

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.15
x = np.arange(len(models))
for i, protocol in enumerate(protocols):
    values = [accuracy[model][i] for model in models]
    ax.bar(x + i * bar_width, values, bar_width, label=protocol, color=colors[protocol])

ax.set_xlabel('Judge')
ax.set_ylabel('Judge Accuracy (%)')
ax.set_title('Protocol Comparison by Judge Accuracy')
ax.set_xticks(x + bar_width * len(protocols) / 2)
ax.set_xticklabels(models)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/errorbar_2.png", bbox_inches="tight")