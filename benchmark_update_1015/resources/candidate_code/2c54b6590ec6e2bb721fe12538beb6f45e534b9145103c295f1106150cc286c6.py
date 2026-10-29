import matplotlib.pyplot as plt
import numpy as np

# Data
judges = ['GPT-3.5-Turbo', 'Claude 2.1', 'GPT-4-Turbo', 'Human']
interaction_types = ['Expert', 'Blind', 'Consultancy', 'Protocol Debate', 'Interactive Consultancy', 'Interactive Debate']
colors = ['blue', 'orange', 'green', 'red', 'purple', 'gray']

# Dummy data (approximated from the image)
data = {
    'GPT-3.5-Turbo': [102, 95, 80, 98, 85, 92],
    'Claude 2.1': [108, 92, 75, 95, 88, 96],
    'GPT-4-Turbo': [105, 88, 70, 92, 82, 90],
    'Human': [115, 100, 85, 105, 90, 102]
}

# Error bar data (approximated)
error = {
    'GPT-3.5-Turbo': [3, 4, 5, 2, 3, 4],
    'Claude 2.1': [2, 3, 4, 2, 3, 2],
    'GPT-4-Turbo': [3, 4, 5, 2, 3, 4],
    'Human': [4, 3, 4, 3, 3, 3]
}

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(judges))
width = 0.15

for i, interaction in enumerate(interaction_types):
    values = [data[judge][i] for judge in judges]
    errors = [error[judge][i] for judge in judges]
    ax.bar(x + i * width, values, width, label=interaction, color=colors[i], yerr=errors, capsize=5)

ax.set_xlabel('Judge')
ax.set_ylabel('Judge Accuracy (%)')
ax.set_title('')  # Remove the default title
ax.set_xticks(x + width * (len(interaction_types) - 1) / 2)
ax.set_xticklabels(judges)
ax.set_ylim(0, 120)
ax.legend()

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/errorbar_2.png", bbox_inches="tight")