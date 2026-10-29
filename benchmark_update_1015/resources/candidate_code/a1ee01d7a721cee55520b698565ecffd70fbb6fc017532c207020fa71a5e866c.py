import matplotlib.pyplot as plt
import numpy as np

# Data for the stacked bar chart
annotators = [1, 2, 3, 4, 5, 6]
scores = {
    'Score 1': [10, 10, 8, 15, 12, 10],
    'Score 2': [12, 12, 12, 9, 12, 7],
    'Score 3': [8, 5, 8, 6, 8, 10],
    'Score 4': [11, 15, 10, 10, 12, 15],
    'Score 5': [10, 8, 12, 10, 6, 8]
}

# Define colors for each score
colors = ['#002B00', '#006400', '#32CD32', '#90EE90', '#D0F0C0']

# Set figure size
plt.figure(figsize=(8, 5))

# Create the stacked bar chart
bottom = np.zeros(len(annotators))
for i, (score_name, values) in enumerate(scores.items()):
    plt.bar(annotators, values, bottom=bottom, color=colors[i], label=score_name, edgecolor='none')
    # Add text labels inside each segment
    for j, value in enumerate(values):
        if value > 0:
            plt.text(annotators[j], bottom[j] + value/2, str(value), ha='center', va='center', fontsize=10, color='black')
    bottom += values

# Customize the plot
plt.xlabel('Human Annotator', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.title('Human Labeling and Agreement Bias Checking', fontsize=14, pad=15)
plt.xticks(annotators, fontsize=10)
plt.yticks(range(0, 51, 10), fontsize=10)
plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

# Add legend
legend = plt.legend(title='Scores', title_fontsize=10, fontsize=10, loc='upper right', bbox_to_anchor=(1.05, 1), frameon=False)

# Adjust layout to prevent clipping
plt.tight_layout()

# Show the plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/bar_70.png", bbox_inches="tight")