import matplotlib.pyplot as plt
import numpy as np

# Data
annotators = ['1', '2', '3', '4', '5', '6']
score1 = [10, 10, 8, 15, 12, 10]
score2 = [12, 12, 12, 9, 12, 7]
score3 = [8, 5, 8, 6, 8, 10]
score4 = [11, 15, 10, 10, 12, 15]
score5 = [10, 8, 12, 10, 6, 8]

# Colors
colors = [plt.get_cmap("BuGn")(1 - i / 5) for i in range(5)]

# Create the stacked bar chart
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(annotators, score1, color=colors[0], label='Score 1')
ax.bar(annotators, score2, bottom=score1, color=colors[1], label='Score 2')
ax.bar(annotators, score3, bottom=np.array(score1) + np.array(score2), color=colors[2], label='Score 3')
ax.bar(annotators, score4, bottom=np.array(score1) + np.array(score2) + np.array(score3), color=colors[3], label='Score 4')
ax.bar(annotators, score5, bottom=np.array(score1) + np.array(score2) + np.array(score3) + np.array(score4), color=colors[4], label='Score 5')

# Add data labels
for i in range(len(annotators)):
    ax.text(annotators[i], score1[i] / 2, str(score1[i]), ha='center', va='center', color='white')
    ax.text(annotators[i], np.array(score1)[i] + score2[i] / 2, str(score2[i]), ha='center', va='center', color='white')
    ax.text(annotators[i], np.array(score1)[i] + np.array(score2)[i] + score3[i] / 2, str(score3[i]), ha='center', va='center', color='white')
    ax.text(annotators[i], np.array(score1)[i] + np.array(score2)[i] + np.array(score3)[i] + score4[i] / 2, str(score4[i]), ha='center', va='center', color='white')
    ax.text(annotators[i], np.array(score1)[i] + np.array(score2)[i] + np.array(score3)[i] + np.array(score4)[i] + score5[i] / 2, str(score5[i]), ha='center', va='center', color='black')

# Set labels and title
ax.set_xlabel('Human Annotator')
ax.set_ylabel('Scores')
ax.set_title('Human Labeling and Agreement Bias Checking')
ax.set_ylim(0, 50)
ax.grid(axis='y', linestyle='--', color='0.7', alpha=0.8)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend
ax.legend(title='Scores', loc='upper left', bbox_to_anchor=(1.04, 1.0))

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/bar_70.png", bbox_inches="tight")
