import matplotlib.pyplot as plt
import numpy as np

# Generate dummy data resembling the original plot's statistics
np.random.seed(42)
t2tt_data = np.random.normal(20.5, 1.2, 100)  # Mean ~20.5 for T2TT
s2tt_data = np.random.normal(19.0, 0.8, 100)  # Mean ~19.0 for S2TT
data = [t2tt_data, s2tt_data]

# Create figure and boxplot
fig, ax = plt.subplots(figsize=(6, 4))
boxes = ax.boxplot(data, positions=[1, 2], patch_artist=True,
                   whiskerprops=dict(linestyle='-', linewidth=1.5),
                   medianprops=dict(color='orange', linewidth=2))

# Customize box colors (optional, original uses hollow boxes)
for box in boxes['boxes']:
    box.set(facecolor='none')

# Add horizontal dashed line connecting medians (approximate y-values)
ax.plot([1, 2.1], [20.5, 20.5], 'k--', linewidth=1.2)

# Add vertical difference arrow
ax.annotate('', xy=(2.1, 20.5), xytext=(2.1, 19.5),
            arrowprops=dict(arrowstyle='->', linestyle='--', linewidth=1.2))
ax.text(2.13, 20.0, '1.0 BLEU', rotation=90, va='center', fontsize=10)

# Axis configuration
ax.set_ylim(16, 22)
ax.set_ylabel('BLEU', fontsize=11)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Llama-70b\n(T2TT)', 'Llama-70b\n(S2TT)'], fontsize=10)
ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/box_11.png", bbox_inches="tight")