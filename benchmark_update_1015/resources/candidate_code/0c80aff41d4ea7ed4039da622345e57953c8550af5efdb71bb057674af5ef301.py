import matplotlib.pyplot as plt
import numpy as np

# Create dummy data based on visual analysis
np.random.seed(42)
t2tt_data = np.array([
    19.0, 19.4, 19.6, 19.7, 19.75, 19.8, 20.0, 20.2,
    20.5, 20.5, 20.5, 20.8, 21.0, 21.2, 21.25, 21.3,
    21.5, 21.7, 22.0
])
s2tt_data = np.array([
    18.0, 18.4, 18.5, 18.6, 18.65, 18.7, 18.8, 18.9,
    19.0, 19.0, 19.0, 19.1, 19.2, 19.3, 19.35, 19.4,
    19.5, 19.7, 20.0
])

# Create figure with specified dimensions
fig, ax = plt.subplots(figsize=(6, 4))

# Create box plots
bp = ax.boxplot([t2tt_data, s2tt_data], 
                 positions=[1, 2],
                 widths=0.5,
                 patch_artist=True,
                 medianprops=dict(color='#FF8C00', linewidth=1.5),
                 boxprops=dict(facecolor='white', edgecolor='black', linewidth=1),
                 whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black', linewidth=1),
                 flierprops=dict(marker='o', markerfacecolor='black', markersize=4))

# Set labels
ax.set_ylabel('BLEU', fontsize=11)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Llama-70b\n(T2TT)', 'Llama-70b\n(S2TT)'], fontsize=10)

# Set y-axis limits
ax.set_ylim(16, 23)

# Add horizontal dashed reference segments
median_t2tt = np.median(t2tt_data)
median_s2tt = np.median(s2tt_data)
annotation_lower = median_t2tt - 1.0
ax.plot([1, 2.35], [median_t2tt, median_t2tt], color='gray', linestyle='--', linewidth=1, zorder=1)
ax.plot([2, 2.35], [annotation_lower, annotation_lower], color='gray', linestyle='--', linewidth=1, zorder=1)

# Add annotation for 1.0 BLEU difference
ax.annotate('', xy=(2.35, median_t2tt), xytext=(2.35, annotation_lower),
            arrowprops=dict(arrowstyle='->', color='black', linewidth=1))
ax.text(2.45, (median_t2tt + annotation_lower) / 2, '1.0 BLEU', 
        fontsize=9, va='center', ha='left')

# Set x-axis limits
ax.set_xlim(0.5, 2.8)

# Add grid on y-axis
ax.yaxis.grid(False)
ax.xaxis.grid(False)

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/box_11.png", bbox_inches="tight")
