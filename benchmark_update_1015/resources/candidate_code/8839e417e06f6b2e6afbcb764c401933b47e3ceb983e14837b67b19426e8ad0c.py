import matplotlib.pyplot as plt
import numpy as np

# Dummy data for box plots
# T2TT: higher BLEU scores, wider spread
t2tt_data = np.array([19.0, 19.5, 19.75, 20.0, 20.5, 21.0, 21.25, 21.5, 22.0])
# S2TT: lower BLEU scores, narrower spread
s2tt_data = np.array([18.0, 18.4, 18.65, 18.8, 19.0, 19.2, 19.35, 19.7, 20.0])

# Create figure with specified dimensions
plt.figure(figsize=(6, 4))

# Create box plot
ax = plt.boxplot([t2tt_data, s2tt_data], 
                 patch_artist=True, 
                 boxprops=dict(facecolor='white', edgecolor='black'),
                 medianprops=dict(color='orange'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 flierprops=dict(markeredgecolor='black'))

# Set labels and title
plt.ylabel('BLEU')
plt.xticks([1, 2], ['Llama-70b\n(T2TT)', 'Llama-70b\n(S2TT)'])

# Set y-axis limits
plt.ylim(16, 23)
plt.yticks([16, 18, 20, 22])

# Add dashed line connecting medians
median_t2tt = np.median(t2tt_data)
median_s2tt = np.median(s2tt_data)
lower_ref = median_t2tt - 1.0
plt.plot([1, 2.25], [median_t2tt, median_t2tt], color='gray', linestyle='--', linewidth=1.5)
plt.plot([2, 2.25], [lower_ref, lower_ref], color='gray', linestyle='--', linewidth=1.5)

# Add arrow and annotation for 1.0 BLEU difference
arrow_start = (2.25, lower_ref)
arrow_end = (2.25, median_t2tt)
plt.annotate('', xy=arrow_end, xytext=arrow_start, 
             arrowprops=dict(arrowstyle='->', color='gray', linewidth=1.5))
plt.text(2.37, (median_t2tt + lower_ref) / 2, '1.0 BLEU', fontsize=10, va='center', ha='left')

# Keep full rectangular border around the plotting area
ax = plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# Show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/box_11.png", bbox_inches="tight")
