import matplotlib.pyplot as plt
import numpy as np

# Set figure size
plt.figure(figsize=(6, 4))

# Dummy data mimicking the boxplot structure
np.random.seed(42)
data_T2TT = np.random.normal(loc=20.5, scale=1.0, size=100)
data_S2TT = np.random.normal(loc=19.5, scale=0.8, size=100)

# Create boxplot
boxprops = dict(edgecolor='black', facecolor='white')
medianprops = dict(color='orange')
whiskerprops = dict(color='black')
capprops = dict(color='black')

bp = plt.boxplot([data_T2TT, data_S2TT],
                 labels=['Llama-70b\n(T2TT)', 'Llama-70b\n(S2TT)'],
                 boxprops=boxprops,
                 medianprops=medianprops,
                 whiskerprops=whiskerprops,
                 capprops=capprops,
                 patch_artist=True)

# Get median values
med_T2TT = np.median(data_T2TT)
med_S2TT = np.median(data_S2TT)

# Add dashed line from T2TT median to right
plt.axhline(y=med_T2TT, color='black', linestyle='--', linewidth=1, zorder=0)

# Add vertical arrow from S2TT median to T2TT median
plt.annotate('', xy=(1.8, med_S2TT), xytext=(1.8, med_T2TT),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))

# Add text "1.0 BLEU" next to the arrow
plt.text(1.85, (med_T2TT + med_S2TT) / 2, '1.0 BLEU', va='center', ha='left', fontsize=10)

# Set y-axis limits and label
plt.ylim(16, 22)
plt.ylabel('BLEU')

# Remove top and right spines for cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/box_11.png", bbox_inches="tight")