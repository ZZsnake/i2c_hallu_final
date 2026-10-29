import numpy as np
import matplotlib.pyplot as plt

# Dummy data approximating the figure
np.random.seed(0)
t2tt = np.random.normal(loc=20.5, scale=0.6, size=30)
s2tt = np.random.normal(loc=18.7, scale=0.5, size=30)

fig, ax = plt.subplots(figsize=(6, 4))

# Boxplot properties
boxprops = dict(facecolor='white', edgecolor='black')
medianprops = dict(color='#ff7f0e', linewidth=1.5)  # Matplotlib default orange
whiskerprops = dict(color='black', linewidth=1.0)
capprops = dict(color='black', linewidth=1.0)

bp = ax.boxplot(
    [t2tt, s2tt],
    positions=[1, 2],
    widths=0.5,
    patch_artist=True,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops
)

# Axes labels and limits
ax.set_xticks([1, 2])
ax.set_xticklabels(['Llama-70b\n(T2TT)', 'Llama-70b\n(S2TT)'])
ax.set_ylabel('BLEU')
ax.set_ylim(16, 22.5)

# Dashed horizontal lines illustrating difference
y_top = 20.5
y_bottom = 18.8
x_left = 1
x_right = 2

ax.hlines(y_top, x_left, x_right, colors='0.5', linestyles='--', linewidth=1.0)
ax.hlines(y_bottom, x_right - 0.2, x_right, colors='0.5', linestyles='--', linewidth=1.0)

# Vertical arrow showing delta
ax.annotate(
    '',
    xy=(x_right, y_top),
    xytext=(x_right, y_bottom),
    arrowprops=dict(arrowstyle='-|>', color='black', linewidth=1.0)
)

# Text label for the difference
ax.text(
    x_right + 0.05,
    (y_top + y_bottom) / 2,
    '1.0 BLEU',
    va='center',
    ha='left'
)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/box_11.png", bbox_inches="tight")