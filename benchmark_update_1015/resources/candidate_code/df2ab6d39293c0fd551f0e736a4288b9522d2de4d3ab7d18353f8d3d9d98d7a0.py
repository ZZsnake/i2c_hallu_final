import matplotlib.pyplot as plt
import numpy as np

# Dummy data approximating the distributions
np.random.seed(0)
t2tt = np.array([19.0, 19.2, 19.4, 19.5, 19.6, 19.7, 19.75, 19.8, 19.9, 20.0,
                 20.1, 20.2, 20.3, 20.4, 20.5, 20.5, 20.6, 20.7, 20.8, 21.0,
                 21.1, 21.2, 21.3, 21.35, 21.4, 21.5, 21.6, 21.7, 21.8, 22.0])  # left box, higher BLEU
s2tt = np.array([18.0, 18.3, 18.5, 18.55, 18.6, 18.6, 18.65, 18.7, 18.75, 18.8,
                 18.85, 18.9, 18.95, 19.0, 19.0, 19.0, 19.05, 19.1, 19.15, 19.2,
                 19.25, 19.3, 19.35, 19.4, 19.45, 19.5, 19.6, 19.7, 19.85, 20.0])  # right box, lower BLEU

fig, ax = plt.subplots(figsize=(6, 4))

# Create boxplots
boxprops = dict(linewidth=1.5)
medianprops = dict(color='orange', linewidth=1.5)
whiskerprops = dict(linewidth=1.5)
capprops = dict(linewidth=1.5)

bp = ax.boxplot(
    [t2tt, s2tt],
    positions=[1, 2],
    widths=0.5,
    patch_artist=False,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops
)

# Axis labels and limits
ax.set_ylabel("BLEU")
ax.set_xticks([1, 2])
ax.set_xticklabels(["Llama-70b\n(T2TT)", "Llama-70b\n(S2TT)"])
ax.set_ylim(16, 23)
ax.set_yticks([16, 18, 20, 22])

# Extract median y-values to position dashed lines
medians = [line.get_ydata()[0] for line in bp['medians']]
median_t2tt = medians[0]
median_s2tt = medians[1]

arrow_x = 2.15
arrow_top = median_t2tt
arrow_bottom = median_s2tt

ax.hlines(
    y=arrow_top,
    xmin=1,
    xmax=arrow_x,
    colors='gray',
    linestyles='dashed',
    linewidth=1
)

ax.hlines(
    y=arrow_bottom,
    xmin=2,
    xmax=arrow_x,
    colors='gray',
    linestyles='dashed',
    linewidth=1
)

ax.annotate(
    '',
    xy=(arrow_x, arrow_top),
    xytext=(arrow_x, arrow_bottom),
    arrowprops=dict(arrowstyle='->', color='black', linewidth=1)
)

ax.text(
    2.26, (arrow_top + arrow_bottom) / 2,
    "1.0 BLEU",
    va='center',
    ha='left',
    fontsize=9
)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/box_11.png", bbox_inches="tight")