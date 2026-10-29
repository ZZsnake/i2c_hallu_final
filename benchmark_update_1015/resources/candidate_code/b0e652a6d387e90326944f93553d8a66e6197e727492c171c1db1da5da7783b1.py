import matplotlib.pyplot as plt

# Sample data
cka_similarity_de = [0.72, 0.81, 0.89, 0.92]
mse_de = [0.125, 0.095, 0.105, 0.093]
labels_de = ['iTrm-En', 'iTrm-All', 'PatchTST', 'TiDE']

cka_similarity_pjm = [0.62, 0.75, 0.85, 0.90]
mse_pjm = [0.47, 0.46, 0.455, 0.57]
labels_pjm = ['iTrm-En', 'iTrm-All', 'PatchTST', 'TimeXer']

colors = {'iTrm-All': 'lightgreen', 'iTrm-En': 'lightgreen', 'PatchTST': 'mediumturquoise', 'TiDE': 'sandybrown', 'TimeXer': 'firebrick'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))

# Plot DE
for label, x, y in zip(labels_de, cka_similarity_de, mse_de):
    ax1.scatter(x, y, color=colors[label], label=label, edgecolor='black')
ax1.set_title('DE')
ax1.set_xlabel('CKA Similarity')
ax1.set_ylabel('MSE')
ax1.set_xlim(0.6, 1.1)
ax1.set_ylim(0.09, 0.13)
ax1.grid(True, linestyle='--')

# Plot PJM
for label, x, y in zip(labels_pjm, cka_similarity_pjm, mse_pjm):
    ax2.scatter(x, y, color=colors[label], label=label, edgecolor='black')
ax2.set_title('PJM')
ax2.set_xlabel('CKA Similarity')
ax2.set_ylabel('MSE')
ax2.set_xlim(0.5, 1.0)
ax2.set_ylim(0.40, 0.60)
ax2.grid(True, linestyle='--')

# Combine legends
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/scatter_9.png", bbox_inches="tight")