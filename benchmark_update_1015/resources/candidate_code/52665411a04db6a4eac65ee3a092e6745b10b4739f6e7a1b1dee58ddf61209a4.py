import matplotlib.pyplot as plt

# Dummy data based on visual inspection
methods = ["Self-refine", "ToT", "CoT(maj@1)", "CoT(maj@5)", "Defint", "SPP", "MAD+judge"]
accuracy = [62.2, 66.0, 64.2, 66.0, 65.0, 68.5, 68.5]
token_cost = [50, 30, 20, 10, 12, 15, 40]
tflops = [2.5, 2.0, 1.0, 0.5, 0.6, 1.5, 2.5]

colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#e377c2', '#9467bd', '#d62728']
# Note: MAD+judge and ToT share same color (red), SPP and MAD+judge share same color (dark red) in image

# Set figure size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left plot: Accuracy vs Token Cost
ax1.scatter(token_cost, accuracy, c=colors, s=50, edgecolors='black', linewidth=0.5)
for i, method in enumerate(methods):
    ax1.annotate(method, (token_cost[i], accuracy[i]), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=10)
ax1.set_title("Logic Grid Puzzle(Accuracy versus token cost)")
ax1.set_xlabel("Token cost($)")
ax1.set_ylabel("Accuracy(%)")
ax1.set_ylim(60, 70)
ax1.set_xlim(0, 50)
ax1.set_yticks([60, 62, 64, 66, 68, 70])
ax1.set_xticks([0, 10, 20, 30, 40, 50])

# Right plot: Accuracy vs TFLOPS
ax2.scatter(tflops, accuracy, c=colors, s=50, edgecolors='black', linewidth=0.5)
for i, method in enumerate(methods):
    ax2.annotate(method, (tflops[i], accuracy[i]), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=10)
ax2.set_title("Logic Grid Puzzle(Accuracy versus TFLOPS)")
ax2.set_xlabel("TFLOPS")
ax2.set_ylabel("Accuracy(%)")
ax2.set_ylim(60, 70)
ax2.set_xlim(0, 3.0)
ax2.set_yticks([60, 62, 64, 66, 68, 70])
ax2.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# Add "166" label at the far right (as seen in image, likely a mislabel)
ax2.text(3.0, 60, "166", ha='right', va='bottom', fontsize=10)

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/scatter_15.png", bbox_inches="tight")