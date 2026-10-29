import matplotlib.pyplot as plt

# Data based on visual inspection
methods = ["Self-refine", "ToT", "CoT(maj@1)", "CoT(maj@5)", "DefInt", "SPP", "MAD+judge"]
accuracy_left = [62.0, 66.0, 64.0, 66.0, 65.0, 68.0, 68.0]
accuracy_right = [62.0, 66.0, 64.0, 66.0, 62.0, 68.0, 68.0]
token_cost = [45, 30, 20, 10, 10, 15, 35]
tflops = [2.5e6, 2.0e6, 1.0e6, 0.5e6, 0.5e6, 1.5e6, 2.5e6]

colors = ['green', 'red', 'blue', 'orange', 'pink', 'purple', 'brown']

# Set figure size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left plot: Accuracy vs Token Cost
ax1.scatter(token_cost, accuracy_left, c=colors, s=36)
for i, method in enumerate(methods):
    ax1.annotate(method, (token_cost[i], accuracy_left[i]), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=10)
ax1.set_title("Logic Grid Puzzle(Accuracy versus token cost)")
ax1.set_xlabel("Token cost($)")
ax1.set_ylabel("Accuracy(%)")
ax1.set_ylim(60, 70)
ax1.set_xlim(50, 0)
ax1.set_yticks([60, 62, 64, 66, 68, 70])
ax1.set_xticks([0, 10, 20, 30, 40, 50])

# Right plot: Accuracy vs TFLOPS
ax2.scatter(tflops, accuracy_right, c=colors, s=36)
for i, method in enumerate(methods):
    ax2.annotate(method, (tflops[i], accuracy_right[i]), textcoords='offset points', xytext=(0, 5), ha='center', fontsize=10)
ax2.set_title("Logic Grid Puzzle(Accuracy versus TFLOPS)")
ax2.set_xlabel("TFLOPS")
ax2.set_ylabel("Accuracy(%)")
ax2.set_ylim(60, 70)
ax2.set_xlim(3.0e6, 0)
ax2.set_yticks([60, 62, 64, 66, 68, 70])
ax2.set_xticks([0.0, 0.5e6, 1.0e6, 1.5e6, 2.0e6, 2.5e6, 3.0e6])
ax2.ticklabel_format(axis='x', style='sci', scilimits=(6, 6))

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/scatter_15.png", bbox_inches="tight")
