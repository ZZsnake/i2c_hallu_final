import matplotlib.pyplot as plt
import numpy as np

methods = ['Self-refine', 'MAD+judge', 'ToT', 'CoT(maj@1)', 'SPP', 'DefInt', 'CoT(maj@5)']
colors = ['green', 'brown', 'red', 'blue', 'purple', 'pink', 'orange']

# Data for the left subplot
token_costs_left = [45, 35, 30, 20, 15, 10, 10]
accuracies_left = [62, 68, 66, 64, 68, 65, 66]

# Data for the right subplot
tflops_right = [2.5e6, 2.5e6, 2.0e6, 1.0e6, 1.5e6, 0.5e6, 0.5e6]
accuracies_right = [62, 68, 66, 64, 68, 62, 66]

# Create the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Left subplot
axs[0].scatter(token_costs_left, accuracies_left, color=colors)
for x, y, label in zip(token_costs_left, accuracies_left, methods):
    axs[0].annotate(label, (x, y), textcoords='offset points', xytext=(-10, 5))
axs[0].set_xlabel('Token cost($)')
axs[0].set_ylabel('Accuracy(%)')
axs[0].set_title('Logic Grid Puzzle(Accuracy versus token cost)')
axs[0].set_xlim(50, 0)
axs[0].set_ylim(60, 70)

# Right subplot
axs[1].scatter(tflops_right, accuracies_right, color=colors)
for x, y, label in zip(tflops_right, accuracies_right, methods):
    axs[1].annotate(label, (x, y), textcoords='offset points', xytext=(-15, 5))
axs[1].set_xlabel('TFLOPS')
axs[1].set_ylabel('Accuracy(%)')
axs[1].set_title('Logic Grid Puzzle(Accuracy versus TFLOPS)')
axs[1].set_xlim(3.0e6, 0)
axs[1].set_ylim(60, 70)

# Show the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/scatter_15.png", bbox_inches="tight")
