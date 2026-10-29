import matplotlib.pyplot as plt
import numpy as np

# Dummy data for the left subplot
token_costs_left = [35, 30, 20, 10, 5]
accuracies_left = [68, 66, 64, 66, 62]

# Dummy data for the right subplot
tflops_right = [2.5, 2.0, 1.5, 1.0, 0.5]
accuracies_right = [68, 66, 64, 62, 62]

# Create the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Left subplot
axs[0].scatter(token_costs_left, accuracies_left, color=['red', 'red', 'blue', 'blue', 'green'])
axs[0].set_xlabel('Token cost($)')
axs[0].set_ylabel('Accuracy(%)')
axs[0].set_title('Logic Grid Puzzle(Accuracy versus token cost)')
axs[0].set_xlim(0, 50)
axs[0].set_ylim(60, 70)

# Right subplot
axs[1].scatter(tflops_right, accuracies_right, color=['red', 'red', 'blue', 'blue', 'green'])
axs[1].set_xlabel('TFLOPS')
axs[1].set_ylabel('Accuracy(%)')
axs[1].set_title('Logic Grid Puzzle(Accuracy versus TFLOPS)')
axs[1].set_xlim(0, 1000000)
axs[1].set_ylim(60, 70)

# Show the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/scatter_15.png", bbox_inches="tight")