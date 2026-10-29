import matplotlib.pyplot as plt
import numpy as np

# Dummy data
models = ['ResNet18', 'AlexNet', 'ResNet50']
perturbations = ['0%', '50%', '4:8', '75%', '6:8', '87.50%', '7:8']
speedup = [
    [1.0, 1.8, 2.6, 3.3, 4.6, 4.5, 7.4],
    [1.0, 1.9, 2.9, 3.4, 5.2, 6.1, 8.7],
    [1.0, 1.6, 1.8, 2.7, 2.9, 3.9, 4.3]
]

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 4))

# Set the positions of the bars
x = np.arange(len(models))
width = 0.25

# Plot the bars
for i, model in enumerate(models):
    for j, perturbation in enumerate(perturbations):
        ax.bar(x + i * width, speedup[i][j], width, label=perturbation, color=plt.cm.tab20c(j))

# Set the x-axis labels
ax.set_xticks(x + width, labels=models)

# Set the y-axis label
ax.set_ylabel('Speedup')

# Set the title
ax.set_title('Speedup Comparison Across Models and Perturbation Levels')

# Add the legend
ax.legend(title='Perturbation')

# Show the grid
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/bar_37.png", bbox_inches="tight")