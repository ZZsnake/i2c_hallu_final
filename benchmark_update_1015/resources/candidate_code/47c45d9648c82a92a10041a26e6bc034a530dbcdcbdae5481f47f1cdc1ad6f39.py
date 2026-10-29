import matplotlib.pyplot as plt
import numpy as np

# Generate dummy data to match the visual trend
ensemble_sizes = np.array([5, 6, 7, 8, 9, 12, 15, 18, 21, 24, 27, 30])
accuracy_mean = np.array([
    75.2, 76.1, 76.7, 77.0, 77.2, 77.5, 77.5, 77.5, 77.5, 77.5, 77.5, 77.5
])
accuracy_upper = np.array([
    76.2, 77.0, 77.5, 77.7, 77.8, 77.8, 77.8, 77.8, 77.8, 77.8, 77.8, 77.8
])
accuracy_lower = np.array([
    74.4, 75.2, 75.9, 76.3, 76.6, 77.2, 77.2, 77.2, 77.2, 77.2, 77.2, 77.2
])

# Create the plot
plt.figure(figsize=(6, 5))

# Plot the mean line
plt.plot(ensemble_sizes, accuracy_mean, color='#1f77b4', linewidth=2, label='Mean Accuracy')

# Fill the confidence interval
plt.fill_between(ensemble_sizes, accuracy_lower, accuracy_upper, color='#1f77b4', alpha=0.2, label='Confidence Interval')

# Set axis labels and limits
plt.xlabel('Ensemble Size', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xlim(3.5, 32)
plt.ylim(74.5, 78.0)

# Set tick positions
plt.xticks(np.arange(6, 31, 3))
plt.yticks(np.arange(74.5, 78.5, 0.5))

# Remove legend and grid for clean look
plt.legend().set_visible(False)
plt.grid(False)

# Tight layout for clean margins
plt.tight_layout()

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/line_37.png", bbox_inches="tight")
