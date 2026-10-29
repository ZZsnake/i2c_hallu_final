import matplotlib.pyplot as plt
import numpy as np

# Generate dummy data to match the visual trend
ensemble_sizes = np.linspace(6, 30, 10)
accuracy_mean = np.array([
    75.2, 76.8, 77.3, 77.5, 77.5, 77.5, 77.5, 77.5, 77.5, 77.5
])
accuracy_std = np.array([
    0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05
])
accuracy_upper = accuracy_mean + accuracy_std
accuracy_lower = accuracy_mean - accuracy_std

# Create the plot
plt.figure(figsize=(6, 5))

# Plot the mean line
plt.plot(ensemble_sizes, accuracy_mean, color='#004C99', linewidth=2, label='Mean Accuracy')

# Fill the confidence interval
plt.fill_between(ensemble_sizes, accuracy_lower, accuracy_upper, color='#ADD8E6', alpha=0.5, label='Confidence Interval')

# Set axis labels and limits
plt.xlabel('Ensemble Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xlim(5.5, 30.5)
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