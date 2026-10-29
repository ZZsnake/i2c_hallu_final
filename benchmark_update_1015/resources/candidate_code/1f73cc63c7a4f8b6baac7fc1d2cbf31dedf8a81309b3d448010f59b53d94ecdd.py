import matplotlib.pyplot as plt
import numpy as np

# Set figure dimensions
plt.figure(figsize=(6.3, 5.2))

# Data from the reference plot
ensemble_size = np.array([5, 6, 7, 8, 9, 12, 30])
accuracy = np.array([75.2, 76.1, 76.7, 77.0, 77.2, 77.5, 77.5])
confidence_interval = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.3, 0.3])

# Plotting
plt.plot(ensemble_size, accuracy, color='#1f77b4')
plt.fill_between(ensemble_size, accuracy - confidence_interval, accuracy + confidence_interval, color='#1f77b4', alpha=0.2)

# Labels
plt.xlabel('Ensemble Size', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)

# Axes ranges and ticks
plt.xlim(3.5, 32)
plt.ylim(74.5, 78.0)
plt.xticks(np.arange(6, 31, 3), fontsize=12)
plt.yticks(np.arange(74.5, 78.1, 0.5), fontsize=12)

# Show plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/line_37.png", bbox_inches="tight")
