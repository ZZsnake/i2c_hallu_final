import matplotlib.pyplot as plt
import numpy as np

# Set figure dimensions
plt.figure(figsize=(6, 5))

# Generate dummy data
ensemble_size = np.arange(0, 31)
accuracy = 74.5 + 0.2 * np.exp(-0.1 * ensemble_size) + 0.1 * np.random.randn(len(ensemble_size))
confidence_interval = 0.1 * np.ones_like(ensemble_size)

# Plotting
plt.plot(ensemble_size, accuracy, color='blue', label='Accuracy')
plt.fill_between(ensemble_size, accuracy - confidence_interval, accuracy + confidence_interval, color='lightblue', alpha=0.5)

# Labels and title
plt.xlabel('Ensemble Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Ensemble Size')

# Grid and legend
plt.grid(True)
plt.legend()

# Show plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/line_37.png", bbox_inches="tight")