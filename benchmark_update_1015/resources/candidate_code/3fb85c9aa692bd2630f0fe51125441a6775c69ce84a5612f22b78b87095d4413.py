import matplotlib.pyplot as plt
import numpy as np

# Data
augmentation_labels = ['0', '0.125', '0.25', '0.5', '1', '2', '4', '8']
augmentation_levels = np.arange(len(augmentation_labels))
processing_time = np.array([2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.1, 1.0])
power_consumption = np.array([1.0, 1.5, 2.0, 2.2, 2.5, 2.7, 2.8, 3.0])
latency = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3])

# Cumulative data for stacked area plot
cumulative_processing_time = processing_time
cumulative_power_consumption = processing_time + power_consumption
cumulative_latency = cumulative_power_consumption + latency

# Plotting
plt.figure(figsize=(10, 6))
plt.stackplot(augmentation_levels, processing_time, power_consumption, latency,
              labels=['Processing Time', 'Power Consumption', 'Latency'],
              colors=['salmon', 'lightsalmon', 'skyblue'])

# Adding labels and title
plt.xlabel('Augmentation Level')
plt.ylabel('Performance Metrics')
plt.title('Cumulative Performance Metrics by Augmentation Level', y=1.05)
plt.xticks(augmentation_levels, augmentation_labels)
plt.xlim(0, 7)
plt.ylim(0, 10)
plt.yticks(np.arange(0, 11, 1))
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False)

# Adding grid
plt.grid(True, linestyle='--', alpha=0.7)

# Hide plot spines for a borderless plotting area
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(False)

# Show plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/area_14.png", bbox_inches="tight")
