import matplotlib.pyplot as plt
import numpy as np

# Data based on visual trends
augmentation_labels = ['0', '0.125', '0.25', '0.5', '1', '2', '4', '8']
augmentation_levels = np.arange(len(augmentation_labels))
processing_time = np.array([2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.1, 1.0])
power_consumption = np.array([1.0, 1.5, 2.0, 2.2, 2.5, 2.7, 2.8, 3.0])
latency = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3])

# Plotting
fig, ax = plt.subplots(figsize=(9, 6))
ax.stackplot(augmentation_levels, processing_time, power_consumption, latency, labels=['Processing Time', 'Power Consumption', 'Latency'], colors=['#d73027', '#fc8d59', '#4575b4'], alpha=0.7)
ax.set_title('Cumulative Performance Metrics by Augmentation Level', fontsize=16, loc='left', y=1.05)
ax.set_xlabel('Augmentation Level', fontsize=14)
ax.set_ylabel('Performance Metrics', fontsize=14)
ax.set_xlim(0, len(augmentation_levels) - 1)
ax.set_ylim(0, 10)
ax.set_xticks(augmentation_levels)
ax.set_xticklabels(augmentation_labels, fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False, fontsize=12)
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/area_14.png", bbox_inches="tight")
