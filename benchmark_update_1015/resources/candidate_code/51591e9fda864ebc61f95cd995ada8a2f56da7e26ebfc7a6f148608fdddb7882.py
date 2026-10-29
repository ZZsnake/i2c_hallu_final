import matplotlib.pyplot as plt
import numpy as np

# Dummy data based on visual trends
augmentation_levels = np.linspace(0, 8, 100)
processing_time = 1.5 - 0.1 * augmentation_levels + 0.5 * np.sin(augmentation_levels)
power_consumption = 1.5 + 0.2 * augmentation_levels + 0.3 * np.sin(augmentation_levels)
latency = 1.5 + 0.1 * augmentation_levels + 0.2 * np.sin(augmentation_levels)

# Cumulative metrics
cumulative_metrics = np.vstack([processing_time, power_consumption, latency]).cumsum(axis=0)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.stackplot(augmentation_levels, cumulative_metrics, labels=['Processing Time', 'Power Consumption', 'Latency'], colors=['red', 'orange', 'blue'], alpha=0.7)
ax.set_title('Cumulative Performance Metrics by Augmentation Level')
ax.set_xlabel('Augmentation Level')
ax.set_ylabel('Performance Metrics')
ax.set_xlim(0, 8)
ax.set_ylim(0, 10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/area_14.png", bbox_inches="tight")