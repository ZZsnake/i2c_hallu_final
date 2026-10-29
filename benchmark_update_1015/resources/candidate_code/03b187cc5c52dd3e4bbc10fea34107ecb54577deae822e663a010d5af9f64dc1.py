import matplotlib.pyplot as plt
import numpy as np

# Set figure size
plt.figure(figsize=(6, 9))

# Dummy data for Exponential Growth
time = np.arange(0, 11)
growth_data = np.array([1.3, 0.8, 0.6, 0.6, 0.4, 0.1, 0.25, 0.15, 0.1, 0.15, 0.1])

# Plot 1: Exponential Growth
plt.subplot(3, 1, 1)
plt.plot(time, growth_data, color='green', marker='o', label='Growth')
plt.title('Exponential Growth Over Time')
plt.ylabel('Value')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

# Dummy data for Exponential Decay
decay_data = np.array([1.1, 0.4, 0.2, 0.1, 0.1, 0.05, 0.0, -0.05, -0.05, -0.1, -0.05])

# Plot 2: Exponential Decay
plt.subplot(3, 1, 2)
plt.plot(time, decay_data, color='red', marker='x', label='Decay')
plt.title('Exponential Decay Over Time')
plt.ylabel('Value')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

# Dummy data for Oscillatory Behavior
oscillation_data = np.array([0.0, 0.8, 1.1, 0.0, -0.8, -1.1, 0.0, 0.9, 1.0, 0.5, -1.1])

# Plot 3: Oscillatory Behavior
plt.subplot(3, 1, 3)
plt.plot(time, oscillation_data, color='blue', marker='s', label='Oscillation')
plt.title('Oscillatory Behavior Over Time')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/line_49.png", bbox_inches="tight")