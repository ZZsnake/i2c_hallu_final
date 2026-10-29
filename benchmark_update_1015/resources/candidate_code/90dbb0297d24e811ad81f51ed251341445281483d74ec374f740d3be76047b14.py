import matplotlib.pyplot as plt
import numpy as np

# Generate dummy data to match visual trends
hours = np.linspace(0, 24, 100)
temperature = 15 + 5 * np.sin(hours / 3)                    # Peaks slightly before 5h
humidity = 50 + 20 * np.cos(hours / 4)                         # Peaks near 70% at 0h and 24h, minimum near 12.5h

# Create plot
plt.figure(figsize=(10, 5))

# Plot lines
plt.plot(hours, temperature, color='red', linewidth=2, label='Temperature (°C)')
plt.plot(hours, humidity, color='blue', linewidth=2, label='Humidity (%)')

# Customize axes
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.title('Temperature and Humidity Over a Day')
plt.xticks([0, 5, 10, 15, 20])
plt.yticks(range(0, 101, 20))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Match visible axes/spines and plot domain
plt.xlim(0, 24)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend
plt.legend(loc='upper right')

# Set y-axis limits to match image
plt.ylim(0, 100)

# Show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/density_10.png", bbox_inches="tight")
