import matplotlib.pyplot as plt
import numpy as np

# Generate dummy data to match visual trends
hours = np.linspace(0, 24, 100)
temperature = 15 + 5 * np.sin(hours * np.pi / 12 - np.pi/2)  # Peaks ~20°C at 5h, dips ~10°C at 13h
humidity = 50 + 20 * np.cos(hours * np.pi / 12)             # Peaks ~70% at 0h and 24h, dips ~30% at 12h

# Create plot
plt.figure(figsize=(10, 5))

# Plot lines
plt.plot(hours, temperature, color='red', label='Temperature (°C)')
plt.plot(hours, humidity, color='blue', label='Humidity (%)')

# Customize axes
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.title('Temperature and Humidity Over a Day')
plt.xticks([0, 5, 10, 15, 20])
plt.yticks(range(0, 101, 20))
plt.grid(True, linestyle=':', color='gray')

# Add legend
plt.legend(loc='upper right')

# Set y-axis limits to match image
plt.ylim(0, 100)

# Show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/density_10.png", bbox_inches="tight")