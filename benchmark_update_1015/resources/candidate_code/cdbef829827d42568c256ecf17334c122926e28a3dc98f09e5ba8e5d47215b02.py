import numpy as np
import matplotlib.pyplot as plt

# Data
time = np.linspace(0, 24, 200)

# Create dummy data approximating the visual trends
temperature = 15 + 5 * np.sin(time / 3)   # ~15 ±5 °C
humidity = 50 + 20 * np.cos(2 * np.pi * time / 25)     # ~50 ±20 %

# Figure and axes
plt.figure(figsize=(10, 5))

# Plot lines
plt.plot(time, temperature, color='red', linewidth=2, label='Temperature (°C)')
plt.plot(time, humidity, color='blue', linewidth=2, label='Humidity (%)')

# Axes labels and title
plt.title('Temperature and Humidity Over a Day')
plt.xlabel('Time (hours)')
plt.ylabel('Value')

# Limits and ticks
plt.xlim(0, 24)
plt.ylim(0, 100)
plt.xticks(np.arange(0, 25, 5))
plt.yticks(np.arange(0, 101, 20))

# Grid and spines
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
plt.legend(loc='upper right')

# Tight layout for better spacing
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/density_10.png", bbox_inches="tight")
