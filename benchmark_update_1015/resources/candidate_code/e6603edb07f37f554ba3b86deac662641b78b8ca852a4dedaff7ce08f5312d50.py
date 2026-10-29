import matplotlib.pyplot as plt
import numpy as np

# Create dummy data based on observed trends
time = np.arange(0, 25, 1)
temperature = [15 + 5 * np.sin(2 * np.pi * t / 19) for t in range(25)]
humidity = [50 + 20 * np.cos(t / 4) for t in range(25)]

# Set up the plot
plt.figure(figsize=(10, 5))
plt.plot(time, temperature, color='red', label='Temperature (°C)')
plt.plot(time, humidity, color='blue', label='Humidity (%)')

# Add labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.title('Temperature and Humidity Over a Day')
plt.ylim(0, 100)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 101, 20))
plt.legend(loc='upper right')

# Match axes frame style
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add grid
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Show the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/density_10.png", bbox_inches="tight")
