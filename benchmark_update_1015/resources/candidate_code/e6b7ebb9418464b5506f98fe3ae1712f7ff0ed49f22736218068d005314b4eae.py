import matplotlib.pyplot as plt
import numpy as np

# Create dummy data based on observed trends
time = np.arange(0, 24, 1)
temperature = [15 + 0.8 * np.sin(2 * np.pi * t / 24) + 0.5 * np.cos(2 * np.pi * t / 24) for t in range(24)]
humidity = [70 - 10 * np.sin(2 * np.pi * t / 24) + 5 * np.cos(2 * np.pi * t / 24) for t in range(24)]

# Set up the plot
plt.figure(figsize=(10, 5))
plt.plot(time, temperature, color='red', label='Temperature (°C)')
plt.plot(time, humidity, color='blue', label='Humidity (%)')

# Add labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.title('Temperature and Humidity Over a Day')
plt.legend()

# Add grid
plt.grid(True)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/density_10.png", bbox_inches="tight")