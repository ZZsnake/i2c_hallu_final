import matplotlib.pyplot as plt
import numpy as np

# Dummy data based on visual trends
time = np.linspace(0, 24, 100)
temperature = 15 + 5 * np.sin(time * np.pi / 12)
humidity = 70 - 35 * np.sin(time * np.pi / 12)

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(time, temperature, 'r', label='Temperature (°C)')
plt.plot(time, humidity, 'b', label='Humidity (%)')

# Set the title and labels
plt.title('Temperature and Humidity Over a Day')
plt.xlabel('Time (hours)')
plt.ylabel('Value')

# Add a legend
plt.legend(loc='upper right')

# Add grid
plt.grid(True)

# Set y-axis limits
plt.ylim(0, 100)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/density_10.png", bbox_inches="tight")