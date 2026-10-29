import matplotlib.pyplot as plt
import numpy as np

# Dummy data based on visual trends
time = np.linspace(0, 24, 100)
temperature = 15 + 5 * np.sin(time / 3)
humidity = 50 + 20 * np.cos(time / 4)

# Create the plot
plt.figure(figsize=(12, 5))
plt.plot(time, temperature, 'r', label='Temperature (°C)')
plt.plot(time, humidity, 'b', label='Humidity (%)')

# Set the title and labels
plt.title('Temperature and Humidity Over a Day')
plt.xlabel('Time (hours)')
plt.ylabel('Value')

# Add a legend
plt.legend(loc='upper right')

# Add grid
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Set axis limits
plt.xlim(0, 24)
plt.ylim(0, 100)

# Hide top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/density_10.png", bbox_inches="tight")
