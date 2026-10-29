import matplotlib.pyplot as plt
import numpy as np

# Create dummy data
time = np.arange(0, 24, 0.1)
temperature = 15 + 5 * np.sin(2 * np.pi * time / 19)
humidity = 50 + 20 * np.cos(time / 4)

# Create the plot
plt.figure(figsize=(12, 5))
plt.plot(time, temperature, color='red', label='Temperature (°C)')
plt.plot(time, humidity, color='blue', label='Humidity (%)')

# Set the axis labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.title('Temperature and Humidity Over a Day')

# Set the axis limits
plt.xlim(0, 24)
plt.ylim(0, 100)

# Add a grid
plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

# Match axes spines to the reference
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add a legend
plt.legend(loc='upper right')

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/density_10.png", bbox_inches="tight")
