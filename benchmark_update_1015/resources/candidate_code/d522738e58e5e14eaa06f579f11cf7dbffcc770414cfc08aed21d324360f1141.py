import matplotlib.pyplot as plt
import numpy as np

# Create dummy data
time = np.arange(0, 24, 0.1)
temperature = 20 + 5 * np.sin(2 * np.pi * time / 24)
humidity = 75 - 45 * np.sin(2 * np.pi * time / 24)

# Create the plot
plt.figure(figsize=(10, 5))
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
plt.grid(True)

# Add a legend
plt.legend()

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/density_10.png", bbox_inches="tight")