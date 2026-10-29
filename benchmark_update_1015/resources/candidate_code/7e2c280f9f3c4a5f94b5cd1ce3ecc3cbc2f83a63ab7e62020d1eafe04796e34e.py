import matplotlib.pyplot as plt
import numpy as np

# Generate dummy data
time = np.arange(24)
temperature = np.sin(time / 2 * np.pi) * 20 + 15
humidity = np.cos(time / 2 * np.pi) * 40 + 30

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(time, temperature, label='Temperature (°C)', color='red')
plt.plot(time, humidity, label='Humidity (%)', color='blue')

# Adding title and labels
plt.title('Temperature and Humidity Over a Day')
plt.xlabel('Time (hours)')
plt.ylabel('Value')

# Adding grid
plt.grid(True)

# Adding legend
plt.legend()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/density_10.png", bbox_inches="tight")