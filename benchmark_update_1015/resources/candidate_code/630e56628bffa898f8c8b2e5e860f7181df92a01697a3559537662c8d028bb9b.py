import matplotlib.pyplot as plt
import numpy as np

# Generate dummy data
time = np.linspace(0, 24, 240)
temperature = np.sin(time / 3) * 5 + 15
humidity = np.cos(time / 4) * 20 + 50

# Plotting
plt.figure(figsize=(12, 5))
plt.plot(time, temperature, label='Temperature (°C)', color='red')
plt.plot(time, humidity, label='Humidity (%)', color='blue')

# Adding title and labels
plt.title('Temperature and Humidity Over a Day')
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.ylim(0, 100)
plt.xlim(0, 24)

# Match visible axes styling
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adding grid
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Adding legend
plt.legend()

# Show plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-7B-Instruct/python/density_10.png", bbox_inches="tight")
