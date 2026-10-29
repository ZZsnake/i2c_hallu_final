import matplotlib.pyplot as plt
import numpy as np

# Generate dummy data based on observed trends
time = np.linspace(0, 24, 100)
humidity = 50 + 20 * np.cos(2 * np.pi * time / 25)  # Smooth daily humidity trend
temperature = 15 + 5 * np.sin(2 * np.pi * time / 19)  # Smooth sinusoidal temperature trend

# Create plot
plt.figure(figsize=(10, 5))
plt.plot(time, temperature, 'r-', label='Temperature (°C)')
plt.plot(time, humidity, 'b-', label='Humidity (%)')

# Formatting
plt.title('Temperature and Humidity Over a Day')
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.legend(loc='best')
plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 5))
plt.ylim(0, 100)
plt.grid(True, linestyle=':', linewidth=0.5)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/density_10.png", bbox_inches="tight")
