import matplotlib.pyplot as plt
import numpy as np

# Set figure size
plt.figure(figsize=(10, 5))

# Generate dummy data based on visual trends
time = np.linspace(0, 24, 240)  # 24 hours with 240 points for smooth curves
temperature = 15 + 5 * np.sin(time / 3)  # Starts near 15°C, peaks near hour 5, dips near hour 15, then rises
humidity = 50 + 20 * np.cos(time / 4)  # Starts high, dips around midday, rises again

# Plot the lines
plt.plot(time, temperature, color='red', label='Temperature (°C)', linewidth=2)
plt.plot(time, humidity, color='blue', label='Humidity (%)', linewidth=2)

# Set axis labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.title('Temperature and Humidity Over a Day')

# Set axis limits
plt.xlim(0, 24)
plt.ylim(0, 100)

# Match visible axes: hide top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/density_10.png", bbox_inches="tight")
