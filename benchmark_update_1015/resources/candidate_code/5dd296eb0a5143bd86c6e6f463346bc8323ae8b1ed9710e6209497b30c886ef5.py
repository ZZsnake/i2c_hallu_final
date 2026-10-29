import matplotlib.pyplot as plt
import numpy as np

# Create time data (0 to 24 hours)
time = np.linspace(0, 24, 100)

# Create temperature data (inverse parabola-like curve)
temperature = 15 + 5 * np.sin((time - 5) * np.pi / 12) - 0.3 * (time - 12)**2 / 12
temperature = np.clip(temperature, 10, 21)

# Create humidity data (parabola-like curve, inverse of temperature pattern)
humidity = 70 - 40 * np.sin((time - 5) * np.pi / 12) + 0.8 * (time - 12)**2 / 12
humidity = np.clip(humidity, 30, 72)

# Create figure with specified dimensions
fig, ax = plt.subplots(figsize=(10, 5))

# Plot temperature and humidity
ax.plot(time, temperature, color='red', linewidth=2, label='Temperature (°C)')
ax.plot(time, humidity, color='darkblue', linewidth=2, label='Humidity (%)')

# Set labels and title
ax.set_xlabel('Time (hours)', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Temperature and Humidity Over a Day', fontsize=12)

# Set axis limits
ax.set_xlim(0, 24)
ax.set_ylim(0, 100)

# Add grid
ax.grid(True, linestyle='-', alpha=0.3, color='gray')

# Add legend
ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)

# Adjust layout
plt.tight_layout()

# Display plot
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/density_10.png", bbox_inches="tight")