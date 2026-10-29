import matplotlib.pyplot as plt
import numpy as np

# Create time data (0 to 24 hours)
time = np.linspace(0, 24, 100)

# Create temperature data (inverse parabola-like curve)
temperature = 15 + 5 * np.sin(time / 3)

# Create humidity data (parabola-like curve, inverse of temperature pattern)
humidity = 50 + 20 * np.cos(time / 4)

# Create figure with specified dimensions
fig, ax = plt.subplots(figsize=(10, 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot temperature and humidity
ax.plot(time, temperature, color='red', linewidth=2, label='Temperature (°C)')
ax.plot(time, humidity, color='blue', linewidth=2, label='Humidity (%)')

# Set labels and title
ax.set_xlabel('Time (hours)', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Temperature and Humidity Over a Day', fontsize=12)

# Set axis limits
ax.set_xlim(0, 24)
ax.set_ylim(0, 100)

# Add grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)

# Adjust layout
plt.tight_layout()

# Display plot
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/density_10.png", bbox_inches="tight")
