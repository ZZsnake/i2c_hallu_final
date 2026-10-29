import numpy as np
import matplotlib.pyplot as plt

# Create time data (0 to 24 hours)
t = np.linspace(0, 24, 200)

# Construct smooth dummy data resembling the figure
# Temperature: oscillates roughly between 10 and 20
temp = 15 + 5 * np.sin((t - 3) * np.pi / 12)

# Humidity: roughly inverse of temperature, between 30 and 70
humid = 50 - 20 * np.sin((t - 3) * np.pi / 12)

# Set figure size 10x5
plt.figure(figsize=(10, 5))

# Plot temperature (red) and humidity (blue)
plt.plot(t, temp, color='red', linewidth=2, label='Temperature (°C)')
plt.plot(t, humid, color='blue', linewidth=2, label='Humidity (%)')

# Axes labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.title('Temperature and Humidity Over a Day')

# Set axis limits to match the image
plt.xlim(0, 24)
plt.ylim(0, 100)

# Add legend in upper right
plt.legend(loc='upper right')

# Tight layout for clean appearance
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/density_10.png", bbox_inches="tight")