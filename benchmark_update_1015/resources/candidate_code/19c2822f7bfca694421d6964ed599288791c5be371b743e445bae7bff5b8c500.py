import matplotlib.pyplot as plt
import numpy as np

# 1. Set up the figure and axes
fig, ax = plt.subplots(figsize=(12, 5))

# 2. Create reasonable dummy data based on visual trends
# Time from 0 to 24 hours to represent the full span of a day shown in the plot
t = np.linspace(0, 24, 500)

# Temperature (°C) model: min=10, max=20, with the curve starting near 15
# and rising to an early-day peak before falling to an afternoon minimum.
temperature = 15 + 5 * np.sin(t / 3)

# Humidity (%) model: min=30, max=70, period=24h.
# Starts high, decreases through the day, and returns close to 70 by hour 24.
humidity = 50 + 20 * np.cos(t / 4)

# 3. Plot the data with specified styling
ax.plot(t, temperature, label='Temperature (°C)', color='red', linewidth=2)
ax.plot(t, humidity, label='Humidity (%)', color='blue', linewidth=2)

# 4. Set titles and labels
ax.set_title('Temperature and Humidity Over a Day', fontsize=14)
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Value', fontsize=12)

# 5. Set axis limits and ticks
ax.set_xlim(0, 24)
ax.set_ylim(0, 100)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_yticks(np.arange(0, 101, 20))

# 6. Add grid
ax.grid(True, linestyle='--', color='lightgray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 7. Add legend
ax.legend(loc='upper right')

# 8. Display the plot
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/density_10.png", bbox_inches="tight")
