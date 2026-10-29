import matplotlib.pyplot as plt
import numpy as np

# Generate dummy data based on observed trends
time = np.linspace(0, 20, 100)
humidity = 0.4 * time**2 - 8 * time + 70  # Quadratic function for humidity
temperature = np.concatenate([
    np.linspace(15, 20, 25),  # Increase to peak at 5 hours
    np.linspace(20, 10, 50),  # Decrease to minimum at 15 hours
    np.linspace(10, 20, 25)   # Increase again towards 20 hours
])

# Create plot
plt.figure(figsize=(10, 5))
plt.plot(time, humidity, 'b-', label='Humidity (%)')
plt.plot(time, temperature, 'r-', label='Temperature (°C)')

# Formatting
plt.title('Temperature and Humidity Over a Day')
plt.xlabel('Time (hours)')
plt.ylabel('Value')
plt.legend(loc='best')
plt.ylim(0, 100)
plt.grid(True)
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/density_10.png", bbox_inches="tight")