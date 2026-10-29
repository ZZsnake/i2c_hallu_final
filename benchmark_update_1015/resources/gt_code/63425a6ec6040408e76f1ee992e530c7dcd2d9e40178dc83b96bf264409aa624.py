# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import numpy as np

# ===================
# Part 2: Data Preparation
# ===================
np.random.seed(0)
# Create some data representing the frequency of different weather events over time
time = np.linspace(0, 24, 240)
temperature = 15 + 5 * np.sin(time / 3)
humidity = 50 + 20 * np.cos(time / 4)

# Extracted strings
temperature_label = 'Temperature (°C)'
humidity_label = 'Humidity (%)'
x_label = 'Time (hours)'
y_label = 'Value'
plot_title = 'Temperature and Humidity Over a Day'

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the data
ax.plot(time, temperature, color="red", lw=2, label=temperature_label)
ax.plot(time, humidity, color="blue", lw=2, label=humidity_label)

# Customize the plot
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)
ax.tick_params(left=True, labelleft=True, bottom=True, labelbottom=True)

ax.set_ylim(0, 100)
ax.set_xlim(0, 24)

# Add labels and title
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_title(plot_title)
ax.legend()

# Add grid lines for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# ===================
# Part 4: Saving Output
# ===================
# Displaying the plot with tight layout to minimize white space
plt.tight_layout()
plt.savefig("density_10.pdf", bbox_inches="tight")