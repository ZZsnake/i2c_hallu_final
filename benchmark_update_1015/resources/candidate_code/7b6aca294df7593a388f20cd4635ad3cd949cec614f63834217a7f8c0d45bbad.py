import numpy as np
import matplotlib.pyplot as plt

# Generate time data from 0 to 5 seconds with 21 points (0.25 sec intervals)
t = np.linspace(0, 5, 21)
# Calculate velocity using the quadratic function
v = 0.5 * t**2 + 2 * t - 1

# Create figure with specified size
plt.figure(figsize=(8, 6))

# Plot data with red diamond markers
plt.plot(t, v, 'D', color='red', markersize=5)

# Set title and axis labels with bold font
plt.title('Quadratic Velocity Function: v(t) = 0.5t² + 2t - 1', fontweight='bold')
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel('Velocity (m/s)', fontweight='bold')

# Configure grid, ticks, and axis limits
plt.grid(True)
plt.xticks(np.arange(0, 6, 1))
plt.yticks(np.arange(0, 25, 5))
plt.xlim(0, 5)
plt.ylim(-1, 22)  # Includes negative start point while matching trend

# Display the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/3e3fa84e1f6e40f1ac62373f79a0ff2b.png", bbox_inches="tight")