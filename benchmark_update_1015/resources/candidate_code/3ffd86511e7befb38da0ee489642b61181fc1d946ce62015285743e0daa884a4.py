import numpy as np
import matplotlib.pyplot as plt

# Generate time data from 0 to 5 seconds with dense points (0.05 sec intervals)
t = np.linspace(0, 5, 101)
# Calculate velocity using the quadratic function
v = 0.5 * t**2 + 2 * t - 1

# Create figure with specified size
plt.figure(figsize=(8, 6))

# Plot data with red square markers with black outlines
plt.plot(t, v, 's', color='red', markeredgecolor='black', markersize=5)

# Set title and axis labels
plt.title('Quadratic Velocity Function: v(t) = 0.5t² + 2t - 1', fontweight='bold')
plt.xlabel('Time (s)', fontweight='normal', fontfamily='serif')
plt.ylabel('Velocity (m/s)', fontweight='normal', fontfamily='serif')

# Configure grid and tick label rotation
plt.grid(True, linestyle=':', color='gray', alpha=0.7)
plt.tick_params(axis='x', rotation=20)
plt.tick_params(axis='y', rotation=70)

plt.tight_layout(pad=0.3)

# Display the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/3e3fa84e1f6e40f1ac62373f79a0ff2b.png")
