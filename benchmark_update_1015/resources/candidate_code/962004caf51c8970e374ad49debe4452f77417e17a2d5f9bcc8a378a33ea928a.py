import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 2 * np.pi, 400)
y1 = np.sin(x)  # Original Signal
y2 = 0.8 * np.sin(x + 0.5) + 0.2 * np.sin(3 * x)  # Distorted Signal

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(x, y1, 'b-', label='Original Signal')
plt.plot(x, y2, 'r-', label='Distorted Signal')

# Set labels and title
plt.xlabel('Distance')
plt.ylabel('Amplitude')

# Add legend
plt.legend()

# Show plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/AsymptoteGraphicPipeline_Electrical_Engineering_3-78.png", bbox_inches="tight")
