import matplotlib.pyplot as plt
import numpy as np

# Create dummy data to match the visual trends
x = np.linspace(0, 2 * np.pi, 500)
original_signal = np.sin(x)  # Blue line: sine wave
distorted_signal = 0.8 * np.sin(x + 0.5) + 0.2 * np.sin(3 * x)  # Red line: distorted version

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the signals
plt.plot(x, original_signal, color='blue', label='Original Signal', linewidth=2)
plt.plot(x, distorted_signal, color='red', label='Distorted Signal', linewidth=2)

# Set axis labels
plt.xlabel('Distance')
plt.ylabel('Amplitude')

# Add legend in top-right
plt.legend(loc='upper right')

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/AsymptoteGraphicPipeline_Electrical_Engineering_3-78.png")
