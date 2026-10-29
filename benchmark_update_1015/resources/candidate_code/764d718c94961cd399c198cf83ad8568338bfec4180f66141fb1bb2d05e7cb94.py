import numpy as np
import matplotlib.pyplot as plt

# Create dummy data
x = np.linspace(0, 6, 400)
original_signal = np.sin(x)
distorted_signal = np.sin(x) * (1 + 0.2 * np.sin(2 * x))

# Set up the plot
plt.figure(figsize=(8, 6))
plt.plot(x, original_signal, label='Original Signal', color='blue')
plt.plot(x, distorted_signal, label='Distorted Signal', color='red')

# Customize the plot
plt.xlabel('Distance')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.xlim(0, 6)
plt.ylim(-1, 1)

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/AsymptoteGraphicPipeline_Electrical_Engineering_3-78.png", bbox_inches="tight")