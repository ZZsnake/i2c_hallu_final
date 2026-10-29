import numpy as np
import matplotlib.pyplot as plt

# Set up the figure
plt.figure(figsize=(8, 6))

# Generate x values
x = np.linspace(0, 2 * np.pi, 1000)

# Original signal
f1 = np.sin(x)
plt.plot(x, f1, 'b', label='Original Signal')

# Distorted signal
f2 = 0.8 * np.sin(x + 0.5) + 0.2 * np.sin(3 * x)
plt.plot(x, f2, 'r', label='Distorted Signal')

# Labels and titles
plt.xlabel('Distance')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')

# Save the figure
plt.savefig('/home/zhoujiawei/work/image2codeBench/image2code/combine_data/graphic_py_dataset_output/images/AsymptoteGraphicPipeline_Electrical_Engineering_3-78.png')
plt.close()