import numpy as np
import matplotlib.pyplot as plt

# Create dummy data
x = np.linspace(0, 2 * np.pi, 400)
original_signal = np.sin(x)

# Smooth distorted waveform reconstructed from the reference curve
control_x = np.array([0.0, 0.25, 0.55, 0.75, 1.0, 1.3, 1.7, 2.2, 2.5, 2.8,
                      3.2, 3.5, 3.85, 4.2, 4.6, 4.9, 5.3, 5.7, 6.0, 2 * np.pi])
control_y = np.array([0.38, 0.67, 0.89, 0.92, 0.82, 0.62, 0.46, 0.42, 0.33, 0.05,
                      -0.45, -0.80, -0.92, -0.78, -0.52, -0.44, -0.40, -0.25, 0.10, 0.38])

def catmull_rom_interpolate(x_new, xp, yp):
    y_new = np.empty_like(x_new)
    for i, xv in enumerate(x_new):
        j = np.searchsorted(xp, xv) - 1
        j = np.clip(j, 0, len(xp) - 2)
        x0, x1 = xp[j], xp[j + 1]
        t = (xv - x0) / (x1 - x0)
        p0 = yp[max(j - 1, 0)]
        p1 = yp[j]
        p2 = yp[j + 1]
        p3 = yp[min(j + 2, len(yp) - 1)]
        y_new[i] = 0.5 * ((2 * p1) + (-p0 + p2) * t +
                          (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
    return y_new

distorted_signal = catmull_rom_interpolate(x, control_x, control_y)

# Set up the plot
plt.figure(figsize=(7.2, 5.3))
plt.plot(x, original_signal, label='Original Signal', color='blue')
plt.plot(x, distorted_signal, label='Distorted Signal', color='red')

# Customize the plot
plt.xlabel('Distance')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.tight_layout()
# Display the plot
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/AsymptoteGraphicPipeline_Electrical_Engineering_3-78.png")
