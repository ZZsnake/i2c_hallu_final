import matplotlib.pyplot as plt
import numpy as np

# Set figure size
plt.figure(figsize=(12, 5))

# Define the grid for the Mandelbrot set
def mandelbrot(h, w, max_iter):
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.5:w*1j]
    c = x + y*1j
    z = c
    divtime = np.zeros(z.shape, dtype=int)
    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        divtime[diverge] = i
        z[diverge] = 2  # To avoid overflow
    return divtime

# Create the overview plot
plt.subplot(1, 2, 1)
max_iter = 100
h, w = 400, 600
m = mandelbrot(h, w, max_iter)
plt.imshow(m.T, extent=[-2, 0.5, -1.0, 1.0], cmap='hot', interpolation='bilinear')
plt.title('Mandelbrot Set Overview')
plt.xlabel('Real Component')
plt.ylabel('Imaginary Component')
plt.colorbar()

# Create the detailed view plot
plt.subplot(1, 2, 2)
max_iter = 100
h, w = 400, 600
m = mandelbrot(h, w, max_iter)
plt.imshow(m.T, extent=[-2, 0.5, -1.0, 1.0], cmap='hot', interpolation='bilinear')
plt.title('Detailed View')
plt.xlabel('Real Component')
plt.ylabel('Imaginary Component')
plt.colorbar()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/c286af86780c4b21b27a9f8e30dfd330.png", bbox_inches="tight")