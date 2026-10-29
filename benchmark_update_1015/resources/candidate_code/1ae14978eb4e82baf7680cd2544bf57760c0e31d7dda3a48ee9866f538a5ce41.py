import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Function to compute the Mandelbrot set
def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def create_mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    mandelbrot_set = np.empty((width, height))
    
    for i in range(width):
        for j in range(height):
            real_part = r1[i]
            imag_part = r2[j]
            c = complex(real_part, imag_part)
            mandelbrot_set[i, j] = mandelbrot(c, max_iter)
    
    return mandelbrot_set

# Parameters
xmin, xmax, ymin, ymax = -2.2, 0.7, -1.4, 1.4
width, height = 800, 600
max_iter = 60

# Create Mandelbrot sets
mandelbrot_overview = create_mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
mandelbrot_detail = create_mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Overview plot
axs[0].imshow(mandelbrot_overview.T, extent=[xmin, xmax, ymin, ymax], cmap='plasma', interpolation='none')
axs[0].add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='#0066CC', linestyle='--', linewidth=1.5))
axs[0].set_title('Mandelbrot Set Overview', fontweight='bold')
axs[0].set_xlabel('Real Component')
axs[0].set_ylabel('Imaginary Component')

# Detailed view plot
axs[1].imshow(mandelbrot_detail.T, extent=[xmin, xmax, ymin, ymax], cmap='plasma', interpolation='none')
axs[1].set_title('Detailed View', fontweight='bold')
axs[1].set_xlabel('Real Component')
axs[1].set_ylabel('Imaginary Component')
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/c286af86780c4b21b27a9f8e30dfd330.png")
