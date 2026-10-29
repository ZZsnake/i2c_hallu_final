import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Set figure size
plt.figure(figsize=(12, 5))

# Define the grid for the Mandelbrot set
def mandelbrot(h, w, max_iter):
    y, x = np.ogrid[-1.4:1.4:h*1j, -2.2:0.7:w*1j]
    c = x + y*1j
    z = np.zeros(c.shape, dtype=complex)
    divtime = np.full(c.shape, max_iter, dtype=int)
    active = np.ones(c.shape, dtype=bool)
    for i in range(max_iter):
        z[active] = z[active]**2 + c[active]
        diverge = (z*np.conj(z) > 2**2) & active
        divtime[diverge] = i
        active[diverge] = False
    return divtime

# Create the overview plot
plt.subplot(1, 2, 1)
max_iter = 100
h, w = 400, 600
m = mandelbrot(h, w, max_iter)
plt.imshow(m, extent=[-2.2, 0.7, -1.4, 1.4], cmap='plasma', interpolation='bilinear', origin='lower')
plt.title('Mandelbrot Set Overview', fontweight='bold')
plt.xlabel('Real Component')
plt.ylabel('Imaginary Component')
selection_box = Rectangle((-2.2, -1.4), 2.9, 2.8, fill=False, edgecolor='tab:blue', linestyle='--', linewidth=1.5)
plt.gca().add_patch(selection_box)

# Create the detailed view plot
plt.subplot(1, 2, 2)
max_iter = 100
h, w = 400, 600
m = mandelbrot(h, w, max_iter)
plt.imshow(m, extent=[-2.2, 0.7, -1.4, 1.4], cmap='plasma', interpolation='bilinear', origin='lower')
plt.title('Detailed View', fontweight='bold')
plt.xlabel('Real Component')
plt.ylabel('Imaginary Component')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/c286af86780c4b21b27a9f8e30dfd330.png", bbox_inches="tight")
