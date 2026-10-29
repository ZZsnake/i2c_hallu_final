import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

class FractalVisualizer:
    def __init__(self, resolution=400, iterations=60, radius=2.5, exponent=2):
        self.resolution = resolution
        self.iterations = iterations
        self.radius = radius
        self.exponent = exponent

    def generate_fractal(self, x_range, y_range):
        x_coords = np.linspace(*x_range, self.resolution)
        y_coords = np.linspace(*y_range, self.resolution).reshape(-1, 1)
        complex_grid = x_coords + 1.0j * y_coords
        divergence_map = np.zeros((self.resolution, self.resolution))
        z_values = np.zeros(divergence_map.shape, dtype=complex)
        active_mask = np.ones(divergence_map.shape, dtype=bool)
        
        for iteration in range(self.iterations):
            z_values[active_mask] = z_values[active_mask]**self.exponent + complex_grid[active_mask]
            active_mask = (np.abs(z_values) < self.radius)
            divergence_map += active_mask
            
        return divergence_map

    def update_visualization(self, axes):
        axes.set_autoscale_on(False)
        self.resolution = axes.patch.get_window_extent().size.round().astype(int)[0]
        axes.images[-1].set(data=self.generate_fractal(axes.get_xlim(), axes.get_ylim()),
                           extent=(*axes.get_xlim(), *axes.get_ylim()))
        axes.figure.canvas.draw_idle()

visualizer = FractalVisualizer()

figure, (main_axes, detail_axes) = plt.subplots(1, 2, figsize=(12, 5))
detail_axes.imshow([[0]], origin="lower", cmap='plasma')
detail_axes.set_title("Detailed View", fontsize=12, fontweight='bold')

bounding_box = Rectangle([0, 0], 0, 0, facecolor="none", 
                        edgecolor="#0066CC", linewidth=1.5, linestyle="--")
main_axes.add_patch(bounding_box)

def update_bounding_box(rect, axes):
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    rect.set_bounds((x_min, y_min, x_max - x_min, y_max - y_min))
    axes.figure.canvas.draw_idle()

detail_axes.callbacks.connect("xlim_changed", lambda ax: update_bounding_box(bounding_box, ax))
detail_axes.callbacks.connect("ylim_changed", lambda ax: update_bounding_box(bounding_box, ax))

detail_axes.callbacks.connect("xlim_changed", visualizer.update_visualization)
detail_axes.callbacks.connect("ylim_changed", visualizer.update_visualization)

detail_axes.set(xlim=(-2.2, 0.7), ylim=(-1.4, 1.4))
image_data = detail_axes.images[0]
detail_axes.images[0].set(clim=(image_data.get_array().min(), image_data.get_array().max()))
main_axes.imshow(image_data.get_array(), extent=image_data.get_extent(), 
                origin="lower", cmap='plasma')

main_axes.set_xlabel("Real Component", fontsize=10)
main_axes.set_ylabel("Imaginary Component", fontsize=10)
detail_axes.set_xlabel("Real Component", fontsize=10)
detail_axes.set_ylabel("Imaginary Component", fontsize=10)
main_axes.set_title("Mandelbrot Set Overview", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()