import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms

def build_matrix_data():
    delta = 0.3
    x = y = np.arange(-4.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1.5)**2 - (Y - 1.5)**2)
    return Z1 - Z2 * 1.2

def draw_transformed_matrix(ax, matrix_data, transformation):
    img = ax.imshow(matrix_data, interpolation='bilinear',
                   origin='lower',
                   extent=[-3, 5, -4, 3], clip_on=True)
    
    combined_transform = transformation + ax.transData
    img.set_transform(combined_transform)

    x_min, x_max, y_min, y_max = img.get_extent()
    ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], "c--",
            linewidth=1.5, transform=combined_transform)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)

fig, axes_grid = plt.subplots(2, 2, figsize=(10, 8))
matrix = build_matrix_data()

draw_transformed_matrix(axes_grid[0, 0], matrix, mtransforms.Affine2D().rotate_deg(45))
draw_transformed_matrix(axes_grid[0, 1], matrix, mtransforms.Affine2D().skew_deg(25, 20))
draw_transformed_matrix(axes_grid[1, 0], matrix, mtransforms.Affine2D().scale(-1.2, 0.6))
draw_transformed_matrix(axes_grid[1, 1], matrix, mtransforms.Affine2D().
        rotate_deg(40).skew_deg(20, 10).scale(-1.1, 0.7).translate(0.8, -1.2))

plt.tight_layout()
plt.show()
