import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors


def radial_function(u, v):
    return 1.0 / (1 + 6 * ((u ** 2) + (v ** 2)))


def add_scale_label(axis_obj, label_text):
    axis_obj.text(-2.3, 1.4, label_text, fontsize=10, fontweight='bold')


N = 180
magnitude = 7
X_grid, Y_grid = np.mgrid[-2.8:2.8:complex(0, N), -1.8:1.8:complex(0, N)]
Z1_val = radial_function(X_grid + 0.4, Y_grid + 0.4)
Z2_val = radial_function(X_grid - 0.4, Y_grid - 0.4)
Z_result = magnitude * Z1_val - Z2_val

plot_options = {'cmap': 'PiYG', 'shading': 'nearest'}
boundary_width = 0.4

fig, axis_array = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 6))

pcm = axis_array[0].pcolormesh(X_grid, Y_grid, Z_result,
                       norm=colors.SymLogNorm(linthresh=boundary_width, linscale=1.2,
                                              vmin=-magnitude, vmax=magnitude, base=10),
                       **plot_options)
fig.colorbar(pcm, ax=axis_array[0], extend='both')
add_scale_label(axis_array[0], 'Logarithmic Scale')

pcm = axis_array[1].pcolormesh(X_grid, Y_grid, Z_result, vmin=-magnitude, vmax=magnitude,
                       **plot_options)
fig.colorbar(pcm, ax=axis_array[1], extend='both')
add_scale_label(axis_array[1], 'Linear Scale')

plt.subplots_adjust(left=0.07, right=0.86, top=0.99, bottom=0.08, hspace=0.08)
