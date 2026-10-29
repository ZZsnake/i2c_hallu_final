import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 4, 6)
X, Y = np.meshgrid(x, x)
U, V = 8 * X, 8 * Y

data = [(-2.0, 0.7, -8, -8),
        (1.2, -1.3, -38, 38),
        (-2.5, -0.8, 15, -15),
        (0.8, 1.8, 70, 70),
        (0.6, 0.3, 20, 12),
        (-1.8, -0.6, -8, 35)]

data = np.array(data, dtype=[('x', np.float32), ('y', np.float32),
                             ('u', np.float32), ('v', np.float32)])

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Wind velocity field - uniform grid
axs[0, 0].barbs(X, Y, U, V, color='navy', linewidth=1.2)

# Force vectors - arbitrary positions with middle pivot
axs[0, 1].barbs(
    data['x'], data['y'], data['u'], data['v'], length=9, pivot='middle',
    color='crimson', linewidth=1.5)

# Colormapped electric field magnitude
axs[1, 0].barbs(
    X, Y, U, V, np.sqrt(U ** 2 + V ** 2), fill_empty=True, rounding=False,
    sizes=dict(emptybarb=0.3, spacing=0.25, height=0.35),
    cmap='viridis', linewidth=1.3)

# Magnetic field vectors with custom colors and increments
axs[1, 1].barbs(data['x'], data['y'], data['u'], data['v'], flagcolor='darkorange',
                 barbcolor=['indigo', 'teal'], flip_barb=True,
                 barb_increments=dict(half=12, full=25, flag=120),
                 linewidth=1.4)

plt.tight_layout()
plt.show()