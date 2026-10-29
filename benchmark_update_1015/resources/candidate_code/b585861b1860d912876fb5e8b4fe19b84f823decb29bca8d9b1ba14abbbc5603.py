import numpy as np
import matplotlib.pyplot as plt

# Create figure and axes
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Common grid
x = np.linspace(-4, 4, 6)
y = np.linspace(-4, 4, 6)
X, Y = np.meshgrid(x, y)

# Vector field: radial, with magnitudes retained for barb feathers
U = 8 * X
V = 8 * Y

# Magnitude for coloring
R = np.hypot(U, V)
R[R == 0] = 1e-9  # avoid division by zero

# --- Top-left: dense, single color (dark blue) ---
ax1 = axes[0, 0]
q1 = ax1.barbs(
    X, Y, U, V,
    color='navy',
    length=7,
    barb_increments=dict(half=5, full=10, flag=100)
)
ax1.set_xlim(-4.5, 4.5)
ax1.set_ylim(-4.5, 4.5)

# --- Top-right: sparse, single color (crimson) ---
ax2 = axes[0, 1]
# Six arbitrary barb positions and values
Xs = np.array([-2.5, -1.85, -1.75, 0.45, 0.75, 1.25])
Ys = np.array([-0.55, 0.75, -0.65, 0.25, 1.75, -1.25])
Us = np.array([-18, 18, -8, 28, -72, 38])
Vs = np.array([18, -14, 36, -12, -62, -42])

q2 = ax2.barbs(
    Xs, Ys, Us, Vs,
    color='crimson',
    length=7
)
ax2.set_xlim(-3, 2)
ax2.set_ylim(-1.5, 2)

# --- Bottom-left: dense, colored by magnitude (viridis) ---
ax3 = axes[1, 0]
q3 = ax3.barbs(
    X, Y, U, V,
    R,                    # color array
    cmap='viridis',
    length=7,
    barb_increments=dict(half=5, full=10, flag=100)
)
ax3.set_xlim(-4.5, 4.5)
ax3.set_ylim(-4.5, 4.5)

# --- Bottom-right: sparse, custom barb colors ---
ax4 = axes[1, 1]
custom_colors = ['indigo', 'indigo', 'teal', 'indigo', 'teal', 'teal']
q4 = ax4.barbs(
    Xs, Ys, Us, Vs,
    barbcolor=custom_colors,
    flagcolor=custom_colors,
    length=7
)
ax4.set_xlim(-3, 2)
ax4.set_ylim(-1.5, 2)

# Remove titles/labels for clean look
for ax in axes.flat:
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/10cf129cb63c4f41bfb3ccb2e0d3818e.png", bbox_inches="tight")
