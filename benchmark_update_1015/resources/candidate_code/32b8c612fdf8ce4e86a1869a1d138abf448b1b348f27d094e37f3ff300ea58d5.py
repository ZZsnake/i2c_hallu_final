import os
import numpy as np
import matplotlib.pyplot as plt

# Figure setup: 5 x 5 outer panels
fig, axes = plt.subplots(5, 5, figsize=(9, 9))
axes = axes.ravel()

# Parameter grids controlling how the small scribbles vary from panel to panel
a_vals = np.linspace(2, 6, 5)
b_vals = np.linspace(3, 9, 5)

t = np.linspace(0, 2 * np.pi, 900)
rng = np.random.default_rng(65)

idx = 0
for i, a0 in enumerate(a_vals):
    for j, b0 in enumerate(b_vals):
        ax = axes[idx]
        idx += 1

        # Each outer panel contains a 4 x 4 array of compact parametric curves.
        for rr in range(4):
            for cc in range(4):
                cx = cc + 0.5
                cy = 3.5 - rr

                # Frequencies and phases vary with the outer panel and inner position,
                # producing many compact, scribble-like parametric plots rather than one large curve.
                fx = a0 + 0.65 * cc + 0.35 * rr + 0.25 * j
                fy = b0 + 0.55 * rr + 0.30 * cc + 0.20 * i
                phase1 = 0.55 * i + 0.37 * j + 0.48 * rr + 0.29 * cc
                phase2 = 0.31 * i - 0.43 * j + 0.36 * cc + 0.27 * rr

                sx = np.sin(fx * t + phase1) + 0.42 * np.sin((fx + fy * 0.55) * t + phase2)
                sy = np.cos(fy * t + phase2) + 0.38 * np.sin((fy - fx * 0.25 + 1.7) * t + phase1)

                # Add a tiny deterministic perturbation so neighboring small plots are visibly distinct.
                sx += 0.10 * np.sin((2.3 + 0.2 * i + 0.1 * cc) * t + rng.uniform(0, 2 * np.pi))
                sy += 0.10 * np.cos((2.1 + 0.2 * j + 0.1 * rr) * t + rng.uniform(0, 2 * np.pi))

                sx = sx / np.max(np.abs(sx))
                sy = sy / np.max(np.abs(sy))

                x = cx + 0.43 * sx
                y = cy + 0.43 * sy
                ax.plot(x, y, color='#1f77b4', linewidth=1.35)

        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Fixed square panel limits holding the 4 x 4 inner array
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_aspect('equal', adjustable='box')

# Adjust layout to get small but visible gaps between subplots
plt.subplots_adjust(wspace=0.15, hspace=0.15)
os.makedirs(r"eval_workspace_gpt5/GPT_5_1/python", exist_ok=True)
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/65d7cfa7918d48e09df30bf0717d65a6.png", bbox_inches="tight")
