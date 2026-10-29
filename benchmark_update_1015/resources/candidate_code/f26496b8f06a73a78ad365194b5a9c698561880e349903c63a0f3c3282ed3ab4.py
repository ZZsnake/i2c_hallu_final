import matplotlib.pyplot as plt
import numpy as np

# Data approximated from the figure
studies = [
    "This work*",
    "Shah et al.",
    "Shah et al.",
    "Banerjee et al.",
    "Favale et al.",
    "Favale et al.",
    "Gomez-Valent",
    "Gomez-Valent",
    "Benisty et al.",
    "Benisty et al."
]

methods = [
    "CC+Pantheon+",
    r"$\theta_{BAO} + r_{CMB} + d_{Pantheon}$",
    r"$\alpha_{BAO} + r_{CMB} + d_{Pantheon}$",
    r"CC+$\Gamma_{BAO}$+Pantheon+ ($\Omega_k \neq 0$)",
    r"CC+BAO+Pantheon+ ($\Omega_k \neq 0$)",
    r"CC+Pantheon+ ($\Omega_k \neq 0$)",
    r"CC+BAO+Pantheon ($\Omega_k \neq 0$)",
    r"CC+BAO+Pantheon",
    r"$\alpha_{BAO} + r_{SHOES} +$ Pantheon",
    r"$\alpha_{BAO} + r_{CMB} + d_{Pantheon}$"
]

# Best-fit MB values (x) and asymmetric errors (left, right)
x_vals = np.array([
    -19.353,
    -19.257,
    -19.394,
    -19.404,
    -19.314,
    -19.344,
    -19.362,
    -19.374,
    -19.220,
    -19.380
])

err_left = np.array([
    0.078,
    0.027,
    0.017,
    0.104,
    0.108,
    0.090,
    0.067,
    0.080,
    0.020,
    0.200
])

err_right = np.array([
    0.073,
    0.028,
    0.018,
    0.099,
    0.050,
    0.116,
    0.078,
    0.080,
    0.020,
    0.200
])

# Y positions (top to bottom)
y_pos = np.arange(len(studies))[::-1]  # reverse so first is at top

fig, ax = plt.subplots(figsize=(10, 8))

# Vertical reference band and line
band_center = -19.22
band_width = 0.06
band_xmin = band_center - band_width / 2
band_xmax = band_center + band_width / 2

ax.axvspan(band_xmin, band_xmax, color='#c17fd3', alpha=0.4, zorder=0)
ax.axvline(band_center, color='red', linestyle='--', linewidth=1.5, zorder=1)

# Plot error bars with asymmetric errors
ax.errorbar(
    x_vals, y_pos,
    xerr=[err_left, err_right],
    fmt='o', color='black',
    ecolor='black',
    elinewidth=1.5,
    capsize=0,
    markersize=5,
    linestyle='none',
    zorder=2
)

# Numeric labels near each point
for x, y, el, er in zip(x_vals, y_pos, err_left, err_right):
    label = rf"$-19.{str(x)[-3:]}^{{+{er:.3f}}}_{{-{el:.3f}}}$"
    ax.text(
        x, y + 0.25, label,
        ha='center', va='bottom',
        fontsize=8
    )

# Left-side study labels (y-axis)
ax.set_yticks(y_pos)
ax.set_yticklabels(studies)

# Right-side method labels
x_text = -18.98  # slightly to the right of axis max
for y, method in zip(y_pos, methods):
    ax.text(
        x_text, y,
        method,
        ha='left', va='center',
        fontsize=9
    )

# Axis limits
ax.set_xlim(-19.6, -19.0)
ax.set_ylim(-1, len(studies))

# Axis labels
ax.set_xlabel(r"$M_B$", fontsize=12)

# Style adjustments
ax.tick_params(axis='both', which='both', direction='out')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/errorpoint_3.png", bbox_inches="tight")