import matplotlib.pyplot as plt
import numpy as np

# Data setup
y_labels = [
    "This work*",
    "Shah et al.",
    "Shah et al.",
    "Banerjee et al.",
    "Favale et al.",
    "Favale et al.",
    "Gomez-Valent",
    "Gomez-Valent",
    "Benisty et al.",
    "Benisty et al.",
]

x_vals = np.array([
    -19.353,
    -19.257,
    -19.390,
    -19.404,
    -19.314,
    -19.344,
    -19.362,
    -19.374,
    -19.220,
    -19.380,
])

# Asymmetric errors: [negative_error, positive_error]
x_err_neg = np.array([0.078, 0.027, 0.017, 0.104, 0.108, 0.090, 0.067, 0.080, 0.020, 0.200])
x_err_pos = np.array([0.073, 0.028, 0.018, 0.099, 0.050, 0.116, 0.078, 0.080, 0.020, 0.200])

# Right side method labels (using LaTeX-style strings)
method_labels = [
    r"CC+Pantheon+",
    r"$\theta_{\mathrm{BAO}} + r_{\mathrm{CMB}} + d_{\mathrm{Pantheon}}$",
    r"$\alpha_{\mathrm{BAO}} + r_{\mathrm{CMB}} + d_{\mathrm{Pantheon}}$",
    r"CC+$r_{\mathrm{BAO}}$+Pantheon+ $(\Omega_k \neq 0)$",
    r"CC+BAO+Pantheon+ $(\Omega_k \neq 0)$",
    r"CC+Pantheon+ $(\Omega_k \neq 0)$",
    r"CC+BAO+Pantheon $(\Omega_k \neq 0)$",
    r"CC+BAO+Pantheon",
    r"$\alpha_{\mathrm{BAO}} + r_{\mathrm{SHOES}} + \mathrm{Pantheon}$",
    r"$\alpha_{\mathrm{BAO}} + r_{\mathrm{CMB}} + d_{\mathrm{Pantheon}}$",
]

# Numerical strings for labels above the points
num_labels = [
    r"$-19.353^{+0.073}_{-0.078}$",
    r"$-19.257^{+0.028}_{-0.027}$",
    r"$-19.390^{+0.018}_{-0.017}$",
    r"$-19.404^{+0.099}_{-0.104}$",
    r"$-19.314^{+0.050}_{-0.108}$",
    r"$-19.344^{+0.116}_{-0.090}$",
    r"$-19.362^{+0.078}_{-0.067}$",
    r"$-19.374^{+0.080}_{-0.080}$",
    r"$-19.220^{+0.020}_{-0.020}$",
    r"$-19.380^{+0.200}_{-0.200}$",
]

# Y positions (top to bottom)
y_pos = np.arange(len(y_labels))[::-1]  # 9..0 top to bottom

fig, ax = plt.subplots(figsize=(10, 8))

# Shaded vertical band around -19.22
band_center = -19.22
band_halfwidth = 0.02  # approx: from -19.24 to -19.20
ax.axvspan(band_center - band_halfwidth, band_center + band_halfwidth,
           color='purple', alpha=0.3, zorder=0)

# Vertical dashed reference line at -19.22
ax.axvline(band_center, color='red', linestyle='--', linewidth=2, zorder=1)

# Errorbar plot
ax.errorbar(
    x_vals, y_pos,
    xerr=[x_err_neg, x_err_pos],
    fmt='o',
    color='black',
    ecolor='black',
    elinewidth=2,
    capsize=0,
    markersize=6,
    zorder=2,
)

# Add numerical labels above each point
for x, y, label in zip(x_vals, y_pos, num_labels):
    ax.text(x, y + 0.18, label, ha='center', va='bottom', fontsize=9)

# Set y-axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=10)

# X-axis settings
ax.set_xlabel(r"$M_B$", fontsize=12)
ax.set_xlim(-19.6, -19.0)
ax.set_xticks(np.linspace(-19.6, -19.0, 7))

# Add right-side method labels as text annotations
# Place them slightly to the right of the plotting range in axis fraction coordinates
for y, method in zip(y_pos, method_labels):
    ax.text(1.02, (y - y_pos.min()) / (y_pos.max() - y_pos.min()),
            method, transform=ax.transAxes,
            va='center', ha='left', fontsize=9)

# Adjust plot aesthetics
ax.invert_yaxis()  # optional: to match visual orientation if needed
ax.tick_params(axis='y', length=0)
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/errorpoint_3.png", bbox_inches="tight")