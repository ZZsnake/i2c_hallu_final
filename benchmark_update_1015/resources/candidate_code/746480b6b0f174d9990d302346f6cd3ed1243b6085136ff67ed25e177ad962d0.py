import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms

# Data for the plot
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

# Central values
central_values = [
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
]

# Asymmetric errors [lower, upper]
errors_lower = [0.078, 0.027, 0.017, 0.104, 0.108, 0.090, 0.067, 0.080, 0.200, 0.200]
errors_upper = [0.073, 0.028, 0.018, 0.099, 0.050, 0.116, 0.078, 0.080, 0.200, 0.200]

# Right side labels (model descriptions)
model_labels = [
    "CC+Pantheon+",
    r"$\theta_{BAO}+r_{CMB}+d_{Pantheon}$",
    r"$\alpha_{BAO}+r_{CMB}+d_{Pantheon}$",
    r"CC+$r_{BAO}$+Pantheon+ ($\Omega_k \neq 0$)",
    r"CC+BAO+Pantheon+ ($\Omega_k \neq 0$)",
    r"CC+Pantheon+ ($\Omega_k \neq 0$)",
    r"CC+BAO+Pantheon ($\Omega_k \neq 0$)",
    "CC+BAO+Pantheon",
    r"$\alpha_{BAO}+r_{SHOES}+Pantheon$",
    r"$\alpha_{BAO}+r_{CMB}+d_{Pantheon}$"
]

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Y positions (reversed to match the image)
y_positions = np.arange(len(studies))

# Plot error bars and points
for i, (y, central, err_low, err_up) in enumerate(zip(y_positions, central_values, errors_lower, errors_upper)):
    ax.errorbar(central, y, xerr=[[err_low], [err_up]],
                fmt='o', color='black', markersize=6,
                capsize=3, capthick=1.5, linewidth=1.5)

    # Add value annotations just above the markers/error bars
    label_text = rf"${central:.3f}^{{+{err_up:.3f}}}_{{-{err_low:.3f}}}$"
    ax.text(central, y - 0.12, label_text, ha='center', va='bottom', fontsize=9)

# Add vertical reference line and shaded region
reference_value = -19.22
ax.axvline(reference_value, color='red', linestyle='--', linewidth=1.5, zorder=0)
ax.axvspan(-19.25, -19.19, alpha=0.3, color='purple', zorder=0)

# Set y-axis labels
ax.set_yticks(y_positions)
ax.set_yticklabels(studies)

# Set x-axis
ax.set_xlabel(r'$M_B$', fontsize=12)
ax.set_xlim(-19.6, -19.0)

# Set y-axis limits
ax.set_ylim(-0.5, len(studies) - 0.5)

# Add right-side labels as plain text beside the plot border, without right-axis ticks
text_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
for y, label in zip(y_positions, model_labels):
    ax.text(1.002, y, label, transform=text_transform,
            ha='left', va='center', fontsize=11, clip_on=False)

# Keep the full plotting box, including the visible top border
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# Invert y-axis to match the image
ax.invert_yaxis()

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/errorpoint_3.png", bbox_inches="tight")
