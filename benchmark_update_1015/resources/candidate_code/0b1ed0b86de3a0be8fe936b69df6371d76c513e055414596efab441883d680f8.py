import matplotlib.pyplot as plt

# Data preparation
y_labels = [
    'This work*', 'Shah et al.', 'Shah et al.', 'Banerjee et al.',
    'Favale et al.', 'Favale et al.', 'Gomez-Valent', 'Gomez-Valent',
    'Benisty et al.', 'Benisty et al.'
]

x_values = [
    -19.353, -19.257, -19.394, -19.404, -19.314,
    -19.344, -19.362, -19.374, -19.220, -19.380
]

lower_errors = [
    0.078, 0.027, 0.017, 0.104, 0.108,
    0.090, 0.067, 0.080, 0.200, 0.200
]

upper_errors = [
    0.073, 0.028, 0.018, 0.099, 0.050,
    0.116, 0.078, 0.080, 0.200, 0.200
]

right_labels = [
    'CC+Pantheon+',
    r'$\theta_{BAO} + r_{CMB} + d_{Pantheon}$',
    r'$\alpha_{BAO} + r_{CMB} + d_{Pantheon}$',
    'CC+$r_{BAO}$+Pantheon+ (Ωk ≠ 0)',
    'CC+BAO+Pantheon+ (Ωk ≠ 0)',
    'CC+Pantheon+ (Ωk ≠ 0)',
    'CC+BAO+Pantheon (Ωk ≠ 0)',
    'CC+BAO+Pantheon',
    r'$\alpha_{BAO} + r_{SHOES} + Pantheon$',
    r'$\alpha_{BAO} + r_{CMB} + d_{Pantheon}$'
]

y_positions = list(range(len(y_labels)))

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot horizontal error bars
ax.errorbar(x_values, y_positions, xerr=[lower_errors, upper_errors],
            fmt='o', color='black', capsize=5, markersize=5)

# Add numeric value/error annotations above each point
for x, y, lo, up in zip(x_values, y_positions, lower_errors, upper_errors):
    ax.text(x, y - 0.18, rf'${x:.3f}^{{+{up:.3f}}}_{{-{lo:.3f}}}$',
            ha='center', va='center', fontsize=8, color='black')

# Add vertical shaded region and red dashed line
ax.axvspan(-19.252, -19.187, color='purple', alpha=0.3)
ax.axvline(-19.219, color='red', linestyle='--')

# Configure axes
ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels)
ax.invert_yaxis()  # Top-to-bottom ordering
ax.set_xlim(-19.6, -19.0)
ax.set_xlabel('$M_B$', fontsize=12)

# Add right-side annotations
for i, label in enumerate(right_labels):
    ax.text(1.002, i, label, transform=ax.get_yaxis_transform(),
            verticalalignment='center', horizontalalignment='left',
            fontsize=11, clip_on=False)

# Adjust layout to accommodate right labels
plt.tight_layout()

# Display plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/errorpoint_3.png", bbox_inches="tight")
