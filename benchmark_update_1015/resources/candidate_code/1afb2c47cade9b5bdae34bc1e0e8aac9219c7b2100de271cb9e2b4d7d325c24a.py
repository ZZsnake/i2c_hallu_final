import matplotlib.pyplot as plt
import numpy as np

# Dummy data based on the image
authors = [
    "This work*", "Shah et al.", "Shah et al.", "Banerjee et al.", "Favale et al.",
    "Favale et al.", "Gomez-Valent", "Gomez-Valent", "Benisty et al.", "Benisty et al."
]
mb_values = [
    -19.353, -19.257, -19.394, -19.404, -19.314, -19.344, -19.362, -19.374, -19.220, -19.380
]
uncertainties = [
    (0.073, 0.078), (0.027, 0.027), (0.018, 0.017), (0.099, 0.104), (0.050, 0.108),
    (0.116, 0.090), (0.078, 0.067), (0.080, 0.080), (0.200, 0.200), (0.200, 0.200)
]
methods = [
    "CC+Pantheon+", "θ_BAO + r_CMB + d_Pantheon", "α_BAO + r_CMB + d_Pantheon",
    "CC+r_BAO+Pantheon+ (Ωk ≠ 0)", "CC+BAO+Pantheon+ (Ωk ≠ 0)", "CC+Pantheon+ (Ωk ≠ 0)",
    "CC+BAO+Pantheon (Ωk ≠ 0)", "CC+BAO+Pantheon", "α_BAO + r_SHOES + Pantheon",
    "α_BAO + r_CMB + d_Pantheon"
]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot horizontal error bars
for i, (author, mb, unc) in enumerate(zip(authors, mb_values, uncertainties)):
    ax.errorbar(mb, i, xerr=np.array(unc).reshape(2, 1), fmt='ko', capsize=5, label=author)

# Add shaded region and reference line
ax.axvspan(-19.25, -19.15, color='purple', alpha=0.3)
ax.axvline(-19.2, color='red', linestyle='--')

# Set labels and ticks
ax.set_xlabel('$M_B$', fontsize=14)
ax.set_yticks(np.arange(len(authors)))
ax.set_yticklabels(authors, fontsize=12)

# Add method descriptions
for i, method in enumerate(methods):
    ax.text(1.05, i, method, fontsize=10, transform=ax.get_yaxis_transform())

# Set limits and grid
ax.set_xlim(-19.6, -19.0)
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/errorpoint_3.png", bbox_inches="tight")