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
    (0.073, 0.078), (0.028, 0.027), (0.018, 0.017), (0.099, 0.104), (0.050, 0.108),
    (0.116, 0.090), (0.078, 0.067), (0.080, 0.080), (0.200, 0.200), (0.200, 0.200)
]
methods = [
    "CC+Pantheon+", r"$\theta_{BAO}+r_{CMB}+d_{Pantheon}$", r"$\alpha_{BAO}+r_{CMB}+d_{Pantheon}$",
    r"CC+$r_{BAO}$+Pantheon+ ($\Omega_k \neq 0$)", r"CC+BAO+Pantheon+ ($\Omega_k \neq 0$)", r"CC+Pantheon+ ($\Omega_k \neq 0$)",
    r"CC+BAO+Pantheon ($\Omega_k \neq 0$)", "CC+BAO+Pantheon", r"$\alpha_{BAO}+r_{SHOES}+Pantheon$",
    r"$\alpha_{BAO}+r_{CMB}+d_{Pantheon}$"
]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Add shaded region and reference line
ax.axvspan(-19.252, -19.187, color='purple', alpha=0.3, zorder=0)
ax.axvline(-19.219, color='red', linestyle='--', zorder=1)

# Plot horizontal error bars and value annotations
for i, (author, mb, unc) in enumerate(zip(authors, mb_values, uncertainties)):
    plus, minus = unc
    ax.errorbar(mb, i, xerr=np.array([[minus], [plus]]), fmt='ko', capsize=5, label=author, zorder=3)
    ax.text(mb, i - 0.15, rf"${mb:.3f}^{{+{plus:.3f}}}_{{-{minus:.3f}}}$", fontsize=9,
            ha='center', va='bottom', zorder=4)

# Set labels and ticks
ax.set_xlabel('$M_B$', fontsize=14)
ax.set_yticks(np.arange(len(authors)))
ax.set_yticklabels(authors, fontsize=10)
ax.invert_yaxis()

# Add method descriptions
for i, method in enumerate(methods):
    ax.text(-19.0, i, method, fontsize=11, ha="left")

# Set limits
ax.set_xlim(-19.6, -19.0)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/errorpoint_3.png", bbox_inches="tight")
