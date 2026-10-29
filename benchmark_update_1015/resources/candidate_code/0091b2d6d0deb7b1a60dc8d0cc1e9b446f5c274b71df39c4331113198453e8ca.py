import matplotlib.pyplot as plt
import numpy as np

# Set figure size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Dummy data for symmetric error bars (left plot)
time = np.arange(1, 11)
price_sym = np.array([161, 158, 157, 153, 152, 151, 150, 159, 155, 154])
error_sym = np.array([4, 3, 3, 3, 3, 3, 3, 4, 3, 3])

# Dummy data for asymmetric error bars (right plot)
price_asym = np.array([161, 158, 157, 153, 152, 151, 150, 159, 155, 154])
error_asym_upper = np.array([4, 3, 3, 3, 3, 3, 3, 4, 3, 3])
error_asym_lower = np.array([3, 2, 2, 2, 2, 2, 2, 3, 2, 2])

# Left subplot: Symmetric error bars
ax1.errorbar(time, price_sym, yerr=error_sym, fmt='^', color='orange', ecolor='gray', capsize=3, markersize=6)
ax1.set_title("Average Market Price with Symmetric Error Margins")
ax1.set_xlabel("Time (Months)")
ax1.set_ylabel("Average Price (USD)")
ax1.set_ylim(148, 162)
ax1.set_yticks(np.arange(148, 164, 2))
ax1.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5)
ax1.legend(['Speed Data'], loc='upper right', frameon=False, markerscale=1.2)

# Right subplot: Asymmetric error bars
ax2.errorbar(time, price_asym, yerr=[error_asym_lower, error_asym_upper], fmt='s', color='green', ecolor='red', capsize=3, markersize=6)
ax2.set_title("Average Market Price with Asymmetric Error Margins")
ax2.set_xlabel("Time (Months)")
ax2.set_ylabel("Average Price (USD)")
ax2.set_ylim(150, 162)
ax2.set_yticks(np.arange(150, 163, 2))
ax2.set_yticklabels([f"{i} × 10²" for i in np.arange(15, 16.3, 0.2)])
ax2.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5)
ax2.legend(['Speed Data'], loc='upper right', frameon=False, markerscale=1.2)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/errorpoint_18.png", bbox_inches="tight")