import matplotlib.pyplot as plt
import numpy as np

# Set figure size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Dummy data for symmetric error bars (left plot)
time = np.arange(1, 11)
price_sym = np.array([160.7, 157.5, 156.8, 153.0, 152.3, 151.5, 150.2, 159.0, 155.2, 154.1])
error_sym = np.array([2.0, 1.5, 1.2, 2.1, 2.0, 2.5, 2.2, 2.0, 2.3, 2.8])

# Dummy data for asymmetric error bars (right plot)
price_asym = np.array([160.7, 157.5, 156.8, 153.0, 152.3, 151.5, 150.2, 159.0, 155.2, 154.1])
error_asym_upper = np.array([2.0, 1.5, 1.2, 2.1, 2.0, 2.5, 2.2, 2.0, 2.3, 2.8])
error_asym_lower = np.array([1.0, 0.7, 0.6, 1.1, 1.0, 1.2, 1.1, 1.0, 1.1, 1.3])

# Left subplot: Symmetric error bars
ax1.errorbar(time, price_sym, yerr=error_sym, fmt='^', color='coral', ecolor='gray', capsize=3, markersize=6)
ax1.set_title("Average Market Price with Symmetric Error Margins")
ax1.set_xlabel("Time (Months)")
ax1.set_ylabel("Average Price (USD)")
ax1.set_ylim(147.25, 163.45)
ax1.set_yticks(np.arange(148, 164, 2))
ax1.legend(['Speed Data'], loc='upper right', frameon=True, markerscale=1.2)

# Right subplot: Asymmetric error bars
ax2.errorbar(time, price_asym, yerr=[error_asym_lower, error_asym_upper], fmt='s', color='green', ecolor='red', capsize=3, markersize=6)
ax2.set_title("Average Market Price with Asymmetric Error Margins")
ax2.set_xlabel("Time (Months)")
ax2.set_yscale('log')
ax2.set_ylim(148.4, 163.4)
ax2.set_yticks(np.arange(150, 164, 2))
ax2.legend(['Speed Data'], loc='upper right', frameon=True, markerscale=1.2)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/errorpoint_18.png", bbox_inches="tight")
