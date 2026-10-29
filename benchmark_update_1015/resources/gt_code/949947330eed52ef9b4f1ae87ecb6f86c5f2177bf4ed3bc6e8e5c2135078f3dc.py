# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import numpy as np


# ===================
# Part 2: Data Preparation
# ===================

np.random.seed(0)
# example sports data
x = np.arange(1, 11, 1)
y = [160.7, 157.5, 156.8, 153.0, 152.3, 151.5, 150.2, 159.0, 155.2, 154.1,]  # average price in USD
error = np.array([2.0, 1.5, 1.2, 2.1, 2.0, 2.5, 2.2, 2.0, 2.3, 2.8])  # symmetric error
lower_error = 0.5 * error  # Adjusted asymmetric error
upper_error = error
asymmetric_error = [lower_error, upper_error]

title1 = "Average Market Price with Symmetric Error Margins"
title2 = "Average Market Price with Asymmetric Error Margins"
xlabel = "Time (Months)"
ylabel = "Average Price (USD)"
legend_label = "Speed Data"

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
fig, (ax0, ax1) = plt.subplots(figsize=(12, 6), ncols=2, sharex=True)

# Plot with symmetric error
ax0.errorbar(x, y, yerr=error, fmt="^", color="coral", ecolor="gray", elinewidth=2, capsize=5, label=legend_label)
ax0.set_title(title1, fontsize=14)
ax0.set_xlabel(xlabel, fontsize=12)
ax0.set_ylabel(ylabel, fontsize=12)
ax0.legend()

# Plot with asymmetric error
ax1.errorbar(x, y, yerr=asymmetric_error, fmt="s", color="#2ca02c", ecolor="#d62728", elinewidth=2, capsize=5, label=legend_label)
ax1.set_title(title2, fontsize=14)
ax1.set_xlabel(xlabel, fontsize=12)
ax1.set_yscale("log")
ax1.legend()

# ===================
# Part 4: Saving Output
# ===================
plt.tight_layout()
plt.savefig("errorpoint_18.pdf", bbox_inches="tight")