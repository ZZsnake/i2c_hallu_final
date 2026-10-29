import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.arange(0.5, 5.5, 0.5)
y = np.exp(-x)
y_standard = y
y_upper_limits = y + 1.0
y_lower_limits = y + 0.25
y_upper_lower_limits = y + 1.25
y_random = y + 2.1

# Error values
yerr_standard = np.full_like(x, 0.15, dtype=float)
yerr_upper_limits = np.full_like(x, 0.15, dtype=float)
yerr_lower_limits = np.full_like(x, 0.15, dtype=float)
yerr_upper_lower_limits = np.full_like(x, 0.15, dtype=float)
yerr_random = np.full_like(x, 0.20, dtype=float)

xerr = np.full_like(x, 0.20, dtype=float)

# Limit selections
upper_lower_uplims = [False, False, False, True, False, False, False, True, False, False]
upper_lower_lolims = [False, True, True, False, False, False, False, False, True, False]
random_uplims = [False, False, False, True, False, False, False, False, False, False]
random_lolims = [False, False, False, False, False, False, True, False, False, False]
random_xuplims = [False, False, False, True, False, False, False, True, False, False]
random_xlolims = [False, False, True, False, False, False, False, False, True, False]

# Create the plot
fig, ax = plt.subplots(figsize=(9, 6))

# Plotting each category
ax.errorbar(x, y_standard, xerr=xerr, yerr=yerr_standard, fmt='+', color='#98df8a', ecolor='#98df8a', label='standard')
ax.errorbar(x, y_upper_limits, xerr=xerr, yerr=yerr_upper_limits, uplims=True, fmt='none', color='#ffd966', ecolor='#ffd966', label='upper limits')
ax.errorbar(x, y_lower_limits, xerr=xerr, yerr=yerr_lower_limits, lolims=True, fmt='none', color='#e74c3c', ecolor='#e74c3c', label='lower limits')
ax.errorbar(x, y_upper_lower_limits, xerr=xerr, yerr=yerr_upper_lower_limits, uplims=upper_lower_uplims, lolims=upper_lower_lolims, fmt='o', color='#f08060', ecolor='#f08060', label='upper and lower limits')
ax.errorbar(x, y_random, xerr=xerr, yerr=yerr_random, uplims=random_uplims, lolims=random_lolims, xuplims=random_xuplims, xlolims=random_xlolims, fmt='o', color='#9aa6d9', ecolor='#9aa6d9', label='random')

# Adding title and legend
ax.set_title('Errorbar upper and lower limits')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=5)

# Display the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/errorpoint_6.png", bbox_inches="tight")
