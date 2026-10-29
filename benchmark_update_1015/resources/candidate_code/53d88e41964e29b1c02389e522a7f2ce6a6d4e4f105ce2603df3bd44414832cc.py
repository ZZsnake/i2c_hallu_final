import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Data (approximate to the figure)
months = np.arange(1, 11)

prices = np.array([160.7, 157.5, 156.9, 153.0, 152.3,
                   151.5, 150.1, 159.0, 155.1, 154.0])

# Symmetric error margins (left subplot)
sym_err = np.array([2.0, 1.5, 1.1, 2.1, 2.0,
                    2.5, 2.3, 2.1, 2.4, 2.9])

# Asymmetric error margins (right subplot)
# lower and upper errors
lower_err = np.array([1.5, 1.0, 0.8, 1.3, 1.2,
                      1.4, 1.5, 1.0, 1.2, 1.1])
upper_err = np.array([2.0, 1.7, 1.4, 2.0, 1.9,
                      1.8, 2.2, 2.1, 2.3, 2.0])

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

# -----------------------------------
# Left subplot: symmetric errorbars
# -----------------------------------
ax1.errorbar(
    months, prices,
    yerr=sym_err,
    fmt='^',                      # marker style
    color='orange',               # marker (and line) color
    ecolor='grey',                # error bar color
    elinewidth=1,
    capsize=4,
    linestyle='none',             # no connecting line
    label='Speed Data'
)

ax1.set_title('Average Market Price with Symmetric Error Margins')
ax1.set_xlabel('Time (Months)')
ax1.set_ylabel('Average Price (USD)')
ax1.set_xticks(months)
ax1.set_xlim(0.5, 10.5)
ax1.set_ylim(148, 163.5)
ax1.legend(loc='upper right', frameon=True)

# -----------------------------------
# Right subplot: asymmetric errorbars
# -----------------------------------
ax2.errorbar(
    months, prices,
    yerr=[lower_err, upper_err],
    fmt='s',
    color='green',
    ecolor='red',
    elinewidth=1,
    capsize=4,
    linestyle='none',
    label='Speed Data'
)

ax2.set_title('Average Market Price with Asymmetric Error Margins')
ax2.set_xlabel('Time (Months)')
ax2.set_xticks(months)
ax2.set_xlim(0.5, 10.5)
ax2.set_ylim(148, 163.5)
ax2.legend(loc='upper right', frameon=True)

# Use scientific notation on the y-axis (10^2 style)
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((2, 2))  # force 10^2 scaling
ax2.yaxis.set_major_formatter(formatter)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(2, 2))

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/errorpoint_18.png", bbox_inches="tight")