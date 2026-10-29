import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# Create dummy data based on visual analysis of the plots
x_data = np.arange(1, 11)

# Data for the left plot (Symmetric Errors)
y_data_left = np.array([160.7, 157.5, 156.8, 153.0, 152.3, 151.5, 150.3, 155.8, 156.0, 154.2])
y_error_left = np.array([2.2, 1.3, 1.2, 2.0, 1.9, 1.7, 2.1, 5.2, 1.5, 2.8])

# Data for the right plot (Asymmetric Errors)
y_data_right = np.array([160.7, 157.5, 156.8, 153.0, 152.3, 151.5, 150.3, 159.0, 155.0, 153.8])
y_error_right_lower = np.array([2.7, 1.0, 0.6, 2.0, 1.8, 1.5, 2.1, 3.0, 2.0, 1.5])
y_error_right_upper = np.array([1.8, 1.4, 1.2, 1.8, 1.5, 1.0, 2.2, 2.0, 3.0, 3.2])
y_error_right = [y_error_right_lower, y_error_right_upper]

# Set up the figure and axes with specified dimensions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# --- Left Subplot: Symmetric Error Margins ---
ax1.errorbar(x_data, y_data_left, yerr=y_error_left, fmt='^', color='#ff7f0e', ecolor='gray', capsize=5, label='Speed Data')
ax1.set_title('Average Market Price with Symmetric Error Margins')
ax1.set_xlabel('Time (Months)')
ax1.set_ylabel('Average Price (USD)')
ax1.set_xticks(np.arange(2, 11, 2))
ax1.set_yticks(np.arange(148, 164, 2))
ax1.set_xlim(0.5, 10.5)
ax1.set_ylim(147, 163)
ax1.legend()

# --- Right Subplot: Asymmetric Error Margins ---
ax2.errorbar(x_data, y_data_right, yerr=y_error_right, fmt='s', color='#2ca02c', ecolor='#a52a2a', capsize=5, label='Speed Data')
ax2.set_title('Average Market Price with Asymmetric Error Margins')
ax2.set_xlabel('Time (Months)')
ax2.set_xticks(np.arange(2, 11, 2))
ax2.set_yticks(np.arange(150, 164, 2))
ax2.set_xlim(0.5, 10.5)
ax2.set_ylim(149, 163)

# Custom formatter for the right y-axis to match scientific notation style
def sci_formatter(y, pos):
    return f'{y/100:.2f} x $10^2$'

formatter = mticker.FuncFormatter(sci_formatter)
ax2.yaxis.set_major_formatter(formatter)

ax2.legend()

# Adjust layout to prevent labels from overlapping
plt.tight_layout()

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/errorpoint_18.png", bbox_inches="tight")