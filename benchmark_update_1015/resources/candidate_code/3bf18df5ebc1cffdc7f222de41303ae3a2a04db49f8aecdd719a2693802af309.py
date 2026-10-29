import matplotlib.pyplot as plt
import numpy as np

# Set figure dimensions
plt.figure(figsize=(9, 6))

# Generate dummy data
x = np.arange(0, 6)
y_standard = [1.8, 1.7, 1.6, 1.5, 1.4, 1.3]
y_upper_limits = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7]
y_lower_limits = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
y_upper_and_lower_limits = [1.3, 1.2, 1.1, 1.0, 0.9, 0.8]
y_random = [2.7, 2.5, 2.3, 2.1, 1.9, 1.7]

# Plotting standard data points with error bars
plt.errorbar(x, y_standard, fmt='gX', label='standard')

# Plotting upper limits data points with error bars
plt.errorbar(x, y_upper_limits, fmt='yo', capsize=5, label='upper limits')

# Plotting lower limits data points with error bars
plt.errorbar(x, y_lower_limits, fmt='r>', capsize=5, label='lower limits')

# Plotting upper and lower limits data points with error bars
plt.errorbar(x, y_upper_and_lower_limits, fmt='o', capsize=5, label='upper and lower limits')

# Plotting random data points without error bars
plt.scatter(x, y_random, c='b', label='random')

# Adding legend
plt.legend()

# Setting axis labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Errorbar Upper and Lower Limits')

# Show plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/errorpoint_6.png", bbox_inches="tight")