import matplotlib.pyplot as plt
import numpy as np

# Set figure size
plt.figure(figsize=(9, 6))

# Define categories and their properties
categories = [
    ('standard', 'green', '+', False, False),
    ('upper limits', 'yellow', '+', True, False),
    ('lower limits', 'red', '+', False, True),
    ('upper and lower limits', 'orange', 'o', True, True),
    ('random', 'blue', 'o', False, False)
]

# Generate dummy data
x = np.linspace(0.5, 5.0, 10)
y_base = [0.5, 1.0, 1.5, 1.8, 2.7]  # Base y-values for each category

# Plot each category
for i, (label, color, marker, uplims, lolims) in enumerate(categories):
    y = np.full_like(x, y_base[i])
    # Generate random error values for demonstration
    yerr = np.random.uniform(0.1, 0.3, size=x.shape)
    xerr = np.random.uniform(0.1, 0.3, size=x.shape)
    
    # For 'random' category, vary the error bar directions
    if label == 'random':
        # Randomly assign upper/lower limits or symmetric
        uplims = np.random.choice([True, False], size=x.shape)
        lolims = np.random.choice([True, False], size=x.shape)
        # Also sometimes make it horizontal
        horizontal = np.random.choice([True, False], size=x.shape)
        for j in range(len(x)):
            if horizontal[j]:
                plt.errorbar(x[j], y[j], xerr=xerr[j], yerr=0, fmt=marker, color=color, 
                             capsize=5, ecolor=color, uplims=uplims[j], lolims=lolims[j])
            else:
                plt.errorbar(x[j], y[j], xerr=0, yerr=yerr[j], fmt=marker, color=color, 
                             capsize=5, ecolor=color, uplims=uplims[j], lolims=lolims[j])
    else:
        # For other categories, use consistent error bar types
        if label == 'upper and lower limits':
            # Horizontal error bars with arrows
            plt.errorbar(x, y, xerr=xerr, yerr=0, fmt=marker, color=color, 
                         capsize=5, ecolor=color, uplims=uplims, lolims=lolims)
        else:
            # Vertical error bars
            plt.errorbar(x, y, xerr=0, yerr=yerr, fmt=marker, color=color, 
                         capsize=5, ecolor=color, uplims=uplims, lolims=lolims)

# Customize plot
plt.title('Errorbar upper and lower limits')
plt.xlim(0, 5.5)
plt.ylim(0, 3.0)
plt.grid(True, alpha=0.3)

# Create legend
legend_elements = [
    plt.Line2D([0], [0], marker='+', color='green', linestyle='None', label='standard'),
    plt.Line2D([0], [0], marker='+', color='yellow', linestyle='None', label='upper limits'),
    plt.Line2D([0], [0], marker='+', color='red', linestyle='None', label='lower limits'),
    plt.Line2D([0], [0], marker='o', color='orange', linestyle='None', label='upper and lower limits'),
    plt.Line2D([0], [0], marker='o', color='blue', linestyle='None', label='random')
]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=True)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/errorpoint_6.png", bbox_inches="tight")