import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the plots
# Each dictionary contains the label, coordinates for both plots, color, and text alignment hints.
data = [
    {'label': 'MAD+judge', 'token_cost': 35, 'tflops': 2.5e6, 'accuracy': 68, 'color': '#b22222'},
    {'label': 'SPP', 'token_cost': 15, 'tflops': 1.5e6, 'accuracy': 68, 'color': '#800080'},
    {'label': 'ToT', 'token_cost': 30, 'tflops': 2.0e6, 'accuracy': 66.0, 'color': 'red'},
    {'label': 'CoT(maj@5)', 'token_cost': 10, 'tflops': 0.5e6, 'accuracy': 66.0, 'color': 'orange'},
    {'label': 'DefInt', 'token_cost': 10, 'tflops': 0.5e6, 'accuracy': 62.0, 'color': '#ffb6c1'},
    {'label': 'CoT(maj@1)', 'token_cost': 20, 'tflops': 1.0e6, 'accuracy': 64.0, 'color': 'blue'},
    {'label': 'Self-refine', 'token_cost': 45, 'tflops': 2.5e6, 'accuracy': 62, 'color': 'green'}
]

# Create the figure and a 1x2 grid of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# --- Plot 1: Accuracy versus token cost ---
ax1.set_title('Logic Grid Puzzle(Accuracy versus token cost)')
ax1.set_xlabel('Token cost($)')
ax1.set_ylabel('Accuracy(%)')

# Set axis limits and ticks
ax1.set_xlim(50, 0)
ax1.set_ylim(60, 70)
ax1.set_xticks(np.arange(0, 51, 10))
ax1.set_yticks(np.arange(60, 71, 2))

# Plot points and annotations
for point in data:
    ax1.scatter(point['token_cost'], 65.0 if point['label'] == 'DefInt' else point['accuracy'], color=point['color'], s=50)
    # Adjust text position for clarity
    x_offset = -1 if point['label'] in ['SPP', 'DefInt'] else 1
    ha = 'right' if x_offset < 0 else 'left'
    ax1.text(point['token_cost'] + x_offset, (65.0 if point['label'] == 'DefInt' else point['accuracy']) + 0.1, point['label'], ha=ha, va='bottom')

# --- Plot 2: Accuracy versus TFLOPS ---
ax2.set_title('Logic Grid Puzzle(Accuracy versus TFLOPS)')
ax2.set_xlabel('TFLOPS')
ax2.set_ylabel('Accuracy(%)')

# Set axis limits and ticks
ax2.set_xlim(3.0e6, 0)
ax2.set_ylim(60, 70)
ax2.set_yticks(np.arange(60, 71, 2))
xticks_val = np.arange(0, 3.1e6, 0.5e6)
xtick_labels = [f'{val/1e6:.1f}' for val in xticks_val]
ax2.set_xticks(xticks_val)
ax2.set_xticklabels(xtick_labels)

# Add the '1e6' label below the x-axis
ax2.text(1.0, -0.08, '1e6', transform=ax2.transAxes, ha='right', va='top')

# Plot points and annotations
for point in data:
    ax2.scatter(point['tflops'], point['accuracy'], color=point['color'], s=50)
    # Adjust text position for clarity
    x_offset = -0.05e6 if point['label'] in ['SPP', 'DefInt'] else 0.05e6
    ha = 'right' if x_offset < 0 else 'left'
    ax2.text(point['tflops'] + x_offset, point['accuracy'] + 0.1, point['label'], ha=ha, va='bottom')

# Adjust layout to prevent labels from overlapping
plt.tight_layout()

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/scatter_15.png", bbox_inches="tight")
