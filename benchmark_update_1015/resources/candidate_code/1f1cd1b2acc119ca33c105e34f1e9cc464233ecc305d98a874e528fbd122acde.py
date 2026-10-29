import matplotlib.pyplot as plt
import numpy as np

# Create the figure and a 3x1 grid of subplots
fig, axs = plt.subplots(3, 1, figsize=(6, 9))

# --- Data Extraction ---
# Common x-axis data
x_data = np.arange(11)

# Data for the first plot (Growth)
y_growth = [1.18, 0.76, 0.61, 0.59, 0.45, 0.09, 0.23, 0.08, 0.06, 0.09, 0.05]

# Data for the second plot (Decay)
y_decay = [1.06, 0.41, 0.14, 0.07, 0.03, 0.07, -0.01, 0.02, -0.04, -0.13, 0.03]

# Data for the third plot (Oscillation)
y_oscillation = [0.1, 0.75, 1.14, 0.0, -0.75, -1.0, -0.12, 0.80, 1.0, 0.45, -0.65]

# --- Plot 1: Exponential Growth Over Time ---
ax1 = axs[0]
ax1.plot(x_data, y_growth, color='green', marker='o', linestyle='-', label='Growth')
ax1.set_title('Exponential Growth Over Time', y=1.1)
ax1.set_ylabel('Value')
ax1.set_ylim(-0.2, 1.3)
ax1.set_xlim(0, 10)
ax1.set_yticks([-0.2, 0.3, 0.8, 1.3])
ax1.set_xticks(np.arange(0, 11, 2))
ax1.grid(True, linestyle='--', color='lightgray')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False)

# --- Plot 2: Exponential Decay Over Time ---
ax2 = axs[1]
ax2.plot(x_data, y_decay, color='red', marker='x', linestyle='-', label='Decay')
ax2.set_title('Exponential Decay Over Time', y=1.1)
ax2.set_ylabel('Value')
ax2.set_ylim(-0.2, 1.3)
ax2.set_xlim(0, 10)
ax2.set_yticks([-0.2, 0.3, 0.8, 1.3])
ax2.set_xticks(np.arange(0, 11, 2))
ax2.grid(True, linestyle='--', color='lightgray')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False)

# --- Plot 3: Oscillatory Behavior Over Time ---
ax3 = axs[2]
ax3.plot(x_data, y_oscillation, color='blue', marker='s', linestyle='-', label='Oscillation')
ax3.set_title('Oscillatory Behavior Over Time', y=1.1)
ax3.set_xlabel('Time')
ax3.set_ylabel('Value')
ax3.set_ylim(-1.2, 1.2)
ax3.set_xlim(0, 10)
ax3.set_yticks([-1.2, -1.0, 0.0, 1.0, 1.2])
ax3.set_xticks(np.arange(0, 11, 2))
ax3.grid(True, linestyle='--', color='lightgray')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False)

# Adjust layout to prevent titles and labels from overlapping
plt.tight_layout()

# Display the plot
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/line_49.png", bbox_inches="tight")
