import matplotlib.pyplot as plt
import numpy as np

# Create shared x data
time = np.arange(0, 11, 1)

# Approximate y data based on visual inspection
growth = np.array([1.15, 0.78, 0.65, 0.63, 0.45, 0.12, 0.25, 0.12, 0.10, 0.13, 0.11])
decay = np.array([1.00, 0.35, 0.20, 0.15, 0.10, 0.15, 0.05, 0.02, -0.05, -0.10, -0.02])
oscillation = np.array([0.10, 0.85, 1.10, 0.05, -0.75, -1.00, -0.10, 0.90, 1.00, 0.70, -0.55])

# Create figure and subplots
fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

# Top subplot: Exponential Growth
axes[0].plot(time, growth, color='green', marker='o', linestyle='-', linewidth=1.5,
             markersize=5)
axes[0].set_title("Exponential Growth Over Time")
axes[0].set_ylabel("Value")
axes[0].set_ylim(-0.2, 1.3)
axes[0].legend(["Growth"], loc='upper center')

# Middle subplot: Exponential Decay
axes[1].plot(time, decay, color='red', marker='x', linestyle='-', linewidth=1.5,
             markersize=6)
axes[1].set_title("Exponential Decay Over Time")
axes[1].set_ylabel("Value")
axes[1].set_ylim(-0.2, 1.3)
axes[1].legend(["Decay"], loc='upper center')

# Bottom subplot: Oscillatory Behavior
axes[2].plot(time, oscillation, color='blue', marker='s', linestyle='-', linewidth=1.5,
             markersize=6)
axes[2].set_title("Oscillatory Behavior Over Time")
axes[2].set_ylabel("Value")
axes[2].set_xlabel("Time")
axes[2].set_ylim(-1.2, 1.2)
axes[2].legend(["Oscillation"], loc='upper center')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/line_49.png", bbox_inches="tight")