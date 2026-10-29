import matplotlib.pyplot as plt
import numpy as np

# Data for both plots
methods = [
    "CoT(maj@1)",
    "Self-refine",
    "ToT",
    "MAD+judge",
    "SPP",
    "CoT(maj@5)",
    "DefInt"
]

# Colors approximated from the figure
colors = {
    "CoT(maj@1)": "blue",
    "Self-refine": "green",
    "ToT": "red",
    "MAD+judge": "salmon",
    "SPP": "purple",
    "CoT(maj@5)": "orange",
    "DefInt": "pink",
}

# Token cost data (left subplot)
token_cost = {
    "CoT(maj@1)": 20,
    "Self-refine": 35,
    "ToT": 30,
    "MAD+judge": 35,
    "SPP": 25,
    "CoT(maj@5)": 15,
    "DefInt": 10
}

accuracy_token = {
    "CoT(maj@1)": 64.0,
    "Self-refine": 62.0,
    "ToT": 66.0,
    "MAD+judge": 68.0,
    "SPP": 68.0,
    "CoT(maj@5)": 66.0,
    "DefInt": 65.0
}

# TFLOPS data (right subplot)
tflops = {
    "CoT(maj@1)": 1.0e6,
    "Self-refine": 2.5e5,
    "ToT": 4.0e5,
    "MAD+judge": 3.0e5,
    "SPP": 1.5e5,
    "CoT(maj@5)": 5.0e5,
    "DefInt": 6.0e5
}

accuracy_tflops = {
    "CoT(maj@1)": 64.0,
    "Self-refine": 62.0,
    "ToT": 66.0,
    "MAD+judge": 68.0,
    "SPP": 68.0,
    "CoT(maj@5)": 66.0,
    "DefInt": 62.0
}

# Create figure and subplots with specified dimensions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# ---- Left subplot: Accuracy vs Token cost ----
for m in methods:
    x = token_cost[m]
    y = accuracy_token[m]
    ax1.scatter(x, y, color=colors[m], s=60)
    # Slight upward offset for text label
    ax1.text(x, y + 0.3, m, fontsize=8, ha='center')

ax1.set_title("Logic Grid Puzzle(Accuracy versus token cost)")
ax1.set_xlabel("Token cost($)")
ax1.set_ylabel("Accuracy(%)")

# Match y-limits and ticks approximately
ax1.set_ylim(60, 70)
ax1.set_yticks(np.arange(60, 71, 2))

# Reverse x-axis as in the figure (higher cost left, lower right)
ax1.set_xlim(50, 0)
ax1.set_xticks([50, 40, 30, 20, 10, 0])

# ---- Right subplot: Accuracy vs TFLOPS ----
for m in methods:
    x = tflops[m]
    y = accuracy_tflops[m]
    ax2.scatter(x, y, color=colors[m], s=60)
    ax2.text(x, y + 0.3, m, fontsize=8, ha='center')

ax2.set_title("Logic Grid Puzzle(Accuracy versus TFLOPS)")
ax2.set_xlabel("TFLOPS")
ax2.set_ylabel("Accuracy(%)")

ax2.set_ylim(60, 70)
ax2.set_yticks(np.arange(60, 71, 2))

# Set x-limits to match the approximate range in the image
ax2.set_xlim(0, 1.1e6)

# Use scientific notation formatting for TFLOPS axis
ax2.ticklabel_format(axis='x', style='sci', scilimits=(5, 5))

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/scatter_15.png", bbox_inches="tight")