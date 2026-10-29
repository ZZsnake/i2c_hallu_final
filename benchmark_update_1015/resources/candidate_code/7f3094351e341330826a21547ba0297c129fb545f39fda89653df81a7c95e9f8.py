import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Data setup
# ----------------------------
categories = [
    "Roleplay",
    "Writing",
    "Humanities",
    "STEM",
    "Extraction",
    "Coding",
    "Math",
    "Reasoning",
]
N = len(categories)

# Scores matching the reference radar chart
gptj =       [7.0, 9.5, 8.0, 8.5, 8.0, 7.5, 9.0, 8.5]
tinyllama =  [5.0, 6.5, 6.0, 7.0, 6.5, 6.0, 7.5, 7.0]
openllama =  [6.0, 8.0, 7.0, 7.5, 7.0, 6.5, 8.0, 7.5]
openmoe =    [8.5, 9.0, 9.0, 9.5, 9.0, 8.0, 9.5, 9.0]

data_series = [
    ("GPT-J-6B", gptj, 0),          # blue
    ("TinyLLaMA-1.1B", tinyllama, 1),  # orange
    ("OpenLLaMA-3B", openllama, 2),    # green
    ("OpenMoE-8B/32E", openmoe, 3),    # red
]

# Close the loop by appending the first value at the end
def close_data(values):
    return values + [values[0]]

# Angles for each axis
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += [angles[0]]

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

# Set starting angle at top and go clockwise
ax.set_theta_offset(np.pi / 2.0)
ax.set_theta_direction(-1)

# Draw each series
for label, vals, color_idx in data_series:
    vals_closed = close_data(vals)
    ax.plot(
        angles,
        vals_closed,
        color=plt.get_cmap("tab10")(color_idx),
        linewidth=2,
        marker="o",
        markersize=6,
    )

# Category labels at the correct angles (without the repeated last angle)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Radial limits and ticks
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels([])  # hide labels to match clean look
ax.grid(color="lightgrey", linewidth=0.8)

# Black outer spine
ax.spines["polar"].set_color("black")

# Legend in upper right outside the main circle
lines = [
    plt.Line2D([], [], color=plt.get_cmap("tab10")(i), linewidth=2, marker="o", markersize=6)
    for i in range(len(data_series))
]
labels = [s[0] for s in data_series]
ax.legend(
    lines,
    labels,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.00),
    frameon=True,
)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/GPT_5_1/python/radar_14.png", bbox_inches="tight")
