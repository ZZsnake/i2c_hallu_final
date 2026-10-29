import numpy as np
import matplotlib.pyplot as plt

# Categories and angles
labels = ["Writing", "Humanities", "STEM", "Extraction",
          "Coding", "Math", "Reasoning", "Roleplay"]
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
# Close the loop
angles = np.concatenate((angles, [angles[0]]))

# Dummy data approximating the visual
gptj =       [0.85, 0.80, 0.78, 0.75, 0.72, 0.78, 0.80, 0.75]
tinyllama =  [0.65, 0.60, 0.62, 0.58, 0.55, 0.60, 0.62, 0.55]
openllama =  [0.78, 0.74, 0.72, 0.68, 0.65, 0.70, 0.72, 0.70]
openmoe =    [0.90, 0.88, 0.87, 0.85, 0.82, 0.88, 0.89, 0.86]

# Close each data series
gptj += [gptj[0]]
tinyllama += [tinyllama[0]]
openllama += [openllama[0]]
openmoe += [openmoe[0]]

# Colors (Matplotlib default cycle)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create figure
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

# Set theta grid (category labels)
theta_degrees = np.degrees(angles[:-1])
ax.set_thetagrids(theta_degrees, labels)

# Set radial limits and grid
ax.set_ylim(0, 1.0)
ax.set_yticks(np.linspace(0.2, 1.0, 5))
ax.set_yticklabels([])  # hide radial tick labels for clean look
ax.grid(color='lightgray', linestyle='-', linewidth=0.8)

# Plot each model
lines = []
labels_models = ["GPT-J-6B", "TinyLLaMA-1.1B", "OpenLLaMA-3B", "OpenMoE-8B/32E"]
data_series = [gptj, tinyllama, openllama, openmoe]

for data, color in zip(data_series, colors):
    line, = ax.plot(angles, data, color=color, linewidth=2, marker='o',
                    markersize=3.5)
    ax.fill(angles, data, color=color, alpha=0.25)
    lines.append(line)

# Title
ax.set_title("Roleplay", y=1.08)

# Legend
ax.legend(lines, labels_models, loc='upper right',
          bbox_to_anchor=(1.25, 1.10), frameon=True)

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/radar_14.png", bbox_inches="tight")