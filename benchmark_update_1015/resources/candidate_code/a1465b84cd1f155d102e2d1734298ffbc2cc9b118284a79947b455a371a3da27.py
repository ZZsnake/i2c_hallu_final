import matplotlib.pyplot as plt
import numpy as np

# Data setup
categories = ['Humanities', 'Writing', 'Roleplay', 'Reasoning', 'Math', 'Coding', 'Extraction', 'STEM']
N = len(categories)

# Data based on the ground-truth radar chart, ordered to match categories above
gpt_j_6b = [0.80, 0.90, 0.70, 0.85, 0.90, 0.75, 0.80, 0.85]
tinyllama_1_1b = [0.60, 0.65, 0.50, 0.70, 0.75, 0.60, 0.65, 0.70]
openllama_3b = [0.70, 0.80, 0.60, 0.75, 0.80, 0.65, 0.70, 0.75]
openmoe_8b_32e = [0.90, 0.95, 0.85, 0.90, 0.95, 0.80, 0.90, 0.95]

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

# Complete the loop
gpt_j_6b += gpt_j_6b[:1]
tinyllama_1_1b += tinyllama_1_1b[:1]
openllama_3b += openllama_3b[:1]
openmoe_8b_32e += openmoe_8b_32e[:1]
angles += angles[:1]

# Create figure
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

# Plot data
ax.plot(angles, gpt_j_6b, 'o-', linewidth=2, label='GPT-J-6B', color='#4472C4')

ax.plot(angles, tinyllama_1_1b, 'o-', linewidth=2, label='TinyLLaMA-1.1B', color='#ED7D31')

ax.plot(angles, openllama_3b, 'o-', linewidth=2, label='OpenLLaMA-3B', color='#70AD47')

ax.plot(angles, openmoe_8b_32e, 'o-', linewidth=2, label='OpenMoE-8B/32E', color='#C55A5A')

# Set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)

# Set radial limits and labels
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([])

# Grid styling
ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)

# Legend
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, fontsize=9)

# Adjust layout
plt.tight_layout(pad=2.0)
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/radar_14.png", bbox_inches="tight")
