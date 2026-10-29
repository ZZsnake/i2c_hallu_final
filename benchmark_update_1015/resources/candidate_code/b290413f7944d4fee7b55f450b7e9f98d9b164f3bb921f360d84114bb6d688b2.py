import matplotlib.pyplot as plt
import numpy as np

# Data setup
categories = ['Roleplay', 'Writing', 'Humanities', 'STEM', 'Extraction', 'Coding', 'Math', 'Reasoning']
N = len(categories)

# Create dummy data based on visual observation
gpt_j_6b = [3.8, 4.2, 4.5, 4.3, 4.0, 4.8, 5.0, 4.5]
tinyllama_1_1b = [3.5, 3.8, 3.2, 3.5, 3.8, 4.2, 4.5, 4.0]
openllama_3b = [3.6, 3.9, 3.5, 3.8, 4.0, 4.5, 4.7, 4.2]
openmoe_8b_32e = [4.0, 4.3, 4.0, 4.2, 4.5, 4.6, 4.8, 4.3]

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
ax.fill(angles, gpt_j_6b, alpha=0.05, color='#4472C4')

ax.plot(angles, tinyllama_1_1b, 'o-', linewidth=2, label='TinyLLaMA-1.1B', color='#ED7D31')
ax.fill(angles, tinyllama_1_1b, alpha=0.05, color='#ED7D31')

ax.plot(angles, openllama_3b, 'o-', linewidth=2, label='OpenLLaMA-3B', color='#70AD47')
ax.fill(angles, openllama_3b, alpha=0.05, color='#70AD47')

ax.plot(angles, openmoe_8b_32e, 'o-', linewidth=2, label='OpenMoE-8B/32E', color='#C55A5A')
ax.fill(angles, openmoe_8b_32e, alpha=0.05, color='#C55A5A')

# Set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)

# Set radial limits and labels
ax.set_ylim(0, 6)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], size=8, color='gray')

# Grid styling
ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)

# Legend
ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1), frameon=True, fontsize=9)

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/radar_14.png", bbox_inches="tight")