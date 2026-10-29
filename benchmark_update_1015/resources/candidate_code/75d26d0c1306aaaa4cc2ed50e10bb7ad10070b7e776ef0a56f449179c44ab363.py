import matplotlib.pyplot as plt
import numpy as np

# Define models
models = [
    'llama-2-70-chat',
    'llama-2-70',
    'hf_falcon-40b',
    'text-bison@002',
    'claude-2',
    'claude-1',
    'text-davinci-002',
    'text-davinci-003',
    'gpt-4'
]

# Create dummy data based on visual observation
# Performance Comparison data
model_basedness = [0.25, 0.30, 1.00, 0.40, 0.90, 0.50, 0.70, 0.65, 0.80]
meta_cognition = [1.35, 1.15, 1.95, 1.40, 1.65, 1.25, 1.80, 1.55, 2.00]

# Strategic Traits data
exploration = [0.95, 0.82, 0.12, 0.73, 0.25, 0.60, 0.42, 0.52, 0.33]
risk_taking = [0.30, 0.22, 0.42, 0.52, 1.00, 0.60, 0.82, 0.72, 0.95]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Y-axis positions for models
y_positions = np.arange(len(models))

# Left subplot - Performance Comparison
ax1.scatter(model_basedness, y_positions, color='#FF8C00', s=60, label='Model-Basedness', zorder=3)
ax1.scatter(meta_cognition, y_positions, color='#0000FF', s=60, label='Meta-Cognition', zorder=3)
ax1.set_yticks(y_positions)
ax1.set_yticklabels(models)
ax1.set_xlabel('Value')
ax1.set_ylabel('Models')
ax1.set_title('Performance Comparison', fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax1.legend(loc='upper right')
ax1.set_xlim(0.15, 2.1)

# Right subplot - Strategic Traits
ax2.scatter(exploration, y_positions, color='#008000', s=60, label='Exploration', zorder=3)
ax2.scatter(risk_taking, y_positions, color='#FF0000', s=60, label='Risk Taking', zorder=3)
ax2.set_yticks(y_positions)
ax2.set_yticklabels([])  # Hide y-axis labels on right plot
ax2.set_xlabel('Value')
ax2.set_title('Strategic Traits', fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax2.legend(loc='lower right')
ax2.set_xlim(0.05, 1.05)

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Claude_Sonnet_4_5/python/scatter_22.png", bbox_inches="tight")