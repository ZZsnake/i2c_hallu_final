import matplotlib.pyplot as plt

# Dummy data based on visual inspection
models = ['gpt-4', 'text-davinci-003', 'text-davinci-002', 'claude-1', 'claude-2', 'hf_falcon-40b', 'llama-2-70', 'llama-2-70-chat']
model_basedness = [1.95, 1.45, 1.75, 1.25, 1.55, 1.85, 1.15, 1.35]
meta_cognition = [0.85, 0.65, 0.75, 0.55, 0.95, 0.95, 0.35, 0.25]
exploration = [0.35, 0.55, 0.75, 0.35, 0.75, 0.25, 0.85, 0.95]
risk_taking = [0.95, 0.75, 0.95, 0.65, 0.95, 0.45, 0.35, 0.25]

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Left plot: Performance Comparison
ax1.scatter(model_basedness, range(len(models)), color='blue', label='Model-Basedness', s=50)
ax1.scatter(meta_cognition, range(len(models)), color='orange', label='Meta-Cognition', s=50)
ax1.set_yticks(range(len(models)))
ax1.set_yticklabels(models)
ax1.set_xlabel('Value')
ax1.set_title('Performance Comparison')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right')

# Right plot: Strategic Traits
ax2.scatter(exploration, range(len(models)), color='green', label='Exploration', s=50)
ax2.scatter(risk_taking, range(len(models)), color='red', label='Risk Taking', s=50)
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models)
ax2.set_xlabel('Value')
ax2.set_title('Strategic Traits')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower right')

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/scatter_22.png", bbox_inches="tight")