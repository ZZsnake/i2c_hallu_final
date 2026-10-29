import matplotlib.pyplot as plt

# Dummy data based on visual inspection
models = ['gpt-4', 'text-davinci-003', 'text-davinci-002', 'claude-1', 'claude-2', 'text-bison@002', 'hf_falcon-40b', 'llama-2-70', 'llama-2-70-chat']
model_basedness = [2.00, 1.50, 1.80, 1.20, 1.60, 1.40, 1.90, 1.10, 1.30]
meta_cognition = [0.80, 0.60, 0.70, 0.50, 0.90, 0.40, 1.00, 0.30, 0.20]
exploration = [0.30, 0.50, 0.40, 0.60, 0.20, 0.70, 0.10, 0.80, 0.90]
risk_taking = [0.90, 0.70, 0.80, 0.60, 1.00, 0.50, 0.40, 0.20, 0.30]

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Left plot: Performance Comparison
ax1.scatter(model_basedness, range(len(models)), color='blue', label='Model-Basedness', s=36)
ax1.scatter(meta_cognition, range(len(models)), color='orange', label='Meta-Cognition', s=50)
ax1.set_yticks(range(len(models)))
ax1.set_yticklabels(models)
ax1.set_xlabel('Value')
ax1.set_ylabel('Models')
ax1.set_title('Performance Comparison')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right')

# Right plot: Strategic Traits
ax2.scatter(exploration, range(len(models)), color='green', label='Exploration', s=36)
ax2.scatter(risk_taking, range(len(models)), color='red', label='Risk Taking', s=50)
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels([])
ax2.set_xlabel('Value')
ax2.set_title('Strategic Traits')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower center')

# Adjust layout
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-235b-a22b-Instruct/python/scatter_22.png", bbox_inches="tight")
