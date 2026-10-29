import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    "gpt-4",
    "text-davinci-003",
    "text-davinci-002",
    "claude-1",
    "claude-2",
    "text-bison@002",
    "hf_falcon-40b",
    "llama-2-70",
    "llama-2-70-chat",
]

y_pos = np.arange(len(models))  # 0 at bottom

# Left subplot data: Performance Comparison
model_basedness = [2.0, 1.5, 1.8, 1.2, 1.6, 1.4, 1.9, 1.1, 1.3]
meta_cognition  = [0.8, 0.6, 0.7, 0.5, 0.9, 0.4, 1.0, 0.3, 0.2]

# Right subplot data: Strategic Traits
exploration = [0.3, 0.5, 0.4, 0.6, 1.0, 0.7, 0.1, 0.8, 0.9]
risk_taking = [0.9, 0.7, 0.8, 0.6, 0.5, 0.4, 0.4, 0.2, 0.3]

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# ----- Left subplot: Performance Comparison -----
ax1.scatter(model_basedness, y_pos, color='tab:blue', label='Model-Basedness')
ax1.scatter(meta_cognition,  y_pos, color='tab:orange', label='Meta-Cognition')

ax1.set_title("Performance Comparison")
ax1.set_xlabel("Value")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(models)
ax1.set_ylabel("Models")
ax1.set_xlim(0, 2.1)

ax1.legend(loc='upper right')

# ----- Right subplot: Strategic Traits -----
ax2.scatter(exploration, y_pos, color='tab:green', label='Exploration')
ax2.scatter(risk_taking, y_pos, color='tab:red', label='Risk Taking')

ax2.set_title("Strategic Traits")
ax2.set_xlabel("Value")
ax2.set_xlim(0, 1.05)

# Share the same y-axis labels
ax2.set_yticks(y_pos)
ax2.set_yticklabels(models)

# Place legend at bottom center inside the right axes
ax2.legend(loc='lower center')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/scatter_22.png", bbox_inches="tight")