import matplotlib.pyplot as plt
import numpy as np

# Data for the plot
x = np.array([40, 50, 60, 70, 80, 90, 100])
y_aptq = np.array([7.5, 6.5, 6.0, 5.5, 5.5, 5.5, 5.5])
y_llama = np.array([5.22] * 7)
y_owo = np.array([5.56] * 7)
y_gptq = np.array([5.62] * 7)
y_llm_qat = np.array([7.4] * 7)

# Create the plot
plt.figure(figsize=(5, 4))

# Plot each line with specified styles
plt.plot(x, y_aptq, marker='o', linestyle='-', color='blue', label='APTQ')
plt.plot(x, y_llama, marker='s', linestyle='--', color='magenta', label='LLaMa-7B (FP16): 5.22')
plt.plot(x, y_owo, marker='^', linestyle='-', color='orange', label='OWO-4bit: 5.56')
plt.plot(x, y_gptq, marker='v', linestyle='-', color='green', label='GPTQ-4bit: 5.62')
plt.plot(x, y_llm_qat, marker='d', linestyle=':', color='brown', label='LLM-QAT-4bit: 7.4')

# Add annotations for data points
annotations = [
    (40, 7.5, '2.91 bit'),
    (50, 6.5, '3.11 bit'),
    (60, 6.0, '3.32 bit'),
    (70, 5.5, '3.53 bit'),
    (80, 5.5, '3.63 bit'),
    (90, 5.5, '3.74 bit'),
    (100, 5.5, '3.94 bit')
]

for x_val, y_val, text in annotations:
    plt.text(x_val, y_val, text, ha='right', va='bottom', fontsize=10)

# Set axis labels and title
plt.xlabel('Ratio of 4-bit Utilization (%)')
plt.ylabel('Perplexity (PPL)')
plt.title('Perplexity vs. 4-bit Utilization Ratio')

# Set axis limits
plt.xlim(35, 105)
plt.ylim(5.0, 7.5)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(loc='upper right', fontsize=8)

# Show plot
plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Qwen3-VL-8B-Instruct/python/line_22.png", bbox_inches="tight")