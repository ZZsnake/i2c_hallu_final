import matplotlib.pyplot as plt
import numpy as np

# Data for APTQ curve
x = np.array([40, 50, 60, 70, 80, 90, 100])
y = np.array([7.5, 6.5, 6.0, 5.8, 5.7, 5.65, 5.6])

bit_labels = ["2.91 bit", "3.11 bit", "3.32 bit", "3.53 bit",
              "3.63 bit", "3.74 bit", "3.94 bit"]

# Baseline perplexities
y_llama = 5.22
y_owq = 5.56
y_gptq = 5.62
y_qat = 7.4

plt.figure(figsize=(5, 4))

# APTQ line with markers
plt.plot(
    x, y,
    marker='o',
    color='C0',
    linewidth=2,
    markersize=6,
    label='APTQ'
)

# Horizontal baseline lines
plt.axhline(y_llama, color='magenta', linestyle='--', linewidth=1.5,
            label='LLaMa-7B (FP16): 5.22')
plt.axhline(y_owq, color='tab:green', linestyle='--', linewidth=1.5,
            label='OWQ-4bit: 5.56')
plt.axhline(y_gptq, color='goldenrod', linestyle='--', linewidth=1.5,
            label='GPTQ-4bit: 5.62')
plt.axhline(y_qat, color='red', linestyle='--', linewidth=1.5,
            label='LLM-QAT-4bit: 7.4')

# Annotations for bits above each marker
for xi, yi, label in zip(x, y, bit_labels):
    plt.text(
        xi,
        yi + 0.12,           # small vertical offset
        label,
        ha='center',
        va='bottom',
        fontsize=9
    )

# Axis labels
plt.xlabel('Ratio of 4-bit Utilization (%)')
plt.ylabel('Perplexity (PPL)')

# Axes limits to visually match
plt.xlim(38, 102)
plt.ylim(5.0, 7.8)

# X ticks
plt.xticks(x)

# Legend styling
legend = plt.legend(loc='upper right', frameon=True)
legend.get_frame().set_facecolor('whitesmoke')
legend.get_frame().set_edgecolor('gray')

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/Gemini3_Pro/python/line_22.png", bbox_inches="tight")