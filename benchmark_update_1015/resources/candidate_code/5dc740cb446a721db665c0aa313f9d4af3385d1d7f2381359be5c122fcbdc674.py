import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Type', 'Span', 'T&S', 'Spurious', 'Total']
methods = ['Manual Mapping', 'LLM-revision w/CoT', 'VerifNER']
colors = ['lightcoral', 'salmon', 'lightgreen']

# Error correction rates for GENIA
genia_rates = [
    [60, 40, 30],   # Type
    [30, 50, 40],   # Span
    [40, 60, 50],   # T&S
    [20, 30, 40],   # Spurious
    [50, 70, 60]    # Total
]

# Error correction rates for BC5CDR
bc5cdr_rates = [
    [80, 60, 70],   # Type
    [60, 80, 70],   # Span
    [70, 90, 80],   # T&S
    [30, 40, 50],   # Spurious
    [70, 90, 80]    # Total
]

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

x = np.arange(len(categories))
width = 0.25

# GENIA Chart
ax1.bar(x - width, [genia_rates[i][0] for i in range(len(categories))], width, label=methods[0], color=colors[0])
ax1.bar(x, [genia_rates[i][1] for i in range(len(categories))], width, label=methods[1], color=colors[1])
ax1.bar(x + width, [genia_rates[i][2] for i in range(len(categories))], width, label=methods[2], color=colors[2])
ax1.set_title('GENIA')
ax1.set_ylabel('Error correction rate (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()

# BC5CDR Chart
ax2.bar(x - width, [bc5cdr_rates[i][0] for i in range(len(categories))], width, label=methods[0], color=colors[0])
ax2.bar(x, [bc5cdr_rates[i][1] for i in range(len(categories))], width, label=methods[1], color=colors[1])
ax2.bar(x + width, [bc5cdr_rates[i][2] for i in range(len(categories))], width, label=methods[2], color=colors[2])
ax2.set_title('BC5CDR')
ax2.set_ylabel('Error correction rate (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()

plt.tight_layout()
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/bar_9.png", bbox_inches="tight")