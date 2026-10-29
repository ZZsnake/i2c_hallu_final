import matplotlib.pyplot as plt
import numpy as np

# Data based on the ground-truth image
categories = ['Type', 'Span', 'T&S', 'Spurious', 'Total']
manual_mapping = [60, 0, 0, 20, 60]
llm_revision = [50, 30, 30, 10, 50]
llm_cot = [40, 50, 40, 30, 30]
verifner = [30, 30, 50, 40, 30]

bc5cdr_manual_mapping = [60, 40, 80, 20, 60]
bc5cdr_llm_revision = [50, 30, 70, 10, 50]
bc5cdr_llm_cot = [80, 60, 90, 30, 80]
bc5cdr_verifner = [90, 70, 80, 40, 90]

# Colors
colors = ['#E39A9A', '#F9CF8B', '#DF5F5F', '#95D3A8']

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

# Plot the first subplot (GENIA)
width = 0.2
x = np.arange(len(categories))
ax1.bar(x - width*3/2, manual_mapping, width, label='Manual Mapping', color=colors[0])
ax1.bar(x - width/2, llm_revision, width, label='LLM-revision', color=colors[1])
ax1.bar(x + width/2, llm_cot, width, label='LLM-revision w/CoT', color=colors[2])
ax1.bar(x + width*3/2, verifner, width, label='VerifNER', color=colors[3])
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.set_ylim(0, 100)
ax1.set_title('GENIA')
ax1.legend(loc='upper right', ncol=2)

# Plot the second subplot (BC5CDR)
ax2.bar(x - width*3/2, bc5cdr_manual_mapping, width, label='Manual Mapping', color=colors[0])
ax2.bar(x - width/2, bc5cdr_llm_revision, width, label='LLM-revision', color=colors[1])
ax2.bar(x + width/2, bc5cdr_llm_cot, width, label='LLM-revision w/CoT', color=colors[2])
ax2.bar(x + width*3/2, bc5cdr_verifner, width, label='VerifNER', color=colors[3])
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.set_ylim(0, 100)
ax2.set_title('BC5CDR')

# Shared y-axis label
fig.text(0.02, 0.5, 'Error correction rate (%)', va='center', rotation='vertical')

# Adjust layout and display the plot
plt.tight_layout(rect=(0.04, 0, 1, 1))
plt.savefig(r"eval_workspace_gpt5/Gemma3-27B-Instruct/python/bar_9.png", bbox_inches="tight")
