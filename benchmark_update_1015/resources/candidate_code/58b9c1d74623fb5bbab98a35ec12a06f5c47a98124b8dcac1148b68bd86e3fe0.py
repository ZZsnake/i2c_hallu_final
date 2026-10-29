# ===================
# Part 1: Importing Libraries
# ===================
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


# ===================
# Part 2: Data Preparation
# ===================
# Data for GENIA
genia_categories = ["Type", "Span", "T&S", "Spurious", "Total"]
genia_manual_mapping = [60, 0, 0, 20, 60]
genia_llm_revision = [50, 30, 30, 10, 50]
genia_llm_revision_wcot = [40, 50, 40, 30, 30]
genia_verifner = [30, 30, 50, 40, 30]

# Data for BC5CDR
bc5cdr_categories = ["Type", "Span", "T&S", "Spurious", "Total"]
bc5cdr_manual_mapping = [60, 40, 80, 20, 60]
bc5cdr_llm_revision = [50, 30, 70, 10, 50]
bc5cdr_llm_revision_wcot = [80, 60, 90, 30, 80]
bc5cdr_verifner = [90, 70, 80, 40, 90]

labels = ["Manual Mapping", "LLM-revision", "LLM-revision w/CoT", "VerifNER"]
title = "Error correction rate (%)"
title1 = "GENIA"
title2 = "BC5CDR"

ylim1 = [0, 100]
yticks1 = np.arange(0, 101, 20)
ylim2 = [0, 100]
yticks2 = np.arange(0, 101, 20)

# ===================
# Part 3: Plot Configuration and Rendering
# ===================
# Set up the figure and axes
figure, axes = plt.subplots(2, 1, figsize=(9, 6))  # Width, Height in inches

# Plot for GENIA
x = np.arange(len(genia_categories))
width = 0.18  # Adjust this value to change the width of the bars
spacing = 0.2  # Adjust this value to change the spacing between the bars

axes[0].bar(
    x - spacing * 1.5,
    genia_manual_mapping,
    width,
    label=labels[0],
    color="#e19e9c",
)
axes[0].bar(
    x - spacing / 2, genia_llm_revision, width, label=labels[1], color="#f7ce91"
)
axes[0].bar(
    x + spacing / 2,
    genia_llm_revision_wcot,
    width,
    label=labels[2],
    color="#de6560",
)
axes[0].bar(x + spacing * 1.5, genia_verifner, width, label=labels[3], color="#94d4ac")
# axes[0].set_ylabel('Error correction rate (%)')
axes[0].set_title(title1)
axes[0].set_xticks(x)
axes[0].set_xticklabels(genia_categories)
axes[0].set_ylim(ylim1)
axes[0].set_yticks(yticks1)
axes[0].legend(loc="upper right", ncol=2)

# Plot for BC5CDR
x = np.arange(len(bc5cdr_categories))
axes[1].bar(x - spacing * 1.5, bc5cdr_manual_mapping, width, color="#e19e9c")
axes[1].bar(x - spacing / 2, bc5cdr_llm_revision, width, color="#f7ce91")
axes[1].bar(x + spacing / 2, bc5cdr_llm_revision_wcot, width, color="#de6560")
axes[1].bar(x + spacing * 1.5, bc5cdr_verifner, width, color="#94d4ac")
# axes[1].set_ylabel('Error correction rate (%)')
axes[1].set_title(title2)
axes[1].set_xticks(x)
axes[1].set_xticklabels(bc5cdr_categories)
axes[1].set_ylim(ylim2)
axes[1].set_yticks(yticks2)

# Add a common y-axis label
figure.text(
    0.0001,
    0.5,
    title,
    va="center",
    rotation="vertical",
    fontsize=12,
)

# ===================
# Part 4: Saving Output
# ===================
# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("bar_9.pdf", bbox_inches="tight")
