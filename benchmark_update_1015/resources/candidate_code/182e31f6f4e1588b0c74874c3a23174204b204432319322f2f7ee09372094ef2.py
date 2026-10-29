import matplotlib.pyplot as plt

# Data for the box plots
data = {
    'Llama-70b (T2TT)': [19.5, 20.5, 21.5, 22.5],
    'Llama-70b (S2TT)': [18.5, 19.5, 20.5, 21.5]
}

# Create figure with specified dimensions
fig, ax = plt.subplots(figsize=(6, 4))

# Plotting the box plots
ax.boxplot(data.values(), labels=data.keys())

# Setting title and labels
ax.set_title('BLEU Scores Comparison')
ax.set_ylabel('BLEU')

# Adding horizontal dashed line at 20 BLEU
ax.axhline(20, color='gray', linestyle='--', linewidth=1)

# Show grid
ax.grid(True)

# Display the plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-8B-Instruct/python/box_11.png", bbox_inches="tight")