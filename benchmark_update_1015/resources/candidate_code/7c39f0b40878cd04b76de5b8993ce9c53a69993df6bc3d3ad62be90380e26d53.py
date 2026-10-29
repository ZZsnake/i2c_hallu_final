import matplotlib.pyplot as plt

# Sample data
data_T2TT = [19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 20.5, 21.0, 20.0, 21.5]
data_S2TT = [18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 19.5, 20.0, 19.0, 20.5]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(6, 4))

# Create box plots
box_T2TT = ax.boxplot(data_T2TT, positions=[0], patch_artist=True, boxprops=dict(facecolor='white', color='black'),
                      medianprops=dict(color='orange'), whiskerprops=dict(color='black'))
box_S2TT = ax.boxplot(data_S2TT, positions=[1], patch_artist=True, boxprops=dict(facecolor='white', color='black'),
                      medianprops=dict(color='orange'), whiskerprops=dict(color='black'))

# Add dashed line and arrow
medians = [box['medians'][0].get_ydata()[0] for box in [box_T2TT, box_S2TT]]
ax.plot([0, 1], medians, 'k--')
ax.annotate('', xy=(1, medians[0]), xytext=(0, medians[0]),
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1))

# Add text for BLEU difference
ax.text(0.5, medians[0] - 0.5, '1.0 BLEU', horizontalalignment='center', verticalalignment='center', rotation=90)

# Set labels and title
ax.set_yticks([16, 18, 20, 22])
ax.set_yticklabels(['16', '18', '20', '22'])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Llama-70b (T2TT)', 'Llama-70b (S2TT)'])
ax.set_ylabel('BLEU')

# Show plot
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/box_11.png", bbox_inches="tight")