import matplotlib.pyplot as plt

# Dummy data mimicking the structure and proportions of the original plot
data = {
    's1': {
        'human': {'O': 100, 'T': 200, 'P': 500},
        'gpt4': {'O': 150, 'T': 300, 'P': 350},
        'gpt4-cot': {'O': 200, 'T': 400, 'P': 200},
    },
    's2': {
        'human': {'O': 150, 'T': 250, 'P': 400},
        'gpt4': {'O': 100, 'T': 350, 'P': 350},
        'gpt4-cot': {'O': 250, 'T': 300, 'P': 250},
    }
}

roles = ['human', 'gpt4', 'gpt4-cot']
colors = ['lightskyblue', 'gold', 'lightcoral']
segment_labels = ['O', 'T', 'P']

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle('Resonance Preference Plot for rich_context', fontsize=14)

# Plot data for s1
for i, role in enumerate(roles):
    ax1.barh(i, data['s1'][role]['O'], color=colors[0], left=0)
    ax1.barh(i, data['s1'][role]['T'], color=colors[1], left=data['s1'][role]['O'])
    ax1.barh(i, data['s1'][role]['P'], color=colors[2], left=data['s1'][role]['O'] + data['s1'][role]['T'])
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(roles)
ax1.set_title('s1')
ax1.set_xlim(0, 800)
ax1.set_xlabel('Count')

# Plot data for s2
for i, role in enumerate(roles):
    ax2.barh(i, data['s2'][role]['O'], color=colors[0], left=0)
    ax2.barh(i, data['s2'][role]['T'], color=colors[1], left=data['s2'][role]['O'])
    ax2.barh(i, data['s2'][role]['P'], color=colors[2], left=data['s2'][role]['O'] + data['s2'][role]['T'])
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(roles)
ax2.set_title('s2')
ax2.set_xlim(0, 800)
ax2.set_xlabel('Count')

# Add common y-label and legend
fig.text(0.04, 0.5, 'Role', ha='center', va='center', rotation='vertical', fontsize=12)
legend_handles = [plt.Rectangle((0,0), 1, 1, color=colors[i]) for i in range(3)]
fig.legend(legend_handles, segment_labels, loc='upper center', ncol=3)

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.savefig(r"eval_workspace_gpt5/InternVL3_5_241B_A28B/python/bar_16.png", bbox_inches="tight")