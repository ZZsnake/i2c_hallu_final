import matplotlib.pyplot as plt
import numpy as np

# Data estimated from the visual representation of the plot
roles = ['human', 'gpt4', 'gpt4-cot']
# Data is entered in the order of the y-axis labels from top to bottom
# s1 data
s1_data = {
    'O': np.array([100, 150, 200]),
    'T': np.array([200, 300, 400]),
    'P': np.array([500, 350, 200])
}
# s2 data
s2_data = {
    'O': np.array([150, 100, 250]),
    'T': np.array([250, 350, 300]),
    'P': np.array([400, 350, 250])
}

# Colors from the plot
colors = {
    'O': 'skyblue',
    'T': 'gold',
    'P': 'lightcoral'
}

# Create the figure and a 2x1 grid of subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# --- Plot for s1 ---
# The left offsets for the stacked bars
left_s1_T = s1_data['O']
left_s1_P = s1_data['O'] + s1_data['T']

# Plot the bars for s1
# We plot in reverse order of roles because barh plots from bottom to top
b1 = ax1.barh(roles, s1_data['O'], height=0.32, color=colors['O'], label='O')
b2 = ax1.barh(roles, s1_data['T'], height=0.32, left=left_s1_T, color=colors['T'], label='T')
b3 = ax1.barh(roles, s1_data['P'], height=0.32, left=left_s1_P, color=colors['P'], label='P')
ax1.set_title('s1', y=1.22, pad=0)

# --- Plot for s2 ---
# The left offsets for the stacked bars
left_s2_T = s2_data['O']
left_s2_P = s2_data['O'] + s2_data['T']

# Plot the bars for s2
ax2.barh(roles, s2_data['O'], height=0.32, color=colors['O'])
ax2.barh(roles, s2_data['T'], height=0.32, left=left_s2_T, color=colors['T'])
ax2.barh(roles, s2_data['P'], height=0.32, left=left_s2_P, color=colors['P'])
ax2.set_title('s2', y=1.0, pad=15)

# --- General Formatting for both axes ---
for ax in [ax1, ax2]:
    ax.set_xlabel('Count')
    ax.set_ylabel('Role')
    ax.set_xlim(0, 800)
    ax.set_xticks(np.arange(0, 801, 100))
    
    # Invert y-axis to have 'human' at the top
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(axis='x', color='lightgray', linestyle='-')
    ax.set_axisbelow(True) # Ensure grid is behind bars
    
    # Keep full rectangular axes frames visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Remove y-axis ticks
    ax.tick_params(axis='y', length=0)

# --- Figure-level elements ---
# Main title
fig.suptitle('Resonance Preference Plot for rich_context', fontsize=14, y=0.99)

# Legend
fig.legend(handles=[b1, b2, b3], labels=['O', 'T', 'P'], 
           loc='upper center', bbox_to_anchor=(0.5, 0.865), 
           ncol=3, frameon=False)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.86]) # rect=[left, bottom, right, top]

# Show the plot
plt.savefig(r"eval_workspace_gpt5/Gemini2_5_Pro/python/bar_16.png", bbox_inches="tight")
