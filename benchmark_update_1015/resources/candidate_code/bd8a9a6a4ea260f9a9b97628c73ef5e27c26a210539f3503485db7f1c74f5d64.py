import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Create figure and main plot
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.3, bottom=0.25)

# Initial parameters
magnitude = 4.0
periodicity = 2.5  # Hz

# Time data
t = np.linspace(0, 2.0, 1200)
y = magnitude * np.cos(2 * np.pi * periodicity * t)

# Plot the oscillatory motion
line, = ax.plot(t, y, '--', color='#2E86AB', linewidth=3)
ax.set_xlabel('Time Interval (s)')
ax.set_ylabel('Displacement (m)')
ax.set_title('Oscillatory Motion Analysis', fontsize=16, fontweight='bold')
ax.grid(True, linestyle=':', color='lightgray')

# Add vertical magnitude slider
ax_mag = plt.axes([0.1, 0.25, 0.025, 0.65], facecolor='none')
mag_slider = Slider(
    ax=ax_mag,
    label='Magnitude',
    valmin=0.0,
    valmax=8.0,
    valinit=magnitude,
    orientation='vertical',
    color='#1f77b4',
    track_color='#ff9900',
    valfmt='%1.0f'
)

# Style the magnitude slider
mag_slider.ax.set_facecolor('none')
mag_slider.ax.set_ylim(0, 8)
mag_slider.ax.set_yticks([])
mag_slider.ax.set_yticklabels([])
mag_slider.ax.spines['left'].set_visible(False)
mag_slider.ax.spines['right'].set_visible(False)
mag_slider.ax.spines['top'].set_visible(False)
mag_slider.ax.spines['bottom'].set_visible(False)

# Add horizontal periodicity slider
ax_per = plt.axes([0.3, 0.1, 0.6, 0.03], facecolor='none')
per_slider = Slider(
    ax=ax_per,
    label='Periodicity (Hz)',
    valmin=0.0,
    valmax=30.0,
    valinit=periodicity,
    color='#1f77b4',
    track_color='#a93a72',
    valfmt='%1.1f'
)

# Style the periodicity slider
per_slider.ax.set_facecolor('none')
per_slider.ax.set_xticks([])
per_slider.ax.set_xticklabels([])
per_slider.ax.spines['left'].set_visible(False)
per_slider.ax.spines['right'].set_visible(False)
per_slider.ax.spines['top'].set_visible(False)
per_slider.ax.spines['bottom'].set_visible(False)

# Add initialize button
ax_init = plt.axes([0.82, 0.03, 0.12, 0.045])
init_button = Button(ax_init, 'Initialize', color='lightgray', hovercolor='lightblue')

# Update function for sliders
def update(val):
    mag = mag_slider.val
    per = per_slider.val
    y = mag * np.cos(2 * np.pi * per * t)
    line.set_ydata(y)
    fig.canvas.draw_idle()

# Connect sliders to update function
mag_slider.on_changed(update)
per_slider.on_changed(update)

# Initialize button function
def init_button_clicked(event):
    mag_slider.set_val(4.0)
    per_slider.set_val(2.5)
    update(4.0)

init_button.on_clicked(init_button_clicked)

# Show the plot
plt.savefig(r"eval_workspace_gpt-5.2-codex/Qwen3-VL-32B-Instruct/python/3e2ce01d8c774a18b12812bfc916d361.png", bbox_inches="tight")
