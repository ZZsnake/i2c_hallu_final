import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.28, bottom=0.25)

# Initial parameters
amp = 4
freq = 2.5

# X data
t = np.linspace(0, 2, 1000)

# Y data
y = amp * np.cos(2 * np.pi * freq * t)

# Plot
line, = ax.plot(t, y, '--', color='#2E86AB', linewidth=3)

# Axes limits
ax.set_xlim(0, 2)
ax.set_ylim(-4, 4)
ax.set_xlabel('Time Interval (s)')
ax.set_ylabel('Displacement (m)')
ax.set_title('Oscillatory Motion Analysis', fontweight='bold')
ax.grid(True, linestyle=':', color='0.8', linewidth=0.8)

# Slider for Magnitude
axamp = plt.axes([0.07, 0.25, 0.02, 0.65])
samplitude = Slider(axamp, 'Magnitude', 0, 4, valinit=amp, color='orange', valstep=0.1, orientation='vertical')

# Slider for Periodicity
axfreq = plt.axes([0.28, 0.10, 0.65, 0.02])
sfreq = Slider(axfreq, 'Periodicity (Hz)', 0, 2.5, valinit=freq, color='magenta', valstep=0.1)

# Button
axinit = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(axinit, 'Initialize', color='gray', hovercolor='lightgray')

# Update function for sliders
def update(val):
    amp = samplitude.val
    freq = sfreq.val
    line.set_ydata(amp * np.cos(2 * np.pi * freq * t))
    fig.canvas.draw_idle()

# Button function
def reset(event):
    samplitude.reset()
    sfreq.reset()

# Connect sliders and button
samplitude.on_changed(update)
sfreq.on_changed(update)
button.on_clicked(reset)
plt.savefig(r"eval_workspace_gpt5/InternVL3_5-38B-Instruct/python/3e2ce01d8c774a18b12812bfc916d361.png", bbox_inches="tight")
