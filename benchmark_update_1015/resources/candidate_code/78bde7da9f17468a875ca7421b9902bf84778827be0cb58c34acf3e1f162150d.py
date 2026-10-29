import numpy as np
import matplotlib.pyplot as plt

# Generate data
t = np.linspace(0, 5, 100)
v = 0.5 * t**2 + 2 * t - 1

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(t, v, 'rD-', label='v(t) = 0.5t² + 2t - 1')

# Customize the plot
plt.title('Quadratic Velocity Function: v(t) = 0.5t² + 2t - 1')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.xlim(0, 5)
plt.ylim(-1, 20)

# Show the plot
plt.legend()
plt.savefig(r"eval_workspace_gpt5/Qwen2.5-VL-72B-Instruct/python/3e3fa84e1f6e40f1ac62373f79a0ff2b.png", bbox_inches="tight")