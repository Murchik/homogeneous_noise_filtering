import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

n = 64
# time vector
t = np.linspace(0, 1, n, endpoint=True)
# sine wave
ds = np.sin(2*np.pi*t)

# discrete step
step_lst = np.array([3, 5, 8, 32])

# plot figure
fig = plt.figure(figsize=(12, 6), dpi=80)
for i in range(4):
    tt = np.linspace(0, 1, step_lst[i], endpoint=True)

    plt.subplot(2, 2, i+1)
    plt.title('Number of points = {}'.format(step_lst[i]))
    plt.plot(t, ds, '-', linewidth=2.0)
    plt.plot(tt, np.sin(2*np.pi*tt), '--o', linewidth=1.5, markersize=8)
    plt.step(tt, np.sin(2*np.pi*tt), linewidth=1.5)
    plt.grid()
    plt.xlim([0, 1])
plt.tight_layout()
plt.show()
