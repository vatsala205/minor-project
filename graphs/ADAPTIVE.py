import numpy as np
import matplotlib.pyplot as plt

time = np.arange(0, 50)

signal = np.random.normal(0, 1, 50).cumsum()
threshold = np.mean(signal) + 0.5 * np.std(signal)

adaptive_threshold = np.convolve(signal, np.ones(5)/5, mode='same')

plt.figure()
plt.plot(signal, label="Mood Drift Signal")
plt.plot(adaptive_threshold, label="Adaptive Threshold")
plt.axhline(threshold, linestyle="--", label="Static Threshold")

plt.xlabel("Time")
plt.ylabel("Signal Value")
plt.title("Fig. 5 — Adaptive Threshold Mechanism")
plt.legend()
plt.show()