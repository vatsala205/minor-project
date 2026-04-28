import numpy as np
import matplotlib.pyplot as plt

time = np.arange(0, 50)

# Simulated mood drift
mood = np.sin(time / 5) + np.random.normal(0, 0.2, 50)

# Detection point
detection_point = 30

plt.figure()
plt.plot(time, mood, label="Mood Signal")
plt.axvline(detection_point, linestyle="--", label="Early Detection")

plt.xlabel("Time")
plt.ylabel("Mood Level")
plt.title("Fig. 4 — Early Detection Timeline")
plt.legend()
plt.show()