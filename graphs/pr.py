import numpy as np
import matplotlib.pyplot as plt


recall = np.linspace(0, 1, 100)

precision_proposed = 1 - 0.4 * recall
precision_baseline = 0.5 * np.ones_like(recall)

plt.figure()
plt.plot(recall, precision_proposed, label="Proposed Model")
plt.plot(recall, precision_baseline, linestyle="--", label="Baseline")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Fig. 3 — Precision–Recall Curve")
plt.legend()
plt.show()