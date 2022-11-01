lines = []
with open("accuracy_multy.txt", "r") as f:
    while line := f.readline():
        lines.append(float(line.split("=")[2]))

import matplotlib.pyplot as plt



plt.figure(figsize=(5, 3))
plt.plot([*range(9, 3000, 50)], lines, ".", linestyle="--", markersize=5)
plt.ylabel(r"Accuracy%", fontsize=12)
plt.xlabel(r"Epochs", fontsize=12)
plt.xlim(0, 3000)
plt.ylim(50, 100)
plt.grid(which='major')
plt.grid(which='minor', alpha=0.5, linestyle='--')
plt.minorticks_on()
plt.tight_layout()
plt.savefig('TrainingAccuracy.eps')
plt.show()