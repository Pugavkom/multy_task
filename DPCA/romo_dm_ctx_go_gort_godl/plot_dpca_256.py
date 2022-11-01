import os

import numpy as np
from matplotlib import pyplot as plt

tasks = [
    "RomoTask1",
    "RomoTask2",
    "DMTask1",
    "DMTask2",
    "CtxDMTask1",
    "CtxDMTask2",
    "GoTask1",
    "GoTask2",
    "GoRtTask1",
    "GoRtTask2",
    "GoDlTask1",
    "GoDlTask2",
]

go_task_list_values = np.linspace(0, 1, 8)


def main():
    symbols = ["st"]
    for task in tasks[:]:
        fig = plt.figure(figsize=(6, 4))
        axes = [fig.add_subplot(210 + i + 1) for i in range(2)]
        for i in range(len(symbols)):
            a = np.load(os.path.join("data", f"Z_a_{symbols[i]}_{task}_256_1_2.npy"))
            v = np.load(os.path.join("data", f"Z_v_{symbols[i]}_{task}_256_1_2.npy"))
            for s in range(v.shape[1]):
                axes[i].plot(
                    a[0, s], label=f"$u_{{mod}}$ = {round(go_task_list_values[s], 3)}"
                )
                axes[i].set_ylabel("dPCA: $a$")
                axes[i + 1].plot(
                    v[0, s], label=f"$u_{{mod}} = {round(go_task_list_values[s], 3)}"
                )
                axes[i + 1].set_ylabel("dPCA: $v$")
                axes[i + 1].set_xlabel("time, ms")
            [
                (
                    el.grid(True),
                    el.minorticks_on(),
                    el.grid(which="minor", linestyle="--", alpha=0.8),
                    el.set_xlim([0, len(a[0, s])]),
                )
                for el in axes
            ]
            # plt.legend()
            # axes[i].set_title(task)
        axes[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.0, 1), shadow=True)
        plt.tight_layout()
        plt.savefig(f"a_{task}.png")
        plt.savefig(f"eps/256/DPCA_{task}_256_1_2.eps")
        # plt.show()
        # plt.close()


if __name__ == "__main__":
    main()
