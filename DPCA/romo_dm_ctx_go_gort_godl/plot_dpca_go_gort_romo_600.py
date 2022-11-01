import os

import matplotlib.pyplot as plt
import numpy as np

tasks = ["GoTask1", "GoRtTask1", "RomoTask2"]

go_task_list_values = np.linspace(0, 1, 8)


def plot_graph(data_a, data_v, ax_a, ax_v, letter, legend=False):
    for s in range(data_v.shape[1]):

        ax_a.plot(
            data_a[0, s], label=f"$u_{{mod}}$ = {round(go_task_list_values[s], 3)}"
        )
        ax_v.plot(
            data_v[0, s],
        )
    ax_a.set_ylabel("dPCA: a")
    ax_v.set_ylabel("dPCA: v")
    [
        (
            el.grid(True),
            el.minorticks_on(),
            el.grid(which="minor", linestyle="--", alpha=0.8),
            el.set_xlim([0, len(data_a[0, s])]),
        )
        for el in [ax_a, ax_v]
    ]
    ax_v.text(-0.05, -0.35, letter, transform=ax_v.transAxes, va="bottom", ha="right")


def main():
    components = [
        "st",
    ]
    fig = plt.figure(figsize=(6, 7.5))
    axes = [
        fig.add_subplot(10 + 100 * len(tasks) * 2 + i + 1)
        for i in range(2 * len(tasks))
    ]
    j = 0
    letters = ["(a)", "(b)", "(c)", "(d)"]
    for n_task, task in enumerate(tasks):
        for i in range(len(components)):
            a = np.load(os.path.join("data", f"Z_a_{components[i]}_{task}_600_1_2.npy"))
            v = np.load(os.path.join("data", f"Z_v_{components[i]}_{task}_600_1_2.npy"))
            plot_graph(a, v, axes[j], axes[j + 1], letters[n_task])
        j += 2
    axes[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.0, 1), shadow=True)
    plt.tight_layout()
    plt.savefig("dpca_600_go_go_rt_romo.eps")


if __name__ == "__main__":
    main()
