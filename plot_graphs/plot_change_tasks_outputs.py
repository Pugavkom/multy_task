import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams["font.size"] = 8
plt.rcParams["hatch.color"] = "#000066"


def main():
    outputs = np.load("data_task_change_outputs/task_change_outputs.npy")
    inputs = np.load("data_task_change_outputs/task_change_inputs.npy")
    dt = 1e-3
    x = np.arange(len(outputs)) * 1e-3
    start_plot = 0
    stop_plot = len(x)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.add_patch(Rectangle((0 * dt, -10), 1850 * dt, 20, alpha=1, facecolor="#FECACA"))
    ax.add_patch(
        Rectangle(
            (1850 * dt, -10), (len(x) - 1850) * dt, 20, alpha=1, facecolor="#FEFDCA"
        )
    )
    # plt.fill_between([3000, 10], [31049, 10], alpha=0.0, hatch="X", edgecolor="b", linewidth=0.1)

    # ax.add_patch(Rectangle((30000 * dt, -10), 1049 * dt, 20, alpha=.5, facecolor='w', hatch='X'))
    # ax.add_patch(Rectangle((32550 * dt, -10), 3000 * dt, 20, alpha=.5, facecolor='w', hatch='X'))
    (l1,) = plt.plot(
        x[start_plot:stop_plot], outputs[start_plot:stop_plot, 0, 0], label="$y_{fix}$"
    )
    (l2,) = plt.plot(
        x[start_plot:stop_plot], outputs[start_plot:stop_plot, 0, 1], label="$y_1$"
    )
    (l3,) = plt.plot(
        x[start_plot:stop_plot], outputs[start_plot:stop_plot, 0, 2], label="$y_2$"
    )
    legend_1 = plt.legend(
        [l1, l2, l3], ["$y_{fix}$", "$y_1$", "$y_2$"], ncol=3, loc="upper left"
    )

    (l_fix,) = plt.plot(
        x[start_plot:stop_plot],
        inputs[:, 0, 0],
        label="$u_{fix}$",
        linewidth=2,
        linestyle="--",
    )
    (l_in,) = plt.plot(
        x[start_plot:stop_plot],
        inputs[:, 0, 1],
        label="$u_{mod_1}$",
        linewidth=2,
        linestyle="--",
    )
    legend_2 = plt.legend([l_fix, l_in], ["$u_{fix}$", "$u_{mod_1}$"], loc=1, ncol=2)
    plt.gca().add_artist(legend_1)
    plt.yticks(np.linspace(outputs[:, 0, :].min(), outputs[:, 0, :].max(), 10))
    plt.ylim(outputs[:, 0, :].min(), outputs[:, 0, :].max())
    plt.xlim(start_plot * dt, stop_plot * dt)
    plt.ylabel("Magnitude")
    plt.xlabel("Time, s")
    plt.minorticks_on()
    plt.grid(linewidth=1)
    plt.grid(which="minor", linestyle="--", alpha=0.8)
    ax.text(0.5, -0.5, "$Romo_1$", fontsize=14)
    ax.text(2.5, -0.5, "$DM_1$", fontsize=14)
    plt.tight_layout()
    plt.savefig("ChangingTasksDemonstration.eps", format="eps")
    plt.show()


if __name__ == "__main__":
    main()
