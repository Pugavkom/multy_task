import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams["font.size"] = 10
plt.rcParams["hatch.color"] = "#000066"


def plot_zero_outputs(
    data,
    dt,
    ax,
    start_fix,
    start_ans,
):
    start_plot = 28000
    stop_plot = 45000
    x = np.arange(len(data)) * dt
    ax.add_patch(
        Rectangle(
            (start_fix * dt, data[:, 0, :].min()),
            (start_ans - start_fix) * dt,
            data[:, 0, :].max() - data[:, 0, :].min(),
            alpha=1,
            facecolor="#FECACA",
        )
    )
    ax.add_patch(
        Rectangle(
            (start_ans * dt, data[:, 0, :].min()),
            250 * dt,
            data[:, 0, :].max() - data[:, 0, :].min(),
            alpha=1,
            facecolor="#FEFDCA",
        )
    )
    # plt.fill_between([3000, 10], [31049, 10], alpha=0.0, hatch="X", edgecolor="b", linewidth=0.1)

    ax.plot(
        x[start_plot:stop_plot], data[start_plot:stop_plot, 0, 0], label="$y_{fix}$"
    )
    ax.plot(x[start_plot:stop_plot], data[start_plot:stop_plot, 0, 1], label="$y_1$")
    ax.plot(x[start_plot:stop_plot], data[start_plot:stop_plot, 0, 2], label="$y_2$")
    ax.set_yticks(np.linspace(data[:, 0, :].min(), data[:, 0, :].max(), 10))
    ax.set_ylim(data[:, 0, :].min(), data[:, 0, :].max())
    ax.set_xlim(start_plot * dt, stop_plot * dt)
    # ax.legend()


def plot_change_tasks(outputs, inputs, dt, ax):
    x = np.arange(len(outputs)) * dt
    start_plot = 0
    stop_plot = len(x)
    ax.add_patch(
        Rectangle(
            (0 * dt, outputs[:, 0, :].min()),
            1850 * dt,
            outputs[:, 0, :].max() - outputs[:, 0, :].min(),
            alpha=1,
            facecolor="#FECACA",
        )
    )
    ax.add_patch(
        Rectangle(
            (1850 * dt, outputs[:, 0, :].min()),
            (len(x) - 1850) * dt,
            outputs[:, 0, :].max() - outputs[:, 0, :].min(),
            alpha=1,
            facecolor="#FEFDCA",
        )
    )
    (l1,) = ax.plot(
        x[start_plot:stop_plot], outputs[start_plot:stop_plot, 0, 0], label="$y_{fix}$"
    )
    (l2,) = ax.plot(
        x[start_plot:stop_plot], outputs[start_plot:stop_plot, 0, 1], label="$y_1$"
    )
    (l3,) = ax.plot(
        x[start_plot:stop_plot], outputs[start_plot:stop_plot, 0, 2], label="$y_2$"
    )
    # legend_1 = ax.legend([l1, l2, l3], ['$y_{fix}$', '$y_1$', '$y_2$'], ncol=3, loc='upper left')
    # l_fix, = ax.plot(x[start_plot:stop_plot], inputs[:, 0, 0], label='$u_{fix}$', linewidth=2, linestyle='--')
    # l_in, = ax.plot(x[start_plot:stop_plot], inputs[:, 0, 1], label='$u_{mod_1}$', linewidth=2, linestyle='--')
    # legend_2 = ax.legend([l_fix, l_in], ['$u_{fix}$', '$u_{mod_1}$'], loc=1, ncol=2)
    # plt.gca().add_artist(legend_1)
    ax.text(0.3, -0.5, "$Romo_1$", fontsize=14)
    ax.text(2.0, -0.5, "$DM_1$", fontsize=14)
    ax.set_yticks(np.linspace(outputs[:, 0, :].min(), outputs[:, 0, :].max(), 10))
    ax.set_ylim(outputs[:, 0, :].min(), outputs[:, 0, :].max())
    ax.set_xlim(start_plot * dt, stop_plot * dt)


def main():
    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    # free 600
    outputs = np.load("data_chaotic_outputs/chaotic_outputs.npy")
    plot_zero_outputs(outputs, 1e-3, ax1, 31050, 32300)

    # free 256
    outputs = np.load("data_chaotic_outputs/chaotic_outputs_256.npy")
    plot_zero_outputs(outputs, 1e-3, ax3, 31350, 32450)
    # Change 600
    outputs = np.load("data_task_change_outputs/task_change_outputs.npy")
    inputs = np.load("data_task_change_outputs/task_change_inputs.npy")
    plot_change_tasks(outputs, inputs, 1e-3, ax2)

    # Change 256
    outputs = np.load("data_task_change_outputs/task_change_outputs_256.npy")
    inputs = np.load("data_task_change_outputs/task_change_inputs_256.npy")
    plot_change_tasks(outputs, inputs, 1e-3, ax4)

    axes = [ax1, ax2, ax3, ax4]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for i, ax in enumerate(axes):
        ax.grid()
        ax.grid(which="minor", linestyle="--")
        ax.minorticks_on()
        ax.text(
            -0.05, -0.15, labels[i], transform=ax.transAxes, va="bottom", ha="right"
        )

    ax1.set_ylabel("Magnitude")
    ax3.set_ylabel("Magnitude")
    ax3.set_xlabel("Time, ms")
    ax4.set_xlabel("Time, ms")
    lgd = ax2.legend(ncol=1, bbox_to_anchor=(1.4, 1.0))
    plt.tight_layout()
    plt.savefig("Ouputs.svg", bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.savefig("Ouputs.eps", bbox_extra_artists=(lgd,), bbox_inches="tight")


if __name__ == "__main__":
    main()
