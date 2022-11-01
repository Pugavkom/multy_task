import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams["font.size"] = 8
plt.rcParams["hatch.color"] = "#000066"


def main():
    data = np.load("data_chaotic_outputs/chaotic_outputs.npy")
    dt = 1e-3
    x = np.arange(len(data)) * 1e-3
    start_plot = 30000
    stop_plot = 34000
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.add_patch(
        Rectangle((31050 * dt, -10), 1250 * dt, 20, alpha=1, facecolor="#FECACA")
    )
    ax.add_patch(
        Rectangle((32300 * dt, -10), 250 * dt, 20, alpha=1, facecolor="#FEFDCA")
    )
    # plt.fill_between([3000, 10], [31049, 10], alpha=0.0, hatch="X", edgecolor="b", linewidth=0.1)

    ax.add_patch(
        Rectangle((30000 * dt, -10), 1049 * dt, 20, alpha=0.5, facecolor="w", hatch="X")
    )
    ax.add_patch(
        Rectangle((32550 * dt, -10), 3000 * dt, 20, alpha=0.5, facecolor="w", hatch="X")
    )
    plt.plot(
        x[start_plot:stop_plot], data[start_plot:stop_plot, 0, 0], label="$y_{fix}$"
    )
    plt.plot(x[start_plot:stop_plot], data[start_plot:stop_plot, 0, 1], label="$y_1$")
    plt.plot(x[start_plot:stop_plot], data[start_plot:stop_plot, 0, 2], label="$y_2$")
    plt.yticks(np.linspace(data[:, 0, :].min(), data[:, 0, :].max(), 10))
    plt.ylim(data[:, 0, :].min(), data[:, 0, :].max())
    plt.xlim(start_plot * dt, stop_plot * dt)
    plt.ylabel("Magnitude")
    plt.xlabel("Time, s")
    plt.legend()
    plt.minorticks_on()
    plt.tight_layout()
    plt.grid(linewidth=1)
    plt.grid(which="minor", linestyle="--", alpha=0.8)
    plt.savefig("LongOutAndTask.eps", format="eps")


if __name__ == "__main__":
    main()
