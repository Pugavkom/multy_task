import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

plt.rcParams["font.size"] = 12
plt.rcParams["svg.fonttype"] = "none"


def main():
    pca_a_dm = np.load("pca/pca_a_dm.npy")
    pca_v_dm = np.load("pca/pca_v_dm.npy")
    colors_dm = np.load("pca/colors_pca_dm.npy")
    dm_values = np.load("pca/dm_values.npy")
    sp_names = [r"$u_{mod_1} = $" + f"{round(el, 2)}" for el in dm_values]
    colors_dm = np.array(colors_dm)
    last_time = []
    for i in range(colors_dm[-1] + 1):
        last_time.append(np.where(colors_dm == i)[0][-1])
    color = cm.rainbow(np.linspace(0, 1, colors_dm[-1] + 1))
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    plt.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.12),
        arrowprops=dict(arrowstyle="<-", color="b"),
    )
    plt.annotate(
        "$PC_1$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.12),
    )
    plt.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0, -0.01),
        arrowprops=dict(arrowstyle="<-", color="b"),
    )
    plt.annotate(
        "$PC_2$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0, -0.01),
    )
    plt.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.01),
        arrowprops=dict(arrowstyle="<-", color="b"),
    )
    plt.annotate(
        "$PC_3$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.01),
    )
    scatter = ax1.scatter(
        pca_v_dm[:, 0],
        pca_v_dm[:, 1],
        pca_v_dm[:, 2],
        s=0.1,
        c=colors_dm,
        cmap="rainbow",
    )
    for i in range((colors_dm[-1] + 1)):
        indexes = np.where(colors_dm == i)[0]
        ax1.plot(
            pca_v_dm[indexes, 0],
            pca_v_dm[indexes, 1],
            pca_v_dm[indexes, 2],
            linestyle="--",
            c="black",
            linewidth=0.02,
        )
    ax1.plot(pca_v_dm[0, 0], pca_v_dm[0, 1], pca_v_dm[0, 2], "*")
    ax1.text(pca_v_dm[0, 0], pca_v_dm[0, 1], pca_v_dm[0, 2], "Start")
    fig.legend(
        bbox_to_anchor=(0.91, 1),
        handles=scatter.legend_elements()[0],
        labels=sp_names,
        borderaxespad=0,
        ncol=5,
    )
    ax2 = fig.add_subplot(122, projection="3d")

    scatter = ax2.scatter(
        pca_a_dm[:, 0], pca_a_dm[:, 1], pca_a_dm[:, 2], s=1, c=colors_dm, cmap="rainbow"
    )
    ax2.text(pca_a_dm[0, 0], pca_a_dm[0, 1], pca_a_dm[0, 2], "Start")
    ax2.plot(pca_a_dm[0, 0], pca_a_dm[0, 1], pca_a_dm[0, 2], "*")
    for i in range((colors_dm[-1] + 1)):
        indexes = np.where(colors_dm == i)[0]
        ax2.plot(
            pca_a_dm[indexes, 0],
            pca_a_dm[indexes, 1],
            pca_a_dm[indexes, 2],
            linestyle="--",
            c="black",
            linewidth=0.02,
        )
    i = 0
    for point in last_time:
        ax1.plot(
            pca_v_dm[point, 0],
            pca_v_dm[point, 1],
            pca_v_dm[point, 2],
            "*",
            markersize=10,
            c=color[i],
        )
        ax2.plot(
            pca_a_dm[point, 0],
            pca_a_dm[point, 1],
            pca_a_dm[point, 2],
            "*",
            markersize=10,
            c=color[i],
        )
        i += 1
    ax1.annotate(
        "$(a)$",
        xy=(0.1, 0.8),
        xycoords="axes fraction",
        xytext=(0.1, 0.8),
    )
    ax2.annotate(
        "$(b)$",
        xy=(0.1, 0.8),
        xycoords="axes fraction",
        xytext=(0.1, 0.8),
    )
    plt.tight_layout()
    plt.savefig("Pca_dm.eps", format="eps")
    plt.savefig("Pca_dm.svg", format="svg")
    plt.show()
    # plt.show()
    #    plt.close()


if __name__ == "__main__":
    main()
