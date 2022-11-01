import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

plt.rcParams["font.size"] = 12


def main():
    pca_a_romo = np.load("pca/256_1_2_low_freq_low_filter/pca_a_godl.npy")
    pca_v_romo = np.load("pca/256_1_2_low_freq_low_filter/pca_v_godl.npy")
    colors_romo = np.load("pca/256_1_2_low_freq_low_filter/colors_pca_godl.npy")
    sp_names = list(np.load("pca/256_1_2_low_freq_low_filter/godl_sp_names.npy"))
    print(sp_names)
    colors_romo = np.array(colors_romo)
    last_time = []
    for i in range(colors_romo[-1] + 1):
        last_time.append(np.where(colors_romo == i)[0][-1])
    color = cm.rainbow(np.linspace(0, 1, colors_romo[-1] + 1))
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
        pca_v_romo[:, 0],
        pca_v_romo[:, 1],
        pca_v_romo[:, 2],
        s=0.1,
        c=colors_romo,
        cmap="rainbow",
    )
    ax1.plot(pca_v_romo[0, 0], pca_v_romo[0, 1], pca_v_romo[0, 2], "*", markersize=10)
    ax1.text(pca_v_romo[0, 0], pca_v_romo[0, 1], pca_v_romo[0, 2], "Start")
    fig.legend(
        bbox_to_anchor=(0.97, 1),
        handles=scatter.legend_elements()[0],
        labels=sp_names,
        borderaxespad=0,
        ncol=3,
    )
    ax2 = fig.add_subplot(122, projection="3d")

    scatter = ax2.scatter(
        pca_a_romo[::2, 0],
        pca_a_romo[::2, 1],
        pca_a_romo[::2, 2],
        s=1,
        c=colors_romo[::2],
        cmap="rainbow",
    )
    ax2.text(pca_a_romo[0, 0], pca_a_romo[0, 1], pca_a_romo[0, 2], "Start")
    ax2.plot(pca_a_romo[0, 0], pca_a_romo[0, 1], pca_a_romo[0, 2], "*", markersize=10)
    for i in range((colors_romo[-1] + 1)):
        indexes = np.where(colors_romo == i)[0]
        ax2.plot(
            pca_a_romo[indexes, 0],
            pca_a_romo[indexes, 1],
            pca_a_romo[indexes, 2],
            linestyle="--",
            c="black",
            linewidth=0.02,
        )
    i = 0
    for point in last_time:
        ax1.plot(
            pca_v_romo[point, 0],
            pca_v_romo[point, 1],
            pca_v_romo[point, 2],
            "*",
            markersize=10,
            c=color[i],
        )
        ax2.plot(
            pca_a_romo[point, 0],
            pca_a_romo[point, 1],
            pca_a_romo[point, 2],
            "*",
            markersize=20,
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

    ax2.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.16),
        arrowprops=dict(arrowstyle="<-", color="b"),
    )
    ax2.annotate(
        "$PC_2$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.12),
    )
    ax2.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(-0.02, -0.01),
        arrowprops=dict(arrowstyle="<-", color="b"),
    )
    ax2.annotate(
        "$PC_3$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0, -0.01),
    )
    ax2.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.01),
        arrowprops=dict(arrowstyle="<-", color="b"),
    )
    ax2.annotate(
        "$PC_1$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.01),
    )

    plt.tight_layout(w_pad=3)
    ax2.view_init(60, 45)
    ax1.view_init(70, 155)
    # plt.savefig('Pca_gort.eps', format='eps')
    plt.show()
    # plt.show()
    #    plt.close()


if __name__ == "__main__":
    main()
