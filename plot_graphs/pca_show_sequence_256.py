import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 10
cmap = "rainbow"


def show_arrow(alpha, ax, r=1, component=1, text_shift=(0, 0)):
    alpha = np.radians(alpha)
    ax.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(r * np.cos(alpha), r * np.sin(alpha) - 0.1),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )

    ax.annotate(
        f"$PC_{component}$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(
            r * np.cos(alpha) + text_shift[0],
            r * np.sin(alpha) - 0.1 + text_shift[1],
        ),
    )


def plot_limits(data, ax):
    ax.set_xlim(data[:, 0].min(), data[:, 0].max())
    ax.set_ylim(data[:, 1].min(), data[:, 1].max())
    ax.set_zlim(data[:, 2].min(), data[:, 2].max())


def main():
    point_size = 0.05
    pca_a = np.load("pca/sequence_ctx_dm_romo_dm_go_256_v_th_0_45/pca_a.npy")
    pca_v = np.load("pca/sequence_ctx_dm_romo_dm_go_256_v_th_0_45/pca_v.npy")
    colors = np.load("pca/sequence_ctx_dm_romo_dm_go_256_v_th_0_45/colors.npy")
    sp_names = ["CtxDM1", "DM1", "Romo1", "Go1", "GoRt1", "GoDl1"]
    print(sp_names)
    fig = plt.figure(figsize=(5, 3.5))

    ax1 = plt.subplot(121, projection="3d")
    ax2 = plt.subplot(122, projection="3d")

    ax1.scatter(
        pca_v[:, 0], pca_v[:, 1], pca_v[:, 2], c=colors, cmap=cmap, s=point_size
    )
    scatter = ax2.scatter(
        pca_a[:, 0], pca_a[:, 1], pca_a[:, 2], cmap=cmap, c=colors, s=point_size
    )

    ax1.plot(pca_v[0, 0], pca_v[0, 1], pca_v[0, 2], "*", markersize=10, c="black")
    ax2.plot(pca_a[0, 0], pca_a[0, 1], pca_a[0, 2], "*", markersize=10, c="black")

    ax1.text(
        pca_v[0, 0] - 0.6,
        pca_v[0, 1] + 0.3,
        pca_v[0, 2],
        "Start",
        ha="left",
        va="center",
    )
    ax2.text(
        pca_a[0, 0] + 0.3,
        pca_a[0, 1] + 0.3,
        pca_a[0, 2] - 0.5,
        "Start",
        ha="left",
        va="center",
    )
    fig.legend(
        bbox_to_anchor=(0.85, 1),
        handles=scatter.legend_elements()[0],
        labels=sp_names,
        ncol=3,
    )

    ax1.view_init(-75, 65)
    show_arrow(180 - 15, ax1, 0.15, 1)
    show_arrow(65, ax1, 0.1, 2, (0.01, -0.05))
    show_arrow(75, ax1, 0.15, 3)
    plot_limits(pca_v, ax1)
    show_arrow(-10, ax2, 0.1, 1)
    show_arrow(89, ax2, 0.1, 2, (-0.05, -0.01))
    show_arrow(45, ax2, 0.15, 3, (-0.01, -0.025))
    plot_limits(pca_a, ax2)

    ax1.annotate(
        "(a)",
        xy=(0.1, 0.8),
        xycoords="axes fraction",
        xytext=(0.1, 0.8),
    )
    ax2.annotate(
        "(b)",
        xy=(0.1, 0.8),
        xycoords="axes fraction",
        xytext=(0.1, 0.8),
    )

    plt.tight_layout(
        w_pad=0.158,
    )
    fig.subplots_adjust(right=0.948, left=0.064, bottom=0, top=1)
    plt.savefig("tasks_sequence_256.eps")
    plt.show()


if __name__ == "__main__":
    main()
