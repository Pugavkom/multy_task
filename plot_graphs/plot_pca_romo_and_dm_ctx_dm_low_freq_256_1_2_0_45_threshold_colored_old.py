import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

plt.rcParams["font.size"] = 12
plt.rcParams["svg.fonttype"] = "none"


def main():
    name_f = "pca/256_1_2_low_freq_low_filter_2999/"
    pca_a_dm = np.load(f"{name_f}pca_a_dm.npy")
    pca_v_dm = np.load(f"{name_f}pca_v_dm.npy")
    colors_dm = np.load(f"{name_f}colors_pca_dm.npy")
    dm_values = np.load(f"{name_f}dm_values.npy")
    sp_names = [r"$u_{mod_1} = $" + f"{round(el, 2)}" for el in dm_values]
    colors_dm = np.array(colors_dm)
    last_time = []
    for i in range(colors_dm[-1] + 1):
        last_time.append(np.where(colors_dm == i)[0][-1])
    color = cm.rainbow(np.linspace(0, 1, colors_dm[-1] + 1))
    fig = plt.figure(figsize=(8, 15))
    ax1 = fig.add_subplot(321, projection="3d")

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
    ax1.plot(pca_v_dm[0, 0], pca_v_dm[0, 1], pca_v_dm[0, 2], "*", c="black")
    ax1.text(
        pca_v_dm[0, 0],
        pca_v_dm[0, 1],
        pca_v_dm[0, 2],
        "Start",
    )
    ax1.view_init(60, 65)
    fig.legend(
        bbox_to_anchor=(0.9, 0.94),
        handles=scatter.legend_elements()[0],
        labels=sp_names,
        borderaxespad=0,
        ncol=4,
    )
    ax2 = fig.add_subplot(322, projection="3d")

    scatter = ax2.scatter(
        pca_a_dm[:, 0], pca_a_dm[:, 1], pca_a_dm[:, 2], s=1, c=colors_dm, cmap="rainbow"
    )
    ax2.text(pca_a_dm[0, 0], pca_a_dm[0, 1], pca_a_dm[0, 2], "Start")
    ax2.plot(pca_a_dm[0, 0], pca_a_dm[0, 1], pca_a_dm[0, 2], "*", c="black")
    trial = 1100
    delay = 0
    answer = 250
    full_time = trial + delay + answer
    # ax2.text(pca_a_dm[trial, 0], pca_a_dm[trial, 1], pca_a_dm[trial, 2], 'Answer')
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
    ax1.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.06),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax1.annotate(
        "$PC_1$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.06),
    )
    ax1.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(-0.055, -0.0),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax1.annotate(
        "$PC_2$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(-0.15, -0.0),
    )
    ax1.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(-0.02, 0.1),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax1.annotate(
        "$PC_3$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(-0.02, 0.1),
    )

    ax2.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.12),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax2.annotate(
        "$PC_1$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.12),
    )
    ax2.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0, -0.01),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax2.annotate(
        "$PC_2$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0, -0.01),
    )
    ax2.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.01),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax2.annotate(
        "$PC_3$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.01),
    )

    pca_a_dm = np.load(f"{name_f}pca_a_ctxdm.npy")
    pca_v_dm = np.load(f"{name_f}pca_v_ctxdm.npy")
    colors_dm = np.load(f"{name_f}colors_pca_ctxdm.npy")
    dm_values = np.load(f"{name_f}ctxdm_values.npy")
    sp_names = [r"$u_{mod_{1/2}} = $" + f"{round(el, 2)}" for el in dm_values]
    colors_dm = np.array(colors_dm)
    last_time = []
    for i in range(colors_dm[-1] + 1):
        last_time.append(np.where(colors_dm == i)[0][-1])
    color = cm.rainbow(np.linspace(0, 1, colors_dm[-1] + 1))
    ax1 = fig.add_subplot(323, projection="3d")

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
    ax1.plot(pca_v_dm[0, 0], pca_v_dm[0, 1], pca_v_dm[0, 2], "*", c="black")
    ax1.text(
        pca_v_dm[0, 0],
        pca_v_dm[0, 1],
        pca_v_dm[0, 2],
        "Start",
    )
    ax1.view_init(60, 65)
    fig.legend(
        bbox_to_anchor=(0.9, 0.65),
        handles=scatter.legend_elements()[0],
        labels=sp_names,
        borderaxespad=0,
        ncol=4,
    )
    ax2 = fig.add_subplot(324, projection="3d")

    scatter = ax2.scatter(
        pca_a_dm[:, 0], pca_a_dm[:, 1], pca_a_dm[:, 2], s=1, c=colors_dm, cmap="rainbow"
    )
    ax2.text(pca_a_dm[0, 0], pca_a_dm[0, 1], pca_a_dm[0, 2], "Start")
    ax2.plot(pca_a_dm[0, 0], pca_a_dm[0, 1], pca_a_dm[0, 2], "*", c="black")
    trial = 1100
    delay = 0
    answer = 250
    full_time = trial + delay + answer
    # ax2.text(pca_a_dm[trial, 0], pca_a_dm[trial, 1], pca_a_dm[trial, 2], 'Answer')
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
        "(c)",
        xy=(0.1, 0.8),
        xycoords="axes fraction",
        xytext=(0.1, 0.8),
    )
    ax2.annotate(
        "(d)",
        xy=(0.1, 0.8),
        xycoords="axes fraction",
        xytext=(0.1, 0.8),
    )
    ax1.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.06),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax1.annotate(
        "$PC_1$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.06),
    )
    ax1.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(-0.055, -0.0),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax1.annotate(
        "$PC_2$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(-0.15, -0.0),
    )
    ax1.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(-0.02, 0.1),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax1.annotate(
        "$PC_3$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(-0.02, 0.1),
    )

    ax2.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.12),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax2.annotate(
        "$PC_1$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.12),
    )
    ax2.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0, -0.01),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax2.annotate(
        "$PC_2$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0, -0.01),
    )
    ax2.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.01),
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    ax2.annotate(
        "$PC_3$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.01),
    )

    pca_a_romo = np.load(f"{name_f}/pca_a_romo.npy")
    pca_v_romo = np.load(f"{name_f}/pca_v_romo.npy")
    colors_romo = np.load(f"{name_f}/colors_pca_romo.npy")
    sp_names = list(np.load(f"{name_f}/romo_sp_names.npy"))
    print(sp_names)
    colors_romo = np.array(colors_romo)
    last_time = []
    for i in range(colors_romo[-1] + 1):
        last_time.append(np.where(colors_romo == i)[0][-1])
    color = cm.rainbow(np.linspace(0, 1, colors_romo[-1] + 1))
    ax1 = fig.add_subplot(325, projection="3d")
    plt.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.12),
        arrowprops=dict(arrowstyle="<-", color="black"),
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
        arrowprops=dict(arrowstyle="<-", color="black"),
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
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    plt.annotate(
        "$PC_3$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.01),
    )
    scatter2 = ax1.scatter(
        pca_v_romo[:, 0],
        pca_v_romo[:, 1],
        pca_v_romo[:, 2],
        s=0.1,
        c=colors_romo,
        cmap="rainbow",
    )
    ax1.plot(
        pca_v_romo[0, 0],
        pca_v_romo[0, 1],
        pca_v_romo[0, 2],
        "*",
        markersize=10,
        c="black",
    )
    ax1.text(pca_v_romo[0, 0], pca_v_romo[0, 1], pca_v_romo[0, 2], "Start")
    fig.legend(
        bbox_to_anchor=(1, 0.35),
        handles=scatter2.legend_elements()[0],
        labels=sp_names,
        borderaxespad=0.1,
        ncol=3,
    )
    ax2 = fig.add_subplot(326, projection="3d")

    scatter = ax2.scatter(
        pca_a_romo[::2, 0],
        pca_a_romo[::2, 1],
        pca_a_romo[::2, 2],
        s=1,
        c=colors_romo[::2],
        cmap="rainbow",
    )
    ax2.text(pca_a_romo[0, 0], pca_a_romo[0, 1], pca_a_romo[0, 2], "Start")
    ax2.plot(
        pca_a_romo[0, 0],
        pca_a_romo[0, 1],
        pca_a_romo[0, 2],
        "*",
        markersize=10,
        c="black",
    )
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

    #
    trial = 300
    delay = 1000
    answer = 250
    full_time = trial * 2 + delay + answer  # 1000ms + 300ms * 2 + 250ms

    # ax2.text(pca_a_romo[1300, 0], pca_a_romo[1300, 1], pca_a_romo[1300, 2], 'Second stimulus', fontsize=11, va='center',
    #         ha='center')
    # ax2.text(pca_a_romo[1300 + full_time * 4, 0], pca_a_romo[1300 + full_time * 4, 1],
    #         pca_a_romo[1300 + full_time * 4, 2], 'Second stimulus', fontsize=11, va='center', ha='center')

    # ax2.text(pca_a_romo[1300 + trial + 1, 0], pca_a_romo[1300 + trial + 1, 1], pca_a_romo[1300 + trial + 1, 2],
    #         'Answer', fontsize=11, rotation=90)
    # ax2.text(pca_a_romo[1300 + trial + 5 + full_time * 4, 0], pca_a_romo[1300 + trial + 5 + full_time * 4, 1],
    #         pca_a_romo[1300 + trial + 5+ full_time * 4, 2], 'Answer', fontsize=11, )
    ax1.annotate(
        "(e)",
        xy=(0.1, 0.8),
        xycoords="axes fraction",
        xytext=(0.1, 0.8),
    )
    ax2.annotate(
        "(f)",
        xy=(0.1, 0.8),
        xycoords="axes fraction",
        xytext=(0.1, 0.8),
    )

    plt.annotate(
        "",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.12),
        arrowprops=dict(arrowstyle="<-", color="black"),
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
        arrowprops=dict(arrowstyle="<-", color="black"),
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
        arrowprops=dict(arrowstyle="<-", color="black"),
    )
    plt.annotate(
        "$PC_3$",
        xy=(0, -0.1),
        xycoords="axes fraction",
        xytext=(0.1, -0.01),
    )

    # ax2.view_init(60, 45)
    fig.subplots_adjust(hspace=0.6, wspace=0.00001, left=0, right=1)

    plt.savefig("romo_dm_ctx_pca_256.eps", format="eps")
    plt.savefig("romo_dm_ctx_pca_256.pdf", format="pdf")

    plt.show()

    # plt.show()
    #    plt.close()


if __name__ == "__main__":
    main()
