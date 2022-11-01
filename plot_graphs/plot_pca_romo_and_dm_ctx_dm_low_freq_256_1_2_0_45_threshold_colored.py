import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.pyplot import cm
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

plt.rcParams["font.size"] = 12
plt.rcParams["svg.fonttype"] = "none"


class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)


def _annotate3D(ax, text, xyz, *args, **kwargs):
    """Add anotation `text` to an `Axes3d` instance."""

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)


setattr(Axes3D, "annotate3D", _annotate3D)


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
    s_points = 0.05
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
    fig = plt.figure(figsize=(8, 13))
    ax1 = fig.add_subplot(321, projection="3d")
    colors_lin = cm.rainbow(np.linspace(0, 1, len(dm_values)))

    first_trial = np.where(colors_dm == colors_dm[0])[0]
    second_trial = np.where(colors_dm == len(dm_values) - 1)[0]
    scatter = ax1.scatter(
        pca_v_dm[first_trial, 0],
        pca_v_dm[first_trial, 1],
        pca_v_dm[first_trial, 2],
        s=s_points,
        c=colors_lin[0],
        cmap="rainbow",
    )
    scatter = ax1.scatter(
        pca_v_dm[second_trial, 0],
        pca_v_dm[second_trial, 1],
        pca_v_dm[second_trial, 2],
        s=s_points,
        c=colors_lin[-1],
    )

    ax1.plot(pca_v_dm[0, 0], pca_v_dm[0, 1], pca_v_dm[0, 2], "*", c="black")
    ax1.text(
        pca_v_dm[0, 0],
        pca_v_dm[0, 1],
        pca_v_dm[0, 2],
        "Start",
        fontsize=10,
    )
    ax1.view_init(60, 65)
    # fig.legend(
    #    bbox_to_anchor=(.9, .94),
    #    handles=scatter.legend_elements()[0],
    #    labels=sp_names,
    #    borderaxespad=0,
    #    ncol=4,
    # )
    ax2 = fig.add_subplot(322, projection="3d")

    scatter = ax2.scatter(
        pca_a_dm[first_trial, 0],
        pca_a_dm[first_trial, 1],
        pca_a_dm[first_trial, 2],
        s=1,
        c=colors_lin[0],
        cmap="rainbow",
    )
    scatter = ax2.scatter(
        pca_a_dm[second_trial, 0],
        pca_a_dm[second_trial, 1],
        pca_a_dm[second_trial, 2],
        s=1,
        c=colors_lin[-1],
        cmap="rainbow",
    )
    ax2.text(
        pca_a_dm[0, 0] + 0.1,
        pca_a_dm[0, 1] - 1.1,
        pca_a_dm[0, 2] + 0.1,
        "Start",
        fontsize=10,
    )
    ax2.plot(pca_a_dm[0, 0], pca_a_dm[0, 1], pca_a_dm[0, 2], "*", c="black")
    trial = 1100
    delay = 0
    answer = 250
    full_time = trial + delay + answer
    # ax2.text(pca_a_dm[trial, 0], pca_a_dm[trial, 1], pca_a_dm[trial, 2], 'Answer')

    ax1.plot(
        pca_v_dm[first_trial, 0],
        pca_v_dm[first_trial, 1],
        pca_v_dm[first_trial, 2],
        linestyle="--",
        c="black",
        linewidth=0.02,
    )
    ax1.plot(
        pca_v_dm[second_trial, 0],
        pca_v_dm[second_trial, 1],
        pca_v_dm[second_trial, 2],
        linestyle="--",
        c="black",
        linewidth=0.02,
    )
    i = 0
    for point in last_time:
        ax1.plot(
            pca_v_dm[point - 150, 0],
            pca_v_dm[point - 150, 1],
            pca_v_dm[point - 150, 2],
            "*",
            markersize=10,
            c=color[i],
            label=sp_names[i],
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
    show_arrow(180 + 18, ax1, 0.1, 1, text_shift=(-0.05, 0.041))
    show_arrow(120 - 180, ax1, 0.16, 2)
    show_arrow(98, ax1, 0.08, 3)
    plot_limits(pca_v_dm, ax1)

    show_arrow(-10, ax2, 0.1, 1)
    show_arrow(89, ax2, 0.1, 2)
    show_arrow(45, ax2, 0.15, 3)
    plot_limits(pca_a_dm, ax2)
    ax1.text(2.24, -6, -2, "Stimulus\nphase", fontsize=10)
    ax1.text(9.5, 16.6, 8, "Response\nphase", fontsize=10, zdir="x")

    ax1.scatter(
        pca_v_dm[first_trial[-250], 0],
        pca_v_dm[first_trial[-250], 1],
        pca_v_dm[first_trial[-250], 2],
        "o",
        c="black",
        s=20,
    )
    ax1.scatter(
        pca_v_dm[second_trial[-250], 0],
        pca_v_dm[second_trial[-250], 1],
        pca_v_dm[second_trial[-250], 2],
        "o",
        c="black",
        s=20,
    )

    ax2.scatter(
        pca_a_dm[first_trial[-250], 0],
        pca_a_dm[first_trial[-250], 1],
        pca_a_dm[first_trial[-250], 2],
        "o",
        c="black",
        s=40,
    )
    ax2.scatter(
        pca_a_dm[second_trial[-250], 0],
        pca_a_dm[second_trial[-250], 1],
        pca_a_dm[second_trial[-250], 2],
        "o",
        c="black",
        s=40,
    )

    ax1.annotate3D(
        "",
        (
            pca_v_dm[first_trial[-250], 0],
            pca_v_dm[first_trial[-250], 1],
            pca_v_dm[first_trial[-250], 2],
        ),
        xytext=(-85, -55),
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )
    ax1.annotate3D(
        "Start response",
        (pca_v_dm[-250, 0], pca_v_dm[-250, 1], pca_v_dm[-250, 2]),
        xytext=(-80, -65),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        rotation=-90
    )
    ax2.annotate3D(
        "",
        # "this",
        (pca_a_dm[-250, 0], pca_a_dm[-250, 1], pca_a_dm[-250, 2]),
        xytext=(-20, -2),
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )
    ax2.annotate3D(
        "Start response",
        (
            pca_a_dm[first_trial[-250], 0],
            pca_a_dm[first_trial[-250], 1],
            pca_a_dm[first_trial[-250], 2],
        ),
        xytext=(-60, -80),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
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
    colors_lin = cm.rainbow(np.linspace(0, 1, len(sp_names)))
    colors_lin = np.concatenate(
        (colors_lin, cm.cividis(np.linspace(0, 1, len(sp_names))))
    )

    first_trial = np.where(colors_dm == colors_dm[0])[0]
    second_trial = np.where(colors_dm == colors_dm[-1])[0]
    scatter = ax1.scatter(
        pca_v_dm[first_trial[: len(first_trial) // 2], 0],
        pca_v_dm[first_trial[: len(first_trial) // 2], 1],
        pca_v_dm[first_trial[: len(first_trial) // 2], 2],
        s=s_points,
        c=colors_lin[0],
    )
    scatter = ax1.scatter(
        pca_v_dm[first_trial[len(first_trial) // 2:], 0],
        pca_v_dm[first_trial[len(first_trial) // 2:], 1],
        pca_v_dm[first_trial[len(first_trial) // 2:], 2],
        s=s_points,
        c=colors_lin[len(colors_lin) // 2],
    )

    scatter = ax1.scatter(
        pca_v_dm[second_trial[: len(second_trial) // 2], 0],
        pca_v_dm[second_trial[: len(second_trial) // 2], 1],
        pca_v_dm[second_trial[: len(second_trial) // 2], 2],
        s=s_points,
        c=colors_lin[len(colors_lin) // 2 - 1],
    )
    scatter = ax1.scatter(
        pca_v_dm[second_trial[len(second_trial) // 2:], 0],
        pca_v_dm[second_trial[len(second_trial) // 2:], 1],
        pca_v_dm[second_trial[len(second_trial) // 2:], 2],
        s=s_points,
        c=colors_lin[-1],
    )
    ax1.plot(
        pca_v_dm[first_trial[len(first_trial) // 2:], 0],
        pca_v_dm[first_trial[len(first_trial) // 2:], 1],
        pca_v_dm[first_trial[len(first_trial) // 2:], 2],
        linestyle="--",
        c="black",
        linewidth=0.02,
    )
    ax1.plot(
        pca_v_dm[second_trial[: len(second_trial) // 2], 0],
        pca_v_dm[second_trial[: len(second_trial) // 2], 1],
        pca_v_dm[second_trial[: len(second_trial) // 2], 2],
        linestyle="--",
        c="black",
        linewidth=0.02,
    )
    ax1.plot(
        pca_v_dm[second_trial[len(second_trial) // 2:], 0],
        pca_v_dm[second_trial[len(second_trial) // 2:], 1],
        pca_v_dm[second_trial[len(second_trial) // 2:], 2],
        linestyle="--",
        c="black",
        linewidth=0.02,
    )
    ax1.plot(
        pca_v_dm[first_trial[: len(first_trial) // 2], 0],
        pca_v_dm[first_trial[: len(first_trial) // 2], 1],
        pca_v_dm[first_trial[: len(first_trial) // 2], 2],
        linestyle="--",
        c="black",
        linewidth=0.02,
    )
    ax1.plot(
        pca_v_dm[first_trial[: len(first_trial) // 2], 0],
        pca_v_dm[first_trial[: len(first_trial) // 2], 1],
        pca_v_dm[first_trial[: len(first_trial) // 2], 2],
        linestyle="--",
        c="black",
        linewidth=0.02,
    )
    ax1.plot(pca_v_dm[0, 0], pca_v_dm[0, 1], pca_v_dm[0, 2], "*", c="black")
    ax1.text(
        pca_v_dm[0, 0],
        pca_v_dm[0, 1] + 0.5,
        pca_v_dm[0, 2] - 0.5,
        "Start",
        fontsize=10,
    )
    ax1.view_init(60, 65)

    ax2 = fig.add_subplot(324, projection="3d")
    ax1.view_init(0, 30)
    ax2.scatter(
        pca_a_dm[first_trial[: len(first_trial) // 2], 0],
        pca_a_dm[first_trial[: len(first_trial) // 2], 1],
        pca_a_dm[first_trial[: len(first_trial) // 2], 2],
        s=0.1,
        c=colors_lin[0],
    )
    ax2.scatter(
        pca_a_dm[first_trial[len(first_trial) // 2:], 0],
        pca_a_dm[first_trial[len(first_trial) // 2:], 1],
        pca_a_dm[first_trial[len(first_trial) // 2:], 2],
        s=0.1,
        c=colors_lin[len(colors_lin) // 2],
    )
    ax2.scatter(
        pca_a_dm[second_trial[: len(second_trial) // 2], 0],
        pca_a_dm[second_trial[: len(second_trial) // 2], 1],
        pca_a_dm[second_trial[: len(second_trial) // 2], 2],
        s=0.1,
        c=colors_lin[len(colors_lin) // 2 - 1],
    )
    ax2.scatter(
        pca_a_dm[second_trial[len(second_trial) // 2:], 0],
        pca_a_dm[second_trial[len(second_trial) // 2:], 1],
        pca_a_dm[second_trial[len(second_trial) // 2:], 2],
        s=0.1,
        c=colors_lin[-1],
    )
    ax2.text(pca_a_dm[0, 0] + 0.6, pca_a_dm[0, 1], pca_a_dm[0, 2], "Start", fontsize=10)
    ax2.plot(pca_a_dm[0, 0], pca_a_dm[0, 1], pca_a_dm[0, 2], "*", c="black")
    trial = 1100
    delay = 0
    answer = 250
    full_time = trial + delay + answer
    # ax2.text(pca_a_dm[trial, 0], pca_a_dm[trial, 1], pca_a_dm[trial, 2], 'Answer')
    handles = []
    i = 0
    for point in last_time:
        print(i, point)
        ax1.plot(
            pca_v_dm[point - 150, 0],
            pca_v_dm[point - 150, 1],
            pca_v_dm[point - 150, 2],
            "*",
            markersize=10,
            c=colors_lin[i + len(dm_values)],
        )
        ax1.plot(
            pca_v_dm[point - (len(pca_a_dm) // 2) - 150, 0],
            pca_v_dm[point - (len(pca_a_dm) // 2) - 150, 1],
            pca_v_dm[point - (len(pca_a_dm) // 2) - 150, 2],
            "*",
            markersize=10,
            c=colors_lin[i],
        )
        (l,) = ax2.plot(
            pca_a_dm[point, 0],
            pca_a_dm[point, 1],
            pca_a_dm[point, 2],
            "*",
            markersize=10,
            c=colors_lin[i + len(dm_values)],
            label=sp_names[i],
        )
        handles.append(l)
        (l,) = ax2.plot(
            pca_a_dm[point - (len(pca_a_dm) // 2), 0],
            pca_a_dm[point - (len(pca_a_dm) // 2), 1],
            pca_a_dm[point - (len(pca_a_dm) // 2), 2],
            "*",
            markersize=10,
            c=colors_lin[i],
            label=sp_names[i],
        )
        handles.append(l)
        i += 1
    ax1.text(-3.7, 3.2, -3.7, "Second\ncontext", fontsize=10)
    ax1.text(-9, 1, 3, "First\ncontext", fontsize=10)
    ax2.text(-4.4, 0.1, 3.3, "Second\ncontext", fontsize=10)
    ax2.text(6, -2.9, 4.3, "First\ncontext", fontsize=10)
    fig.legend(
        [(handles[i], handles[i + 1]) for i in range(0, len(sp_names) * 2, 2)],
        [sp_names[i] for i in range(0, len(sp_names), 1)],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        bbox_to_anchor=(0.935, 0.7),
        ncol=4,
    )
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
    show_arrow(15 + 180, ax1, 0.1, 1, (-0.05, 0.05))
    show_arrow(0, ax1, 0.15, 2)
    show_arrow(90, ax1, 0.15, 3)
    plot_limits(pca_v_dm, ax1)

    show_arrow(-10, ax2, 0.1, 1)
    show_arrow(89, ax2, 0.1, 2)
    show_arrow(45, ax2, 0.15, 3)
    plot_limits(pca_a_dm, ax2)

    ax1.scatter(
        pca_v_dm[first_trial[-250], 0],
        pca_v_dm[first_trial[-250], 1],
        pca_v_dm[first_trial[-250], 2],
        "o",
        c="black",
        s=20,
    )
    ax1.scatter(
        pca_v_dm[second_trial[-250], 0],
        pca_v_dm[second_trial[-250], 1],
        pca_v_dm[second_trial[-250], 2],
        "o",
        c="black",
        s=20,
    )

    ax1.scatter(
        pca_v_dm[first_trial[-250 - len(first_trial) // 2], 0],
        pca_v_dm[first_trial[-250 - len(first_trial) // 2], 1],
        pca_v_dm[first_trial[-250 - len(first_trial) // 2], 2],
        "o",
        c="black",
        s=20,
    )
    ax1.scatter(
        pca_v_dm[second_trial[-250 - len(second_trial) // 2], 0],
        pca_v_dm[second_trial[-250 - len(second_trial) // 2], 1],
        pca_v_dm[second_trial[-250 - len(second_trial) // 2], 2],
        "o",
        c="black",
        s=20,
    )

    ax2.scatter(
        pca_a_dm[first_trial[-250], 0],
        pca_a_dm[first_trial[-250], 1],
        pca_a_dm[first_trial[-250], 2],
        "o",
        c="black",
        s=40,
    )
    ax2.scatter(
        pca_a_dm[second_trial[-250], 0],
        pca_a_dm[second_trial[-250], 1],
        pca_a_dm[second_trial[-250], 2],
        "o",
        c="black",
        s=40,
    )

    ax2.scatter(
        pca_a_dm[first_trial[-250 - len(first_trial) // 2], 0],
        pca_a_dm[first_trial[-250 - len(first_trial) // 2], 1],
        pca_a_dm[first_trial[-250 - len(first_trial) // 2], 2],
        "o",
        c="black",
        s=40,
    )
    ax2.scatter(
        pca_a_dm[second_trial[-250 - len(second_trial) // 2], 0],
        pca_a_dm[second_trial[-250 - len(second_trial) // 2], 1],
        pca_a_dm[second_trial[-250 - len(second_trial) // 2], 2],
        "o",
        c="black",
        s=40,
    )

    ax1.annotate3D(
        "Start response",
        (
            pca_v_dm[first_trial[-250], 0],
            pca_v_dm[first_trial[-250], 1],
            pca_v_dm[first_trial[-250], 2],
        ),
        xytext=(-50, 30),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax1.annotate3D(
        "",
        (
            pca_v_dm[second_trial[-250], 0],
            pca_v_dm[second_trial[-250], 1],
            pca_v_dm[second_trial[-250], 2],
        ),
        xytext=(-60, 35),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax1.annotate3D(
        "",
        (
            pca_v_dm[first_trial[-250 - len(first_trial) // 2], 0],
            pca_v_dm[first_trial[-250 - len(first_trial) // 2], 1],
            pca_v_dm[first_trial[-250 - len(first_trial) // 2], 2],
        ),
        xytext=(-42, 20),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax1.annotate3D(
        "",
        (
            pca_v_dm[second_trial[-250 - len(second_trial) // 2], 0],
            pca_v_dm[second_trial[-250 - len(second_trial) // 2], 1],
            pca_v_dm[second_trial[-250 - len(second_trial) // 2], 2],
        ),
        xytext=(-60, 30),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax2.annotate3D(
        "Start response",
        (
            pca_a_dm[first_trial[-250], 0],
            pca_a_dm[first_trial[-250], 1],
            pca_a_dm[first_trial[-250], 2],
        ),
        xytext=(120, -10),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax2.annotate3D(
        "",
        (
            pca_a_dm[second_trial[-250], 0],
            pca_a_dm[second_trial[-250], 1],
            pca_a_dm[second_trial[-250], 2],
        ),
        xytext=(40, 10),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax2.annotate3D(
        "",
        (
            pca_a_dm[first_trial[-250 - len(first_trial) // 2], 0],
            pca_a_dm[first_trial[-250 - len(first_trial) // 2], 1],
            pca_a_dm[first_trial[-250 - len(first_trial) // 2], 2],
        ),
        xytext=(-40, 24),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax2.annotate3D(
        "",
        (
            pca_a_dm[second_trial[-250 - len(second_trial) // 2], 0],
            pca_a_dm[second_trial[-250 - len(second_trial) // 2], 1],
            pca_a_dm[second_trial[-250 - len(second_trial) // 2], 2],
        ),
        xytext=(-50, 49),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    pca_a_romo = np.load(f"{name_f}/pca_a_romo.npy")
    pca_v_romo = np.load(f"{name_f}/pca_v_romo.npy")
    colors_romo = np.load(f"{name_f}/colors_pca_romo.npy")
    sp_names = list(np.load(f"{name_f}/romo_sp_names.npy"))
    colors_romo = np.array(colors_romo)
    first_trial = np.where(colors_romo == colors_romo[0])[0]
    second_trial = np.where(colors_romo == colors_romo[-1])[0]
    colors_romo = np.array(colors_romo)
    colors_lin = cm.rainbow(np.linspace(0, 1, len(colors_romo)))
    last_time = []
    for i in range(colors_romo[-1] + 1):
        last_time.append(np.where(colors_romo == i)[0][-1])
    color = cm.rainbow(np.linspace(0, 1, colors_romo[-1] + 1))
    ax1 = fig.add_subplot(325, projection="3d")
    ax1.view_init(0, 30)

    scatter2 = ax1.scatter(
        pca_v_romo[first_trial, 0],
        pca_v_romo[first_trial, 1],
        pca_v_romo[first_trial, 2],
        s=0.05,
        c=colors_lin[0],
    )
    scatter2 = ax1.scatter(
        pca_v_romo[second_trial, 0],
        pca_v_romo[second_trial, 1],
        pca_v_romo[second_trial, 2],
        s=0.05,
        c=colors_lin[-1],
    )

    ax1.plot(pca_v_romo[0, 0], pca_v_romo[0, 1], pca_v_romo[0, 2], "*", c="black")
    ax1.text(
        pca_v_romo[0, 0],
        pca_v_romo[0, 1] + 0.5,
        pca_v_romo[0, 2] - 0.5,
        "Start",
        fontsize=10,
    )
    ax1.plot(
        pca_v_romo[first_trial, 0],
        pca_v_romo[first_trial, 1],
        pca_v_romo[first_trial, 2],
        linestyle="--",
        c="black",
        linewidth=0.02,
    )
    ax1.plot(
        pca_v_romo[second_trial, 0],
        pca_v_romo[second_trial, 1],
        pca_v_romo[second_trial, 2],
        linestyle="--",
        c="black",
        linewidth=0.02,
    )
    ax2 = fig.add_subplot(326, projection="3d")

    scatter = ax2.scatter(
        pca_a_romo[first_trial, 0],
        pca_a_romo[first_trial, 1],
        pca_a_romo[first_trial, 2],
        s=s_points,
        c=colors_lin[0],
    )
    scatter = ax2.scatter(
        pca_a_romo[second_trial, 0],
        pca_a_romo[second_trial, 1],
        pca_a_romo[second_trial, 2],
        s=s_points,
        c=colors_lin[-1],
    )
    ax2.text(pca_a_romo[0, 0], pca_a_romo[0, 1], pca_a_romo[0, 2], "Start", fontsize=10)
    ax2.plot(pca_a_romo[0, 0], pca_a_romo[0, 1], pca_a_romo[0, 2], "*", c="black")
    i = 0
    handles = []
    for point in last_time:
        ax1.plot(
            pca_v_romo[point - 150, 0],
            pca_v_romo[point - 150, 1],
            pca_v_romo[point - 150, 2],
            "*",
            markersize=10,
            c=color[i],
        )
        (l,) = ax2.plot(
            pca_a_romo[point - 150, 0],
            pca_a_romo[point - 150, 1],
            pca_a_romo[point - 150, 2],
            "*",
            markersize=10,
            c=color[i],
            label=sp_names[i],
        )
        handles.append(l)
        i += 1
    fig.legend(
        bbox_to_anchor=(1, 0.35),
        handles=handles,
        ncol=3,
    )

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
    ax1.view_init(45, 45)
    ax1.text(6.8, 2.7, -1.66, "Stimulus phase", fontsize=10, zdir="x")
    ax1.text(-1.0, -4.4, 2, "Response phase", fontsize=10, zdir=(-0.7, 0.8, 0))
    show_arrow(180 + 45, ax1, 0.1, 1, (-0.05, 0.05))
    show_arrow(139, ax1, 0.15, 2)
    show_arrow(95, ax1, 0.15, 3)
    plot_limits(pca_v_romo, ax1)

    show_arrow(-10, ax2, 0.1, 1)
    show_arrow(89, ax2, 0.1, 2)
    show_arrow(45, ax2, 0.15, 3)
    plot_limits(pca_a_romo, ax2)
    ax1.scatter(
        pca_v_romo[first_trial[-550:-250], 0],
        pca_v_romo[first_trial[-550:-250], 1],
        pca_v_romo[first_trial[-550:-250], 2],
        linewidths=0.1,
        facecolors="none",
        edgecolors="b",
    )

    ax1.scatter(
        pca_v_romo[second_trial[-550:-250], 0],
        pca_v_romo[second_trial[-550:-250], 1],
        pca_v_romo[second_trial[-550:-250], 2],
        linewidths=0.1,
        facecolors="none",
        edgecolors="b",
    )
    ax1.annotate3D(
        "Second stimulus",
        (-1.7, -7.24, -3),
        xytext=(-10, 40),
        fontsize=10,
        c="w",
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="blue", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )
    ax1.annotate3D(
        "",
        (-4.6, -1.4, -2.2),
        xytext=(-78, 52),
        fontsize=10,
        c="w",
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="blue", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )
    ax1.annotate3D(
        "First stimulus",
        (10.4, 4.62, 6.14),
        xytext=(-50, 10),
        fontsize=10,
        c="w",
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="blue", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax2.plot(
        pca_a_romo[first_trial[-550], 0],
        pca_a_romo[first_trial[-550], 1],
        pca_a_romo[first_trial[-550], 2],
        "+",
        c="blue",
        markersize=12,
    )

    ax2.plot(
        pca_a_romo[second_trial[-550], 0],
        pca_a_romo[second_trial[-550], 1],
        pca_a_romo[second_trial[-550], 2],
        "+",
        c="blue",
        markersize=12,
    )

    ax2.plot(
        pca_a_romo[first_trial[300], 0],
        pca_a_romo[first_trial[300], 1],
        pca_a_romo[first_trial[300], 2],
        "+",
        c="blue",
        markersize=12,
    )

    ax2.plot(
        pca_a_romo[second_trial[300], 0],
        pca_a_romo[second_trial[300], 1],
        pca_a_romo[second_trial[300], 2],
        "+",
        c="blue",
        markersize=12,
    )

    ax2.scatter(
        pca_a_romo[first_trial[-250], 0],
        pca_a_romo[first_trial[-250], 1],
        pca_a_romo[first_trial[-250], 2],
        "o",
        c="black",
        s=40,
    )
    ax2.scatter(
        pca_a_romo[second_trial[-250], 0],
        pca_a_romo[second_trial[-250], 1],
        pca_a_romo[second_trial[-250], 2],
        "o",
        c="black",
        s=40,
    )
    ax2.annotate3D(
        "Delay",
        (2.24, -2.9, -0.012),
        xytext=(10, -30),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="red", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )
    ax2.annotate3D(
        "",
        (3.18, -2.7, 0.53),
        xytext=(-20, -45),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax2.annotate3D(
        "First stimulus",
        (-8.23, 1.68, 0.56),
        xytext=(-20, -60),
        fontsize=10,
        c="w",
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="blue", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax2.annotate3D(
        "",
        (-8.68, 1.57, -0.33),
        xytext=(-17, -22),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax2.annotate3D(
        "Second stimulus",
        (1.8, 1.44, 0.54),
        xytext=(0, 25),
        fontsize=10,
        c="w",
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="blue", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax2.annotate3D(
        "",
        (3.14, -1.77, 1.08),
        xytext=(-30, 50),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax2.annotate3D(
        "Start response",
        (pca_a_romo[-250, 0], pca_a_romo[-250, 1], pca_a_romo[-250, 2]),
        xytext=(-30, -50),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )
    ax2.annotate3D(
        "",
        (
            pca_a_romo[first_trial[-250], 0],
            pca_a_romo[first_trial[-250], 1],
            pca_a_romo[first_trial[-250], 2],
        ),
        xytext=(-160, -90),
        fontsize=10,
        textcoords="offset points",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )
    #
    # ax2.view_init(60, 45)
    plt.tight_layout(h_pad=0.5)
    fig.subplots_adjust(hspace=0.6, wspace=0.1, left=0.062, right=0.955, bottom=0.04)

    plt.savefig("romo_dm_ctx_pca_256.eps", format="eps")
    # plt.savefig('romo_dm_ctx_pca_256.pdf', format='pdf')

    plt.show()
    # plt.close()
    # plt.plot(pca_a_romo[first_trial, 1])
    # plt.show()
    #    plt.close()


if __name__ == "__main__":
    main()
