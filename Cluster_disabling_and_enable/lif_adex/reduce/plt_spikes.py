import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    theme = 'dark'
    matplotlib.rcParams['axes.edgecolor'] = '#ff0000'
    sns.axes_style("darkgrid")
    N = 2999
    n_clusters = 14
    fixation_end = 1100

    s_dm = np.load('s_dm.npy')
    s_go = np.load('s_romo.npy')
    cluster_path = os.path.join('..', '..', '..',
                                fr'models\low_freq\mean_fr_filter_less_v_th_0_45\{N}clusters{n_clusters}.npy')
    clusters = np.load(cluster_path, allow_pickle=True)
    cluster_for_dm = 5  # 6 cluster
    cluster_for_go = 8  # 11 cluster
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412)
    ax3 = plt.subplot(413)
    ax4 = plt.subplot(414)
    with sns.axes_style("whitegrid"):
        sns.heatmap(s_dm[:, 0, clusters[cluster_for_dm]].T, ax=ax1, cbar=False, rasterized=True,
                    cmap=('Greys_r' if theme == 'dark' else 'Greys'))
        sns.heatmap(s_dm[:, 0, clusters[cluster_for_go]].T, ax=ax2, cbar=False, rasterized=True,
                    cmap=('Greys_r' if theme == 'dark' else 'Greys'))
        sns.heatmap(s_go[:, 0, clusters[cluster_for_dm]].T, ax=ax3, cbar=False, rasterized=True,
                    cmap=('Greys_r' if theme == 'dark' else 'Greys'))
        sns.heatmap(s_go[:, 0, clusters[cluster_for_go]].T, ax=ax4, cbar=False, rasterized=True,
                    cmap=('Greys_r' if theme == 'dark' else 'Greys'))
        ax1.plot([fixation_end] * 2, [0, len(clusters[cluster_for_dm])])
        ax2.plot([fixation_end] * 2, [0, len(clusters[cluster_for_go])])
        ax3.plot([fixation_end] * 2, [0, len(clusters[cluster_for_dm])])
        ax4.plot([fixation_end] * 2, [0, len(clusters[cluster_for_go])])

        ax4.set_xlabel('Time, ms')
        ax1.set_ylim(1, len(clusters[cluster_for_dm]) + 1)
        ax2.set_ylim(1, len(clusters[cluster_for_go]) + 1)
        ax3.set_ylim(1, len(clusters[cluster_for_dm]) + 1)
        ax4.set_ylim(1, len(clusters[cluster_for_go]) + 1)
        ax1.set_yticks([*range(0, len(clusters[cluster_for_dm]), 4)], [*range(1, len(clusters[cluster_for_dm]), 4)],
                       rotation=0)
        ax2.set_yticks([*range(0, len(clusters[cluster_for_go]), 4)], [*range(1, len(clusters[cluster_for_go]), 4)],
                       rotation=0)
        ax3.set_yticks([*range(0, len(clusters[cluster_for_dm]), 4)], [*range(1, len(clusters[cluster_for_dm]), 4)],
                       rotation=0)
        ax4.set_yticks([*range(0, len(clusters[cluster_for_go]), 4)], [*range(1, len(clusters[cluster_for_go]), 4)],
                       rotation=0)

        ax1.set_xticks([*range(0, s_go.shape[0] + 100, 100)], [*range(0, s_go.shape[0] + 100, 100)], rotation=0)
        ax2.set_xticks([*range(0, s_go.shape[0] + 100, 100)], [*range(0, s_go.shape[0] + 100, 100)], rotation=0)
        ax3.set_xticks([*range(0, s_go.shape[0] + 100, 100)], [*range(0, s_go.shape[0] + 100, 100)], rotation=0)
        ax4.set_xticks([*range(0, s_go.shape[0] + 100, 100)], [*range(0, s_go.shape[0] + 100, 100)], rotation=0)
        ax1.set_xlim(0, len(s_go))
        ax2.set_xlim(0, len(s_go))
        ax3.set_xlim(0, len(s_go))
        ax4.set_xlim(0, len(s_go))
        axes_list = [ax1, ax2, ax3, ax4]
        list_labels = ['(a)', '(b)', '(c)', '(d)']
        if theme != 'dark':
            ax1.plot([-0.05] * 2, [0, len(clusters[cluster_for_dm]) + 1], c='black')
            ax2.plot([-0.05] * 2, [0, len(clusters[cluster_for_go]) + 1], c='black')
            ax3.plot([-0.05] * 2, [0, len(clusters[cluster_for_dm]) + 1], c='black')
            ax4.plot([-0.05] * 2, [0, len(clusters[cluster_for_go]) + 1], c='black')

        for i, ax in enumerate(axes_list):
            ax.set_ylabel('Neuron\nnumber')
            ax.annotate(
                list_labels[i],
                xy=(0.1, 0.8),
                xycoords="axes fraction",
                xytext=(.96, .6),
                c='w' if theme == 'dark' else 'black'
            )
            if theme != 'dark':
                ax.plot([0, len(s_go)], [-0.01] * 2, c='black')
        plt.tight_layout()
        plt.savefig('spikes_dm_go_6__9__clusters.eps')
        plt.show()


if __name__ == '__main__':
    main()
