import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    theme = 'white'
    matplotlib.rcParams['axes.edgecolor'] = '#ff0000'
    sns.axes_style("darkgrid")
    N = 2999
    n_clusters = 14
    fixation_end = 1100

    s_dm = np.load('s_dm.npy')
    s_go = np.load('s_romo.npy')
    s_romo = np.load('romo_full.npy')
    s_romo_mean = np.mean(s_romo[400:1400, 0, :], axis=0)
    s_romo_mean_s = np.argsort(s_romo_mean)
    print(s_romo_mean_s)
    s_romo_new = np.zeros_like(s_romo)
    show_romo = 256
    for i in range(s_romo.shape[-1]):
        s_romo_new[:, :, i] = s_romo[:, :, s_romo_mean_s[i]]
    s_romo = s_romo_new
    cluster_path = os.path.join('..', '..', '..',
                                fr'models\low_freq\mean_fr_filter_less_v_th_0_45\{N}clusters{n_clusters}.npy')
    clusters = np.load(cluster_path, allow_pickle=True)
    cluster_for_dm = 5  # 6 cluster
    cluster_for_go = 8  # 11 cluster
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(511)
    ax2 = plt.subplot(512)
    ax3 = plt.subplot(513)
    ax4 = plt.subplot(514)
    ax5 = plt.subplot(515)
    with sns.axes_style("whitegrid"):
        sns.heatmap(s_dm[:, 0, clusters[cluster_for_dm]].T, ax=ax1, cbar=False, rasterized=True,
                    cmap=('Greys_r' if theme == 'dark' else 'Greys'))
        sns.heatmap(s_dm[:, 0, clusters[cluster_for_go]].T, ax=ax2, cbar=False, rasterized=True,
                    cmap=('Greys_r' if theme == 'dark' else 'Greys'))
        sns.heatmap(s_go[:, 0, clusters[cluster_for_dm]].T, ax=ax3, cbar=False, rasterized=True,
                    cmap=('Greys_r' if theme == 'dark' else 'Greys'))
        sns.heatmap(s_go[:, 0, clusters[cluster_for_go]].T, ax=ax4, cbar=False, rasterized=True,
                    cmap=('Greys_r' if theme == 'dark' else 'Greys'))
        sns.heatmap(s_romo[:, 0, -show_romo:].T, ax=ax5, cbar=False, rasterized=True,
                    cmap=('Greys_r' if theme == 'dark' else 'Greys'))
        ax1.plot([fixation_end] * 2, [0, len(clusters[cluster_for_dm])], c='b')
        ax2.plot([fixation_end] * 2, [0, len(clusters[cluster_for_go])], c='b')
        ax3.plot([fixation_end] * 2, [0, len(clusters[cluster_for_dm])], c='b')
        ax4.plot([fixation_end] * 2, [0, len(clusters[cluster_for_go])], c='b')
        #ax5.plot([fixation_end] * 2, [0, s_romo.shape[-1]])
        ax5.set_xlabel('Time, ms')
        ax1.set_ylim(1, len(clusters[cluster_for_dm]) + 1)
        ax2.set_ylim(1, len(clusters[cluster_for_go]) + 1)
        ax3.set_ylim(1, len(clusters[cluster_for_dm]) + 1)
        ax4.set_ylim(1, len(clusters[cluster_for_go]) + 1)
        ax5.set_ylim(1, show_romo)
        ax1.set_yticks([*range(0, len(clusters[cluster_for_dm]), 4)], [*range(1, len(clusters[cluster_for_dm]), 4)],
                       rotation=0)
        ax2.set_yticks([*range(0, len(clusters[cluster_for_go]), 4)], [*range(1, len(clusters[cluster_for_go]), 4)],
                       rotation=0)
        ax3.set_yticks([*range(0, len(clusters[cluster_for_dm]), 4)], [*range(1, len(clusters[cluster_for_dm]), 4)],
                       rotation=0)
        ax4.set_yticks([*range(0, len(clusters[cluster_for_go]), 4)], [*range(1, len(clusters[cluster_for_go]), 4)],
                       rotation=0)
        ax5.set_yticks([1, 64, 128, 192, 256], [1, 64, 128, 192, 256])
        ax1.set_xticks([*range(0, s_go.shape[0] + 100, 100)], [*range(0, s_go.shape[0] + 100, 100)], rotation=0)
        ax2.set_xticks([*range(0, s_go.shape[0] + 100, 100)], [*range(0, s_go.shape[0] + 100, 100)], rotation=0)
        ax3.set_xticks([*range(0, s_go.shape[0] + 100, 100)], [*range(0, s_go.shape[0] + 100, 100)], rotation=0)
        ax4.set_xticks([*range(0, s_go.shape[0] + 100, 100)], [*range(0, s_go.shape[0] + 100, 100)], rotation=0)
        ax5.set_xticks([*range(0, len(s_romo) , 200)], [*range(0, len(s_romo) , 200)], rotation=0)
        ax1.set_xlim(0, len(s_go))
        ax2.set_xlim(0, len(s_go))
        ax3.set_xlim(0, len(s_go))
        ax4.set_xlim(0, len(s_go))
        axes_list = [ax1, ax2, ax3, ax4, ax5]
        list_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
        if theme != 'dark':
            ax1.plot([-0.05] * 2, [0, len(clusters[cluster_for_dm]) + 1], c='black')
            ax2.plot([-0.05] * 2, [0, len(clusters[cluster_for_go]) + 1], c='black')
            ax3.plot([-0.05] * 2, [0, len(clusters[cluster_for_dm]) + 1], c='black')
            ax4.plot([-0.05] * 2, [0, len(clusters[cluster_for_go]) + 1], c='black')
            ax5.plot([-0.05] * 2, [0, s_romo.shape[-1]], c='black')

        for i, ax in enumerate(axes_list):
            ax.minorticks_on()
            ax.set_ylabel('Neuron\nnumber')
            ax.annotate(
                list_labels[i],
                xy=(0.1, 0.8),
                xycoords="axes fraction",
                xytext=(.98, .65),
                c='w' if theme == 'dark' else 'black'
            )
            if theme != 'dark':
                ax.plot([0, len(s_go)], [-0.01] * 2, c='black')
        ax5.plot([0, len(s_romo)], [0] * 2, c='black')
        ax5.plot([400] * 2, [0, show_romo], c='r')
        ax5.plot([1400] * 2, [0, show_romo], c='r')
        ax5.plot([2000] * 2, [0, show_romo], c='b')
        plt.tight_layout()
        plt.savefig('spikes_dm_go_6__9__clusters.eps')
        plt.show()


if __name__ == '__main__':
    main()
