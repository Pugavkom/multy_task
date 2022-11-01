import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

tasks = [
    "RomoTask1",
    "RomoTask2",
    "DMTask1",
    "DMTask2",
    "CtxDMTask1",
    "CtxDMTask2",
    "GoTask1",
    "GoTask2",
    "GoRtTask1",
    "GoRtTask2",
    "GoDlTask1",
    "GoDlTask2",
]


# cmap = sns.cm.magma

def main():
    #plt.rcParams['font.size'] = 14
    N = 2999
    n_clusters = 14
    data = np.load(f'{N}accuracy{n_clusters}.npy')
    data = np.array(data)
    # data = data.T
    # data *= 100
    # data = np.log(data)
    fig = plt.figure()
    ax = plt.subplot(111)
    g = sns.heatmap(data, annot=True, cmap="Greys_r", fmt='.0%', ax=ax, cbar=False)
    # plt.ylim(0, 12)
    plt.title('Точность работы сети')
    plt.xticks(np.arange(0, len(tasks)) + .5, sorted(tasks), rotation=45)
    plt.ylim(0, data.shape[0])
    plt.yticks(np.arange(data.shape[0]) + .5, ['Full'] + [*range(1, n_clusters + 1)] * 2, rotation=0)
    ax.text(-.8, 9, 'Включенный\nкластер', rotation=90, va='center', ha='center')
    ax.text(-.8, 22, 'Выключеный\n кластер', rotation=90, va='center', ha='center')
    plt.tight_layout()
    plt.savefig('accuracy_256_14_clusters.eps')

if __name__ == '__main__':
    main()
