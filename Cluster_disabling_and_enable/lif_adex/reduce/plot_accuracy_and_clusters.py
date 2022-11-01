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
    plt.rcParams['font.size'] = 10
    N = 2999
    n_clusters = 12
    data = np.load(f'{N}accuracy{n_clusters}.npy')
    data = np.array(data)
    # data = data.T
    # data *= 100
    # data = np.log(data)
    fig = plt.figure()
    ax = plt.subplot(121)
    g = sns.heatmap(data, annot=True, cmap="jet", fmt='.0%', ax=ax, cbar=False)
    # plt.ylim(0, 12)
    plt.title('Accuracy')
    plt.xticks(np.arange(0, len(tasks)) + .5, sorted(tasks), rotation=45)
    plt.ylim(0, data.shape[0])
    plt.yticks(np.arange(data.shape[0]) + .5, ['Full'] + [*range(1, n_clusters + 1)] * 2)
    ax.text(-.8, 7, 'Enable only cluster #', rotation=90, va='center', ha='center')
    ax.text(-.8, 19, 'Disable only cluster #', rotation=90, va='center', ha='center')

    ax = plt.subplot(122)



    plt.tight_layout()
    #plt.savefig('accuracy_256_12_clusters.eps')


if __name__ == '__main__':
    main()
