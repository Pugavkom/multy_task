import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

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


def main():
    N = 2999
    n_clusters = 14
    n_trials = 10
    data = np.load(f'{N}mse_loses{n_clusters}.npy')
    data = np.array(data)
    data = np.log(data)
    data_new = np.zeros((data.shape[0], data.shape[1] // n_trials))
    for i in range(data.shape[1] // n_trials):
        data_new[:, i] = np.mean(data[:, i * n_trials:(i + 1) * n_trials], axis=1)
    fig = plt.figure()
    ax = plt.subplot(111)
    print(data.shape)
    g = sns.heatmap(data_new, annot=True, cmap='jet_r', ax=ax, cbar=False)
    plt.title('log(MSE)')
    plt.xticks(np.arange(0, len(tasks)) + .5, [el.replace('Task', '') for el in sorted(tasks)], rotation=45)
    plt.ylim(0, data.shape[0])
    plt.yticks(np.arange(data.shape[0]) + .5, ["Full"] + [*range(1, 15)] * 2, rotation=0)
    ax.text(-.8, 9, 'Enable only cluster #', rotation=90, va='center', ha='center')
    ax.text(-.8, 22, 'Disable only cluster #', rotation=90, va='center', ha='center')
    plt.tight_layout()
    plt.savefig('mse_256_14_clusters.eps')


if __name__ == '__main__':
    main()
