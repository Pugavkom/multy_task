from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

all_epochs = 100


def parse_accuracy(file: str) -> Tuple[np.array, np.array]:
    accuracy_list = []
    epochs_list = []
    with open(file) as f:
        while readline := f.readline():
            readline = readline.split(";")
            epoch = int(readline[0].split("=")[-1])
            accuracy = float(readline[1].split("=")[-1])
            accuracy_list.append(accuracy)
            epochs_list.append(epoch)
    return np.array(epochs_list), np.array(accuracy_list)


def plot(file: str, label: str, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy, %")
    epochs, accuracy = parse_accuracy(file)
    max_values = []
    if len(epochs) > all_epochs:
        for i in range(len(epochs) // all_epochs + 1):
            ax.plot(
                epochs[all_epochs * i : all_epochs * (i + 1)],
                accuracy[all_epochs * i : all_epochs * (i + 1)],
                ".",
                linestyle="--",
                label=label,
            )
    else:
        ax.plot(
            epochs,
            accuracy,
            ".",
            linestyle="--",
            label=label,
        )


def main():
    first_file = "classic/accuracy_multy_classic.txt"
    second_file = "random_reconnect/accuracy_multy_classic.txt"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot(first_file, "classic", ax)
    plot(second_file, "random", ax)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
