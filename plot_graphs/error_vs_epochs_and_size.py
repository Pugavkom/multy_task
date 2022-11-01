import os
import sys

from dotenv import dotenv_values
from matplotlib import pyplot as plt

plt.rcParams["font.size"] = "9"
plt.rcParams["svg.fonttype"] = "none"


def _collect_data(file):
    data = dict()
    while l := file.readline():
        params, accuracy = tuple(l.split(":"))
        accuracy = float(accuracy) * 100
        params = params.split("_")
        epoch = int(params[1])
        size = int(params[3])
        if size not in data:
            data[size] = dict()
        data[size][epoch] = accuracy
    return data


def main():
    config = sys.argv[1]
    conf_file = dotenv_values(config)
    user_folder = conf_file["USER_FOLDER"]
    learning_rates = conf_file["LR"].split(":")
    fig = plt.figure(figsize=(5, 6))
    for j in range(len(learning_rates)):
        ax = plt.subplot(311 + j)
        if j == 0:
            ax.set_title(rf'$\tau_a = {conf_file["TAU"].replace("_", "/")}$')
        lr = learning_rates[j]
        # labels = ['(a)', '(b)', '(c)', '(d)']
        labels = ["(d)", "(e)", "(f)"]
        with open(
            os.path.join(
                user_folder,
                "v_th_" + conf_file["V_TH"],
                "tau_" + conf_file["TAU"],
                "accuracy_" + conf_file["ACCURACY"],
                f"accuracy_{lr}.txt",
            ),
        ) as f:
            data = _collect_data(f)
            max_value = -1000
            size_max = 0
            for i in data:
                x, y = [], []
                for key, item in data[i].items():
                    x.append(key)
                    y.append(item)
                plt.plot(x, y, label=f"$N = {i}$")
                plt.ylabel("Accuracy, %")
                plt.grid()
                plt.minorticks_on()
                plt.grid(which="minor", linestyle="--", alpha=0.5)
                plt.xlim([0, 3000])
                plt.ylim([0, 100.01])
                max_value = max((max_value, max(y)))
                if max_value == max(y):
                    size_max = i
            plt.plot([0, 3000], [max_value] * 2, c="black", linestyle="--")
            plt.text(
                3100, 10, f"max: {round(max_value, 1)}%\n$N = {size_max}$", rotation=90
            )
            plt.text(1600, 5, f'Learning rate: {float(lr.replace("e", "e-"))}')
            ax.text(
                -0.06,
                -0.3,
                labels[j],
                transform=ax.transAxes,
            )
            if j == 0:
                plt.legend(
                    bbox_to_anchor=(0, 1.02, 1, 0.2),
                    loc="lower left",
                    mode="expand",
                    ncol=4,
                )
    plt.xlabel("Number of epochs")

    plt.tight_layout()
    plt.savefig("accuracy.svg")
    plt.show()


if __name__ == "__main__":
    main()
