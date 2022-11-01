import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["font.size"] = 8


def main():
    data = pd.read_csv("J_vs_k_256_both_stages.csv")
    sse = data["$J$"]
    fig = plt.figure(figsize=(5, 4))
    ax2 = plt.subplot(212)
    g = sns.lineplot(data=data, x="$k$", y="$J$", linestyle="--", ax=ax2)
    g.set(
        xlim=(1, len(sse)),
        xticks=range(1, len(sse) + 1, 2),
        ylim=(min(sse), max(sse)),
        yticks=np.arange(min(sse), max(sse) + 50, 50),
    )
    g.minorticks_on()
    g.grid()
    g.grid(which="minor", linestyle="--")
    ax1 = plt.subplot(211)
    data = pd.read_csv("600_J_vs_k_fixation_answers.csv")
    sse = data["$J$"]
    g = sns.lineplot(data=data, x="$k$", y="$J$", hue="Phase", linestyle="--", ax=ax1)
    g.set(
        xlim=(1, len(sse) // 2),
        xticks=range(1, len(sse) // 2 + 1, 2),
        ylim=(min(sse), max(sse)),
        yticks=np.arange(min(sse) - 1, max(sse) + 1000, 1000),
    )
    g.minorticks_on()
    g.grid()
    g.grid(which="minor", linestyle="--")

    ax1.text(-0.05, -0.15, "(a)", transform=ax1.transAxes, va="bottom", ha="right")

    ax2.text(-0.05, -0.25, "(b)", transform=ax2.transAxes, va="bottom", ha="right")
    plt.tight_layout()
    plt.savefig("J_vs_k_600_and_256.eps")


if __name__ == "__main__":
    main()
