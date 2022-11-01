import matplotlib.patches as patches
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def parser(line_text: str) -> tuple:
    """
    Function parses text in form:
    ```v_name_1=v1:v_name_2:v2```
    and return (v1, v2)
    :param line_text:
    :return: (v1, v2)
    """
    line_text = line_text.split(":")
    print(line_text)
    v1 = line_text[0].split("=")[1]
    v2 = line_text[1].split("=")[1]
    return float(v1), float(v2)


def main():
    x, y = [], []
    # with open('accuracy_vs_noise.txt', 'r') as f:
    with open(
            r"accuracy_vs_noise.txt",
            "r",
    ) as f:
        while line := f.readline():
            t_x, t_y = parser(line)
            x.append(t_x)
            y.append(t_y)
    fig= plt.figure(figsize=(5, 3))
    ax=fig.add_subplot(111)
    ax.plot(x, y, ".", linestyle="--")
    # ax.plot([.5]*2, [50, 100])
    ax.set_ylabel("Accuracy,%")
    ax.set_xlabel(r"$\sigma$")

    ax.add_patch(
        patches.Rectangle(
            (0, 50), 0.5, 50, edgecolor="grey", facecolor="grey", alpha=0.5, fill=True
        )
    )
    plt.minorticks_on()
    plt.grid(True)
    plt.grid(which='minor', alpha=.5, linestyle='--')
    plt.xlim(0, 2)
    plt.ylim(50, 100)
    plt.tight_layout()
    plt.savefig('AccuracyVsNoise.eps')
    plt.show()


if __name__ == "__main__":
    main()
