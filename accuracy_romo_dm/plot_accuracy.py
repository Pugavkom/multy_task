import numpy as np
from matplotlib import pyplot as plt


def parse_line(text: str):
    split_text = text.split(':')
    result = []
    for i in range(len(split_text)):
        result.append(float(split_text[i].split('_')[1]))
    return result


def main():
    data = {}
    max_value = -1000000
    result = None
    with open('check_params_lr_0.01', 'r') as f:
        while line := f.readline():
            split_data = parse_line(line)
            if split_data[-1] > max_value:
                max_value = split_data[-1]
                result = split_data[:-1]
            if split_data[0] not in data:
                data[split_data[0]] = []
            data[split_data[0]].append(split_data[1:])

        print(result, max_value)

        #data = (data[400])

        #print(data[400][0][-1])
        for key in data:
            plot_data = []
            start_epoch = -100
            for el in data[key]:
                if start_epoch != el[0]:

                    start_epoch = el[0]

                    plot_data.append([el[-1],])
                    print(plot_data)
                else:
                    plot_data[-1].append(el[-1])
            tau_start = 2
            tau_stop = 1 / 6.5
            plt.figure()
            plt.imshow(plot_data, aspect='auto', origin='lower', extent=[tau_start, tau_stop, 39, 2439], cmap="jet", vmin=70, vmax=100)
            plt.xlabel(r'$\tau_{ada}, s$')
            plt.colorbar()
            plt.title(f'{key}')
            plt.savefig(f'{key}.png')
            #plt.show()
            plt.close()
        #print(np.matrix(data[2]))

if __name__ == '__main__':
    main()
