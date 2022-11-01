import sys

import matplotlib.pyplot as plt
from pyparsing import lineStart
from tkinter import Tk 
from tkinter.filedialog import askopenfilename

def main():
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    while root:
        file_name = askopenfilename()
        if not file_name:
            return
        data_matrix = dict()
        with open(file_name) as f:
            l = f.readline()
            while (l != '' or l):
                params, accuracy = tuple(l.split(':'))
                accuracy = float(accuracy) * 100
                params = params.split('_')
                epoch = int(params[1])
                size = int(params[3])
                if size not in data_matrix:
                    data_matrix[size] = dict()
                data_matrix[size][epoch] = accuracy
                l = f.readline()
        plt.figure(figsize=(5, 3))
        for i in data_matrix:
            x, y = [], []
            for key, item in data_matrix[i].items():
                x.append(key)
                y.append(item)
            plt.plot(x, y, label=f'$N = {i}$')
        plt.grid()
        plt.minorticks_on()
        plt.grid(which='minor', linestyle = '--', alpha=.5)
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy, %')
        plt.xlim([0, 3000])
        plt.ylim([0, 100.01])
        plt.legend()
        plt.tight_layout()
        plt.show()
    


if __name__ == '__main__':
    main()
