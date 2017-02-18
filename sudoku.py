import os
import sys
import numpy as np
from imageProcesser import Processer
from networking import NeuralNetwork
from formating import SudokuStr
import pickle


def get_sudoku(image_path):
    with open(os.getcwd() + '/networks/net') as in_file:
        sizes, biases, wts = pickle.load(in_file)
        #print ('\n Network size\n\n{}'.format(sizes))
    net = NeuralNetwork(customValues=(sizes, biases, wts))

    for row in Processer(os.path.abspath(image_path)).cells:
        for cell in row:
            x = net.feedforward(np.reshape(cell, (784, 1)))
            x[0] = 0
            digit = np.argmax(x)
            yield str(digit) if list(x[digit])[0] / sum(x) > 0.8 else '.'


if __name__ == '__main__':

    grid = ''.join(cell for cell in get_sudoku(image_path=str(sys.argv[1])))
    d = SudokuStr(grid)
    print('\n Predicted Grid...\n\n{}'.format(d))
    try:
        print('\nSolving...\n\n{}'.format(d.solve()))
    except ValueError:
        print('No solution found.  Please rescan the puzzle.')
