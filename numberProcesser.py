import numpy as np
import cv2
import pickle

from DigitBuilder import Digits
from Displayer import DisplayImage

class Numbers(object):

    def __init__(self, sudoku):
        self.displayer = DisplayImage()
        self.cells = self.extract_cells(sudoku)

    def extract_cells(self, sudoku):
        cells = []
        W, H = sudoku.shape
        cell_size = W / 9
        i, j = 0, 0
        for r in range(0, W, cell_size):
            row = []
            j = 0
            for c in range(0, W, cell_size):
                cell = sudoku[r:r + cell_size, c:c + cell_size]
                cell = cv2.resize(cell, (28, 28))
                cell = self.clean(cell)
                digit = Digits(cell).digit
                #self.displayer.display_image(digit, 'After clean')
                digit = self.center(digit)
                #self.displayer.display_image(digit, 'After centering')
                row.append(digit // 255)
                j += 1
            cells.append(row)
            i += 1
        return cells

    def clean(self, cell):
        contours, h = cv2.findContours(
            cell.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cell = cv2.resize(cell[y:y + h, x:x + w], (28, 28))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
        cell = 255 * (cell / 130)
        return cell

    def center(self, digit):

        #center x axis
        topLine = 0
        bottomLine = 0
        for i, row in enumerate(digit.copy()):
            if np.any(row):
                topLine = i
                break
        #look from the bottom side upwards
        for i in xrange(digit.shape[0] - 1, -1, -1):
            if np.any(digit[i]):
                bottomLine = i
                break
        if topLine is None or bottomLine is None:
            return digit
        centerLine = (topLine + bottomLine) >> 1
        imageCenter = digit.shape[0] >> 1
        digit = self.shift_row(digit, start=topLine, end=bottomLine, length=imageCenter - centerLine)

        #center y axis
        leftLine = 0
        rightLine = 0
        for i in xrange(digit.shape[1]):
            if np.any(digit[:, i]):
                leftLine = i
                break
        #look from the end backwards
        for i in xrange(digit.shape[1]-1, -1, -1):
            if np.any(digit[:, i]):
                rightLine = i
                break
        if leftLine is None or rightLine is None:
            return digit
        centerLine = (leftLine + rightLine) >> 1
        imageCenter = digit.shape[1] >> 1
        digit = self.shift_column(digit, start=leftLine, end=rightLine, length=imageCenter - centerLine)
        return digit

    def shift_row(self, digit, start, end, length):
        shifted = np.zeros(digit.shape)
        if start + length < 0:
            length = -start
        elif end + length >= digit.shape[0]:
            length = digit.shape[0] - 1 - end

        for row in xrange(start, end + 1):
            shifted[row + length] = digit[row]
        return shifted

    def shift_column(self, digit, start, end, length):
        shifted = np.zeros(digit.shape)
        if start + length < 0:
            length = -start
        elif end + length >= digit.shape[1]:
            length = digit.shape[1] - 1 - end

        for col in xrange(start, end + 1):
            shifted[:, col + length] = digit[:, col]
        return shifted