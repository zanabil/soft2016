import cv2
import numpy as np
import Queue

class Digits(object):
    '''
        Goes through pixels of the cell and finds the largest connected contour
    '''
    def __init__(self, image):
        self.graph = image.copy()
        self.W, self.H = self.graph.shape
        self.visited = [[False for _ in xrange(
            self.H)] for _ in xrange(self.W)]
        self.digit = [[None for _ in xrange(self.H)] for _ in xrange(self.W)]
        self.find_digit()

    def find_digit(self):
        componentId = 0
        #edges of the cell
        A, C = self.H / 4, 3 * self.H / 4 + 1
        B, D = self.W / 4, 3 * self.W / 4 + 1

        for i in xrange(A, C):
            for j in xrange(B, D):
                if not self.visited[i][j]:
                    self.analyse_pixel(i, j, componentId)
                    componentId += 1
        componentSizes = [0 for _ in xrange(componentId)]
        for row in self.digit:
            for cell in row:
                if cell is not None:
                    componentSizes[cell] += 1

        #take largest contour
        largest = componentSizes.index(max(componentSizes))
        for i in xrange(self.H):
            for j in xrange(self.W):
                self.digit[i][j] = 255 if self.digit[i][j] == largest else 0
        self.digit = np.asarray(self.digit, dtype=np.uint8)

    def analyse_pixel(self, i, j, num):
        q = Queue.Queue()
        q.put((i, j))
        while not q.empty():
            i, j = q.get()
            inValidRow = i not in xrange(0, self.H)
            inValidCol = j not in xrange(0, self.W)
            invalidCell = inValidRow or inValidCol
            invalidPixel = invalidCell or self.graph[i][j] != 255
            if invalidPixel or self.visited[i][j]:
                continue
            self.digit[i][j] = num
            self.visited[i][j] = True
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    q.put((i + di, j + dj))