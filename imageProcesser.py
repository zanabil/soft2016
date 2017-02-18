import cv2
import sys
import numpy as np
import os
from Displayer import DisplayImage
from numberProcesser import Numbers

class Processer(object):

    def __init__(self, path):
        self.displayer = DisplayImage()
        print str(os.path.abspath(path))
        self.image = cv2.imread(str(os.path.abspath(path)))
        if self.image is None:
            raise IOError('Image not loaded')
        self.displayer.display_image(self.image, 'Starting image')
        self.thresholds()
        self.displayer.display_image(self.image, 'After threshold')
        board = self.find_sudoku()
        self.displayer.display_image(board, 'After cropping grid')
        board = self.fix_position(board)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        #board = cv2.morphologyEx(board, cv2.MORPH_OPEN, kernel)
        self.displayer.display_image(board, 'After warping position')
        self.cells = Numbers(board).cells

    def thresholds(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #self.image = cv2.fastNlMeansDenoising(self.image, None, 10, 21, 7)
        self.image = cv2.adaptiveThreshold(self.image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 3)
        self.image = 255 - self.image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        #self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)


    def find_sudoku(self):
        contours, h = cv2.findContours(self.image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mainContour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(mainContour)
        resultImage = self.image.copy()
        resultImage = resultImage[y:y + h, x:x + w]
        board = cv2.resize(resultImage, (min(resultImage.shape), min(resultImage.shape)))
        return board

    def fix_position(self, board):
        largestBoard = self.largest4SideContour(board.copy())
        if largestBoard is not None:
            app = self.approx(largestBoard)
            rectangle = self.get_rectangle_corners(app)
            board = self.warp_perspective(rectangle, board)
        return board

    def largest4SideContour(self, image):
        contours, h = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:min(5,len(contours))]:
            #im = image.copy()
            #cv2.drawContours(im, cnt, -1, (0,255,0), 5)
            #self.displayer.display_image(im,'contour')
            if len(self.approx(cnt)) == 4:
                return cnt
        return None

    def approx(self, cnt):
        peri = cv2.arcLength(cnt, True)
        app = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        return app

    def get_rectangle_corners(self, cnt):
        pts = cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def warp_perspective(self, rect, grid):
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(grid, M, (maxWidth, maxHeight))
        return cv2.resize(warp, (306,306))