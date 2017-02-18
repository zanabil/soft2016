import cv2


class DisplayImage(object):

    def display_image(self, img, winName='Image'):
        screen_res = 1280.0, 520.0
        scale_width = screen_res[0] / img.shape[1]
        scale_height = screen_res[1] / img.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(img.shape[1] * scale)
        window_height = int(img.shape[0] * scale)

        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, window_width, window_height)
        cv2.imshow(winName, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
