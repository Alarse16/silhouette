import cv2
import numpy as np

class Silhuet:
    def __init__(self, name):
        self.name = name


    def drawSilhuets(self, image_numper, source_image):
        img1 = source_image
        image_name = ('Tai_Chi_Pose_' + str(image_numper+1) + '.jpg')
        img2 = cv2.imread(image_name)

        alpha = 0.7
        img3 = np.uint8(img1 * alpha + img2 * (1 - alpha))
        return img3
