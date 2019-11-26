import cv2
import numpy as np

class Silhuet:
    def __init__(self, name):
        self.name = name


    def drawSilhuets(self, image_numper, source_image):
        img1 = source_image
        img2 = cv2.imread('dab_pose.png')

        img3 = img1.copy()
        # replace values at coordinates (100, 100) to (399, 399) of img3 with region of img2
        img3[100:400, 100:400, :] = img2[100:400, 100:400, :]
        cv2.imshow('Result1', img3)
