import cv2
import numpy as np


# the constructor
class Positions:
    def __init__(self, name):
        self.name = name

    # this function draws circles on the positions and deteckts if they are filled
    def draw_positions_points(self, pos1, pos2, in_img, source_image):
        # draws the circle, (pos1[1], pos1[0]) looks like this becouse python takes y value first
        # cv2.circle(source_image, (pos1[1], pos1[0]), 10, (255, 0, 0), 2)
        # cv2.circle(source_image, (pos2[1], pos2[0]), 10, (255, 0, 0), 2)

        # if both points are filled
        if in_img[pos1] == 255 and in_img[pos2] == 255:
            return True
        return False
