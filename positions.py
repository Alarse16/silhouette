import cv2
import numpy as np


class Positions:
    def __init__(self, name):
        self.name = name

    def draw_positions_points(self, pos1, pos2, in_img):
        cv2.circle(in_img, pos1, 10, (255, 0, 0), 2)
        cv2.circle(in_img, pos2, 10, (255, 0, 0), 2)
        if in_img[pos1] != 0 and in_img[pos2] != 0:
            return True
        return False

    # def if_point_are_hit(self, pos1, pos2, in_img):
