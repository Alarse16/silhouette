import cv2
import numpy as np
from positions import Positions
from silhuet import Silhuet

_positions = Positions("pos")   # initialising a Positions class
_silhuet = Silhuet("sil")       # initialising a Silhuet class
points = 0                      # the score of the user

# appllyes some amout of open and close operation on an image
def open_close(img):
    kernel = np.ones((5, 5), np.uint8)  #a 5x5 matrix consisting of 1's
    out = img.copy()
    out = cv2.blur(out, (10, 10))
    out = cv2.medianBlur(out, 21)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)  # opening
    for i in range(100):  # dose this operation 100 times
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)  # closing

    return out


def calibrate():
    bf = cv2.createBackgroundSubtractorMOG2()  # an open cv function that saves a few frames as the background and
    # displays the differents between the back and for ground
    return bf


def PointHandeler(in_image):

    point_text = 'Points: ' + str(points)
    font = cv2.FONT_HERSHEY_SIMPLEX         # font
    org = (50, 50)                          # org
    font_scale = 1                          # fontScale
    color = (255, 0, 0)                     # Blue color in BGR
    thickness = 2                           # Line thickness of 2 px
    out = cv2.putText(in_image, point_text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    return out


def positions(i):
    switcher = {                # positions of detection points
        0: (135, 205, 440, 460),
        1: (50, 435, 105, 435),
        2: (250, 170, 90, 415),
    }
    return switcher.get(i)

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)  # the output from the camera
    calibrated = False
    p = 0  # value for determening what position is to be displaid

    while True:
        _, source_image = capture.read()  # asigns the vedio capture to an image
        source_image = cv2.flip(source_image, 1)  # flips the image so it mirros the user
        gray_img = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)  # converts the image to grayscale

        if not calibrated:
            if cv2.waitKey(1) & 0xFF == ord('c'):  # if c is pressed
                background_function = calibrate()
                calibrated = True
        if calibrated:
            background_img = background_function.apply(source_image, None, 0.001)
            # cv2.imshow('frame', background_img)

            opcl_img = open_close(background_img)  # calls the open close function on the image
            (thresh, out_img) = cv2.threshold(opcl_img, 200, 255, cv2.THRESH_BINARY)  # appllyes a threshold on the
            # image to remove noice

            source_image = _silhuet.drawSilhuets(p, source_image)

            # asigns positions for pos1 and pos2 based on the function positions()
            pos1 = (positions(p)[0], positions(p)[1])
            pos2 = (positions(p)[2], positions(p)[3])

            # calls the draw_positions_points() functions from the Positions class
            if _positions.draw_positions_points(pos2, pos1, out_img, source_image):
                points = points + 1
                if p < 2:
                    p = p + 1
                else:
                    p = 0

            cv2.imshow('opCl image', out_img)

        PointHandeler(source_image)
        cv2.imshow('source_image', source_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
