import cv2
import numpy as np


def open_close(img):
    kernel = np.ones((5, 5), np.uint8)

    out = img.copy()
    out = cv2.blur(out, (10, 10))
    out = cv2.medianBlur(out, 21)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)  # opening
    for i in range(100):
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)  # closing

    return out


def calibrate():
    bf = cv2.createBackgroundSubtractorMOG2()
    return bf


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    # var = capture.set(cv2.CAP_PROP_EXPOSURE, 0)
    calibrated = False

    while True:
        _, source_image = capture.read()
        source_image = cv2.flip(source_image, 1)
        gray_img = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

        if not calibrated:
            if cv2.waitKey(1) & 0xFF == ord('c'):
                background_function = calibrate()
                calibrated = True
        if calibrated:
            background_img = background_function.apply(source_image, None, 0.001)
            cv2.imshow('frame', background_img)

            opcl_img = open_close(background_img)
            (thresh, out_img) = cv2.threshold(opcl_img, 200, 255, cv2.THRESH_BINARY)
            cv2.imshow('opCl image', out_img)

        cv2.imshow('source_image', source_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
