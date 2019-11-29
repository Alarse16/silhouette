import time

import cv2
import numpy as np
import winsound

points = 0  # the score of the user
delay = 0  # number of seconds to freeze the screen after the user hits the correct pose


def draw_silhouette(pose_index, camera_frame):
    image_name = ('Tai_Chi_Pose_' + str(pose_index + 1) + '.jpg')
    pose_image = cv2.imread(image_name)

    alpha = 0.7
    return np.uint8(camera_frame * alpha + pose_image * (1 - alpha))


def draw_silhouette_in_green(pose_index, camera_frame):
    image_name = ('Tai_Chi_Pose_' + str(pose_index + 1) + '_G' + '.jpg')
    pose_image = cv2.imread(image_name)

    alpha = 0.7
    return np.uint8(camera_frame * alpha + pose_image * (1 - alpha))


# Returns True if both of the hit points are hit
def are_hitpoints_hit(hit_point1, hit_point_2, segmented_img):
    # if both points are filled
    if segmented_img[hit_point1] == 255 and segmented_img[hit_point_2] == 255:
        return True

    return False


# Applies one open operation and a hundred close operations
def open_close(source_image):
    kernel = np.ones((5, 5), np.uint8)  # A 5x5 matrix consisting of 1's
    out = source_image.copy()
    out = cv2.blur(out, (10, 10))  # Apply Gaussian blur
    out = cv2.medianBlur(out, 21)  # Apply median blur

    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)  # opening

    for i in range(100):
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)  # closing

    return out


# Returns the background subtraction function.
# The result is an image where pixels are marked as background (black) or foreground (white)
# based on the historical data per each pixel and some clever Gaussian mixturing
def background_subtraction():
    return cv2.createBackgroundSubtractorMOG2()


# Returns an image with the current number of points displayed on the image passed in
def draw_points(source_image):
    point_text = 'Points: ' + str(points)  # String to display
    font = cv2.FONT_HERSHEY_SIMPLEX  # Normal size sans-serif font
    position = (50, 50)  # Position of the upper left corner of the text
    font_scale = 1
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2  # Line thickness of 2 px

    # Draw the "points" text on the source_image
    return cv2.putText(source_image, point_text, position, font, font_scale, color, thickness, cv2.LINE_AA)


# Returns (x1, y1, x2, y2), which are xy coordinates for two points.
# These coordinates represent positions of the two points one has to hit in webcam frames
# for the program to consider the movement performed correctly.
# There are three sets of these 4 coordinates (pose_index values 0, 1, 2), one for each pose.
def positions(pose_index):
    detection_points = {
        0: (135, 205, 440, 460),
        1: (50, 435, 105, 435),
        2: (250, 170, 90, 415),
    }

    return detection_points.get(pose_index)

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)   # Camera footage
    calibrated = False              # Whether the detection has been started
    pose_index = 0                  # 0, 1 or 2

    while True:
        _, source_image = capture.read()  # Current frame from the camera footage
        source_image = cv2.flip(source_image, 1)  # Flips the frame vertically, so it works like looking at mirror
        gray_img = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)  # Converts the image to grayscale

        if not calibrated:
            if cv2.waitKey(1) & 0xFF == ord('c'):
                calibrated = True
        if calibrated:
            segmented_image = background_subtraction().apply(source_image, None, 0.001)

            open_closed_image = open_close(segmented_image)
            (_, out_img) = cv2.threshold(open_closed_image, 200, 255, cv2.THRESH_BINARY)  # Removes noise

            time.sleep(delay)
            source_image = draw_silhouette(pose_index, source_image)
            delay = 0  # Reset the delay

            # The positions of the two hit points for the current pose
            hitpoint1 = (positions(pose_index)[0], positions(pose_index)[1])
            hitpoint2 = (positions(pose_index)[2], positions(pose_index)[3])

            if are_hitpoints_hit(hitpoint2, hitpoint1, out_img):
                points = points + 1
                source_image = draw_silhouette_in_green(pose_index, source_image)
                delay = 1  # Frame after this one should be freezed for one second
                winsound.PlaySound("Confirm_Sound_Piano.wav", winsound.SND_ASYNC)

                # Set the next pose for the next frame
                if pose_index < 2:
                    pose_index = pose_index + 1
                else:
                    pose_index = 0

            cv2.imshow('Computer vision', out_img)

        draw_points(source_image)
        cv2.imshow('Camera feed', source_image)

        # quit the program if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break