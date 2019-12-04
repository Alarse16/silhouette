import time

import cv2
import numpy as np
import winsound

# How fast the background subtraction algorithm learns the background to subtract
# A number between 0 and 1
LEARNING_RATE = 0.001

score = 0  # the user score
delay = 0  # number of seconds to freeze the screen after the user hits the correct pose


def draw_silhouette(pose_index, camera_frame):
    image_name = ('Tai_Chi_Pose_' + str(pose_index) + '.jpg')
    pose_image = cv2.imread(image_name)

    alpha = 0.7
    return np.uint8(camera_frame * alpha + pose_image * (1 - alpha))


def draw_silhouette_in_green(pose_index, camera_frame):
    image_name = ('Tai_Chi_Pose_' + str(pose_index) + '_G' + '.jpg')
    pose_image = cv2.imread(image_name)

    alpha = 0.7
    return np.uint8(camera_frame * alpha + pose_image * (1 - alpha))


# Returns True if both of the hit points are hit
def are_hit_points_hit(hit_point1, hit_point2, segmented_img):
    # If both points are filled
    if segmented_img[hit_point1] == 255 and segmented_img[hit_point2] == 255:
        return True

    return False


# Applies one open operation and a hundred close operations
def open_close(source_image):
    kernel = np.ones((5, 5), np.uint8)  # A 5x5 matrix consisting of 1's
    out = source_image.copy()
    out = cv2.blur(out, (10, 10))  # Apply Gaussian blur
    out = cv2.medianBlur(out, 21)  # Apply median blur

    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)  # Apply open hole operation

    for i in range(100):
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)  # Apply close hole operation

    return out


# Returns the background subtraction function.
# The result is an image where pixels are marked as background (black) or foreground (white)
# based on the historical data per each pixel and some clever Gaussian mixturing
def background_subtraction():
    return cv2.createBackgroundSubtractorMOG2()


# Returns an image with the current score displayed on the image passed in
def draw_score(source_image):
    score_text = 'Score: ' + str(score)  # String to display
    font = cv2.FONT_HERSHEY_SIMPLEX  # Normal size sans-serif font
    position = (50, 50)  # Position of the upper left corner of the text
    font_scale = 1
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2  # Line thickness of 2 px

    # Draw the "score" text on the source_image
    return cv2.putText(source_image, score_text, position, font, font_scale, color, thickness, cv2.LINE_AA)


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


# this function removes the black border that can occurs on some images
def remove_black_borders(source_image):
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # make a threshold for black
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find edges of image
    cnt = contours[0]  # save the edges
    x, y, w, h = cv2.boundingRect(cnt)  # make a boundingbox to contain image inside
    source_image = source_image[y:y + h, x:x + w]  # save image within bounding rect
    return source_image


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)  # Camera footage
    segmentation_function = None  # The process which segments the image into foreground and background
    pose_index = 0  # 0, 1 or 2

    while True:
        _, source_image = capture.read()  # Current frame from the camera footage
        source_image = cv2.resize(source_image, (640, 480))   # resize the image to a correct size
        source_image = cv2.flip(source_image, 1)  # Flips the frame vertically, so it works like looking at mirror
        cornerValue = source_image[0, 0]  # the RGB value of the top left corner of the image
        if cornerValue[0] == 0 and cornerValue[1] == 0 and cornerValue[3] == 0:  # if the corner pixel is black (the image has a black border)
            source_image = remove_black_borders(source_image)
        gray_img = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)  # Converts the image to grayscale

        if segmentation_function is None:
            # Wait until user presses "c" on the keyboard (blocking)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                segmentation_function = background_subtraction()

        if segmentation_function is not None:
            segmented_image = segmentation_function.apply(source_image, None, LEARNING_RATE)

            open_closed_image = open_close(segmented_image)
            _, out_img = cv2.threshold(open_closed_image, 200, 255, cv2.THRESH_BINARY)  # Removes noise

            time.sleep(delay)
            source_image = draw_silhouette(pose_index, source_image)
            delay = 0  # Reset the delay

            # The positions of the two hit points for the current pose
            hitpoint1 = (positions(pose_index)[0], positions(pose_index)[1])
            hitpoint2 = (positions(pose_index)[2], positions(pose_index)[3])

            if are_hit_points_hit(hitpoint2, hitpoint1, out_img):
                score = score + 1
                source_image = draw_silhouette_in_green(pose_index, source_image)
                delay = 1  # Frame after this one should be freezed for one second
                winsound.PlaySound("Confirm_Sound_Piano.wav", winsound.SND_ASYNC)

                # Set the next pose for the next frame
                if pose_index < 2:
                    pose_index = pose_index + 1
                else:
                    pose_index = 0

            cv2.imshow('Computer vision', out_img)

        draw_score(source_image)
        cv2.imshow('Camera feed', source_image)

        # quit the program if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
