import cv2
import numpy as np


class Fruit:
    def __init__(self, parent_frame):
        self.image = cv2.imread('assets/apple.png')
        self.image.resize((50, 50))
        self.height, self.width = self.image.shape[:2]
        self.parent_frame = parent_frame

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.set_pos(0, 0)

    def set_pos(self, x, y):
        lower_white = np.array([200, 200, 200])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(self.image, lower_white, upper_white)

        # Invert the mask to get the non-white areas
        mask_inv = cv2.bitwise_not(mask)

        # Extract the regions of interest (ROI) from the frame
        roi = self.parent_frame[y:y + self.height, x:x + self.width]

        # Black-out the white area of the self.image in ROI
        frame_bg = cv2.bitwise_and(roi, roi, mask=mask)

        # Extract the self.image region without the white background
        overlay_fg = cv2.bitwise_and(self.image, self.image, mask=mask_inv)

        # Add the overlay_fg to frame_bg
        result = cv2.add(frame_bg, overlay_fg)

        # Place the result back into the original frame
        self.parent_frame[y:y + self.height, x:x + self.width] = result
