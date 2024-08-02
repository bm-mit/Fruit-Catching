import cv2
import numpy as np


class Fruit:
    def __init__(self, parent_frame, x, y):
        self.image = cv2.imread('assets/apple.png')
        self.image = cv2.resize(self.image, (100, 100))

        self.height, self.width = self.image.shape[:2]
        self.parent_frame = parent_frame

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        lower_white = np.array([200, 200, 200])
        upper_white = np.array([255, 255, 255])
        self.mask = cv2.inRange(self.image, lower_white, upper_white)

        self.mask_inv = cv2.bitwise_not(self.mask)

        self.x = x
        self.y = y

        self.set_pos(x, y)

    def set_pos(self, x, y):

        x -= self.width // 2
        y -= self.height // 2

        roi = self.parent_frame[y:y + self.height, x:x + self.width]

        try:
            frame_bg = cv2.bitwise_and(roi, roi, mask=self.mask)

            overlay_fg = cv2.bitwise_and(self.image, self.image, mask=self.mask_inv)

            result = cv2.add(frame_bg, overlay_fg)

            self.parent_frame[y:y + self.height, x:x + self.width] = result
        except:
            return

    def move_down(self):
        self.set_pos(self.x, self.y + 2)
