import cv2
import numpy as np


def show_text(frame, point, pos):
    text = str(point)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_thickness = 5
    position = (0, 0)
    text_pos = pos
    text_color = (255, 0, 0)

    text_img = np.zeros_like(frame)

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size

    text_pos = (text_pos[0], text_pos[1] + text_height)

    cv2.putText(text_img, text, text_pos, font, font_scale, text_color, font_thickness)

    flipped_text_img = cv2.flip(text_img, 1)

    gray_text = cv2.cvtColor(flipped_text_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_text, 1, 255, cv2.THRESH_BINARY)

    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    flipped_text_img = cv2.resize(flipped_text_img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    x, y = position
    h, w = flipped_text_img.shape[:2]
    
    roi = frame[y:y + h, x:x + w]

    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))

    flipped_text_img = cv2.bitwise_and(flipped_text_img, flipped_text_img, mask=mask)
    result = cv2.add(roi_bg, flipped_text_img)

    frame[y:y + h, x:x + w] = result

    return frame
