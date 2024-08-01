import typing

import cv2
import mediapipe
import pygame

from config import *
from views.point import show_point

mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils


class MainWindow:
    screen: pygame.Surface
    camera: cv2.VideoCapture
    result: typing.NamedTuple

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)

        self.point = 0
        self.screen = pygame.display.set_mode(SCREEN_DIMENSIONS, pygame.RESIZABLE)
        self.camera = cv2.VideoCapture(0)
        self.hand_rect = pygame.Rect(0, 0, 50, 50)
        self.hands = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=2,
                                    min_detection_confidence=0.7,
                                    model_complexity=1)

    def update(self):
        ret, frame = self.camera.read()

        self.screen.fill([0, 0, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.screen.get_size())
        show_point(frame, self.point)

        if DEBUG:
            self.draw_hand_connections(frame)
        self.check_hand_close(frame)

        frame_surface = pygame.surfarray.make_surface(frame)
        self.screen.blit(pygame.transform.rotate(frame_surface, -90), (0, 0))
        pygame.display.flip()

    def draw_hand_connections(self, frame):
        self.result = self.hands.process(frame)

        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    def check_hand_close(self, frame):
        multi_hand_landmarks = self.result.multi_hand_landmarks

        window_size = self.screen.get_size()
        window_width, window_height = window_size

        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                x_tip, y_tip = int(middle_tip.x * window_width), int(middle_tip.y * window_height)
                x_mcp, y_mcp = int(middle_mcp.x * window_width), int(middle_mcp.y * window_height)
