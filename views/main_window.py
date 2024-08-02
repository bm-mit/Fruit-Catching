import typing
from random import randint

import cv2
import mediapipe
import pygame

from config import *
from utils import get_time, distance
from views.fruit import Fruit
from views.text import show_text

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
        self.time = PLAYTIME
        self.last_tick = 0
        self.last_move = 0
        self.fruits = [0] * MAX_FRUIT
        self.screen = pygame.display.set_mode(SCREEN_DIMENSIONS, pygame.RESIZABLE)
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FPS, 120)
        self.hand_rect = pygame.Rect(0, 0, 50, 50)
        self.hands = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=2,
                                    min_detection_confidence=0.7,
                                    model_complexity=1)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.count = 0

    def update(self):
        ret, frame = self.camera.read()

        self.screen.fill([0, 0, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.screen.get_size())

        if get_time() - self.last_tick >= 1 and self.time > 0:
            self.last_tick = get_time()
            self.time -= 1

        for i in range(len(self.fruits)):
            if self.fruits[i] == 0:
                if randint(0, SPAWN_RATE) == 1:
                    self.count += int(self.time > 0)
                    self.fruits[i] = Fruit(frame, randint(100, frame.shape[1] - 100), 50)
            else:
                self.fruits[i] = Fruit(frame, self.fruits[i].x, self.fruits[i].y + SPEED)
                if self.fruits[i].y > frame.shape[0] - 100:
                    self.fruits[i] = 0
            if self.fruits[i]:
                self.fruits[i] = Fruit(frame, self.fruits[i].x, self.fruits[i].y)

        show_text(frame, str(self.point), (frame.shape[1] - 100 * len(str(self.point)), frame.shape[0] - 125))
        show_text(frame, str(self.time), (30, frame.shape[0] - 125))

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
        if self.time < 1:
            show_text(frame, f"{(self.point/self.count) * 100:.2f}%", (SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2 - 50))

        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                x_tip, y_tip = int(middle_tip.x * window_width), int(middle_tip.y * window_height)
                x_mcp, y_mcp = int(middle_mcp.x * window_width), int(middle_mcp.y * window_height)

                if self.time > 0:
                    for i in range(MAX_FRUIT):
                        if self.fruits[i] and distance(x_mcp, y_mcp, self.fruits[i].x,
                                                       self.fruits[i].y) <= CATCH_DISTANCE:
                            self.fruits[i] = 0
                            self.point += 1
