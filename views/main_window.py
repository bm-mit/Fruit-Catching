import cv2
import pygame

from config import *


class MainWindow:
    screen: pygame.Surface
    camera: cv2.VideoCapture

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)

        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        self.camera = cv2.VideoCapture(0)

    def update(self):
        ret, frame = self.camera.read()

        self.screen.fill([0, 0, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, SCREEN_DIMENSIONS)
        frame_surface = pygame.surfarray.make_surface(frame)

        self.screen.blit(pygame.transform.rotate(frame_surface, -90), (0, 0))

        pygame.display.flip()
