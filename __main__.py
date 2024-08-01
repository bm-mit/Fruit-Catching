import pygame

from views.main_window import MainWindow

if __name__ == "__main__":
    window = MainWindow()

    running = True
    while running:
        window.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
