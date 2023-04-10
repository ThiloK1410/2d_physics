import numpy
import numpy as np
import pygame
from ball import *
from spatial_hashing import *
import math



class App:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.time_per_frame = 1000 / self.fps
        self.colors = {"RED": (255, 0, 0), "YELLOW": (255, 255, 0)}
        self._running = True
        self._display_surf = None
        self.spatial_hashing = None
        self.size = 1200, 800
        self.collision_cache = set([])

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True

        self.spatial_hashing = Spatial_Hashing(self._display_surf, Ball.balls, Ball.max_radius)

        # create scenario
        Ball(self._display_surf, 800, 400, 10)
        #Ball(self._display_surf, 400, 400, 10, color=(0, 0, 255), weight=20)

        self.on_execute()

    def handle_collision(self):
        # fetch all possible collisions and prevent inverse doubles
        possible_collisions = []
        for actor1 in Ball.balls:
            for actor2 in self.spatial_hashing.get_proximity(actor1):
                if not [actor2, actor1] in possible_collisions:
                    possible_collisions.append([actor1, actor2])

        collisions = set()

        # let all with actual collisions collide
        for pairs in possible_collisions:
            collision_distance = pairs[0].get_rad() + pairs[1].get_rad()
            distance = np.linalg.norm(pairs[0].get_pos() - pairs[1].get_pos())
            if distance < collision_distance:
                collisions.update(pairs)            # maintain a collision cache, to prevent repeated collisions
                if pairs[0] not in self.collision_cache:
                    pairs[0].bounce_on_ball(pairs[1])
                else:
                    #if overlap happens bruteforce them apart
                    pass
        self.collision_cache.clear()
        self.collision_cache = collisions


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                Ball(self._display_surf, dx=10)
            if event.key == pygame.K_i:
                self.spatial_hashing.show_all()
            if event.key == pygame.K_o:
                print(self.spatial_hashing.get_proximity(Ball.balls[0]))

    # loop which will be executed at fixed rate (for physics and such)
    def on_loop(self):
        self.spatial_hashing.refresh()
        self.handle_collision()
        for x in Ball.balls:
            x.on_update()

    # loop which will only be called when enough cpu time is available
    def on_render(self):
        self._display_surf.fill(self.colors.get("YELLOW"))

        for x in Ball.balls:
            x.draw()

        pygame.display.update()

    @staticmethod
    def on_cleanup():
        pygame.quit()

    def on_execute(self):
        previous = pygame.time.get_ticks()
        lag = 0.0

        while self._running:
            current = pygame.time.get_ticks()
            elapsed = current - previous
            lag += elapsed
            previous = current

            for event in pygame.event.get():
                self.on_event(event)

            while lag > self.time_per_frame:
                self.on_loop()
                lag -= self.time_per_frame
            self.on_render()
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_init()
