import numpy as np
import pygame
#from neural_network import NeuralNetwork


class Ball:
    balls = []
    max_radius = 30

    def __init__(self, surface, x=None, y=None, dx=0, dy=0, weight=None, color=None):
        self._surface = surface
        if x is None or y is None:
            x = np.random.rand() * self._surface.get_width()
            y = np.random.rand() * self._surface.get_height()
        self.balls.append(self)
        self._weight = weight
        self._color = color
        self._position = np.array([x, y]).astype("float64")
        self._movement = np.array([dx, dy]).astype("float64")
        self.number_of_refreshes = 1

        self._acceleration = np.array([0, 0]).astype("float64")
        random_vector = np.random.rand(2)
        self._direction = random_vector / np.linalg.norm(random_vector)

        self.perception = [self._position[0], self._position[1], self._movement[0], self._movement[1]]

        # predefined values

        self._radius = 30    # check max_radius
        if self._weight is None:
            #self._weight = np.pi * self._radius ** 2
            self._weight = 1
        if self._color is None:
            self._color = (255, 0, 0)
        self.new_movement = np.array([0, 0]).astype("float64")
        self.new_movement_x = 1

    def get_weight(self):
        return self._weight

    def get_pos(self):
        return self._position

    def get_mov(self):
        return self._movement

    def get_mov_fraction(self, fraction_factor):
        return self._movement/fraction_factor

    def get_mov_abs(self):
        return np.linalg.norm(self._movement)

    def set_mov(self, movement):
        self._movement = movement

    def get_dir(self):
        return self._direction

    def get_rad(self):
        return self._radius

    def get_area(self):
        return np.pi*self._radius**2

    def get_nn_output(self):
        self.brain.calculate(self.perception)
        return self.brain.outputs

    def active_move(self, factor=1):
        direction = self.brain.calculate(self.perception)
        if not len(direction) == len(self._acceleration):
            raise ValueError("Wrong input for active_move")
        for i, x in enumerate(direction):
            self._acceleration[i] = direction[i] * factor

    def get_collision_push_force(self, distance, radius=0):
        if np.linalg.norm(distance) < self._radius + radius:
            if not np.linalg.norm(distance) == 0:
                return 50 / np.linalg.norm(distance)
        return 0

    def draw(self):
        pygame.draw.circle(self._surface, self._color, self._position, self._radius)

    def on_update(self):
        self.number_of_refreshes = 1 + int(self.get_mov_abs()) * 1

        for x in range(self.number_of_refreshes):
            movement_fraction = self.get_mov_fraction(self.number_of_refreshes)

            # continuous collision for screen borders
            # if self is out of bounds in next check
            if self.is_out_of_bounds(self._position + movement_fraction):
                x = self.get_intersect()
                self._position += movement_fraction
                self.bounce_on_axis(self.is_out_of_bounds(self._position + movement_fraction, give_axis=True))
                self._position += self.new_movement * (1-x)
                self.new_movement_x = 1
                self._movement = self.new_movement

            # when self will collide next frame:
            elif self.new_movement_x < 1:
                self._position += movement_fraction * self.new_movement_x + self.new_movement * (1 - self.new_movement_x)
                self._movement = self.new_movement
                self.new_movement_x = 1
            else:
                self._position += movement_fraction

            #self.active_move()
            self._movement += self._acceleration * 1/self.number_of_refreshes
        self._acceleration = np.array([0, 0]).astype("float64")

    # returns an 0<=x<=1 which shows where between frames collisions happen,
    # when no ball is given, box collisions will be checked
    def get_intersect(self, ball=None):
        temp = 1
        # checking for box collisions
        if ball is None:
            if not self.is_out_of_bounds(self._position + self.get_mov_fraction(self.number_of_refreshes)):
                return 1

            def rec_intersect(x, rec_counter=0):
                depth = rec_counter + 1
                if rec_counter > 10:
                    return x
                elif self.is_out_of_bounds(self._position + self.get_mov_fraction(self.number_of_refreshes) * x):
                    x -= 1 / (2**depth)
                    return rec_intersect(x, depth)
                else:
                    x += 1 / (2**depth)
                    return rec_intersect(x, depth)

            return rec_intersect(temp)
        # checking for ball collisions
        else:
            if not self.will_collide(ball):
                return 1

            def rec_collision(x, rec_counter=0):
                depth = rec_counter + 1
                if rec_counter > 10:
                    return x
                elif self.will_collide(ball, x):
                    x -= 1 / (2**depth)
                    return rec_collision(x, depth)
                else:
                    x += 1 / (2**depth)
                    return rec_collision(x, depth)

            return rec_collision(temp)

    def is_out_of_bounds(self, position, give_axis=False):
        out = None
        r = self._radius

        # the return values are: out_of_x = 0, out_of_y = 1, both = 2
        if give_axis:
            if position[0] + r > self._surface.get_width() or position[0] - r < 0:
                out = 0
            if position[1] + r > self._surface.get_height() or position[1] - r < 0:
                out = 1
            if position[1] + r > self._surface.get_height() or position[1] - r < 0 and\
                    position[0] + r > self._surface.get_width() or position[0] - r < 0:
                out = 2

        # true if out of bounds
        else:
            out = False
            if position[0] + r > self._surface.get_width() or position[0] - r < 0:
                out = True
            if position[1] + r > self._surface.get_height() or position[1] - r < 0:
                out = True

        return out

    def will_collide(self, ball, x=1):
        future_distance = self.get_pos() + self.get_mov_fraction(self.number_of_refreshes)*x - \
                          (ball.get_pos() + ball.get_mov_fraction(self.number_of_refreshes)*x)
        future_abs_distance = np.linalg.norm(future_distance)
        collision_distance = self.get_rad() + ball.get_rad()

        if future_abs_distance <= collision_distance:
            return True

    # using default value as global variable
    def bounce_on_axis(self, axis, cache=[False]*2):
        if axis == 0:
            self.new_movement[0] *= -1
            cache[0] = True
        elif axis == 1:
            self.new_movement[1] *= -1
            cache[1] = True
        elif axis == 2:
            self.new_movement *= -1
            cache[0], cache[1] = True, True
        else:
            cache[0], cache[1] = False, False

    def bounce_on_ball(self, ball):
        normal = self.get_pos() - ball.get_pos()
        u_normal = normal / np.linalg.norm(normal)

        u_tangent = np.array([-u_normal[1], u_normal[0]])

        mov = self.get_mov()
        mov2 = ball.get_mov()

        # casting mov on normal and tangent vector (as scalars):
        # will stay the same:
        v_t1 = np.dot(mov, u_tangent)
        v_t2 = np.dot(mov2, u_tangent)
        # needs to be changed:
        v_n1 = np.dot(mov, u_normal)
        v_n2 = np.dot(mov2, u_normal)

        w1 = self.get_weight()
        w2 = ball.get_weight()

        new_v_n1 = (v_n1*(w1 - w2) + 2 * w2 * v_n2) / (w1 + w2)
        new_v_n2 = (v_n2*(w2 - w1) + 2 * w1 * v_n1) / (w1 + w2)

        new_mov1 = new_v_n1 * u_normal + v_t1 * u_tangent
        new_mov2 = new_v_n2 * u_normal + v_t2 * u_tangent

        self.new_movement = new_mov1
        ball.new_movement = new_mov2

        self.new_movement_x = self.get_intersect(ball)
        ball.new_movement_x = ball.get_intersect(self)



