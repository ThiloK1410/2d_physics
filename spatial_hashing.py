import numpy
import pygame


class Spatial_Hashing:
    def __init__(self, display, subject_list, max_hitbox_radius):
        self.display = display
        self.window_size = (display.get_width(), display.get_height())
        self.subjects = subject_list

        self.spacing = 2 * max_hitbox_radius
        self.hash_table_shape = (int(self.window_size[1] / self.spacing), int(self.window_size[0] / self.spacing))

        # creating a 2d dictionary with hash_table_shape
        self.hash_table = {i: {j: [] for j in range(self.hash_table_shape[1])} for i in range(self.hash_table_shape[0])}

    def refresh(self):
        # first clear all buckets
        for i in self.hash_table:
            for j in self.hash_table[i]:
                self.hash_table[i][j].clear()
        # add all subjects to hash_table
        for x in self.subjects:
            self.hash(x)

    def get_proximity(self, item):
        list_of_subjects = []
        x, y = self.get_hash(item)

        # iterating over all adjacent cells
        for i in range(x-1, x+2):
            if i < 0 or i > self.hash_table_shape[1]-1:
                continue
            for j in range(y-1, y+2):
                if j < 0 or j > self.hash_table_shape[0]-1:
                    continue
                list_of_subjects += self.hash_table[j][i]
                if item in list_of_subjects: list_of_subjects.remove(item)

        return list_of_subjects

    def delete_subject(self):
        pass

    def get_hash(self, item):
        pos = item.get_pos()
        hash_x, hash_y = int(pos[0] / self.spacing), int(pos[1] / self.spacing)
        return hash_x, hash_y

    def hash(self, item):
        pos = item.get_pos()

        # hash the position so that out of bounds items get assigned to the nearest cell
        hash_x = min(self.hash_table_shape[1]-1, max(0, int(pos[0] / self.spacing)))
        hash_y = min(self.hash_table_shape[0]-1, max(0, int(pos[1] / self.spacing)))

        self.hash_table.get(hash_y).get(hash_x).append(item)

    def show_all(self):
        print("These are the properties of the spatial_hashing:")
        print(f"Spacing: {self.spacing}   Table shape: {self.hash_table_shape}")
        for i, d in enumerate(self.hash_table):
            print(f"Row {i}: {self.hash_table[i]}")
        print(f"There are/is {len(self.subjects)} containing object(s), which are/is: {self.subjects}")

    def draw_grid(self):
        for i in range(self.hash_table_shape[1]+1):
            pygame.draw.line(self.display, (159, 160, 164), (i*self.spacing, 0), (i*self.spacing, self.window_size[1]))
        for j in range(self.hash_table_shape[0]+1):
            pygame.draw.line(self.display, (159, 160, 164), (0, j*self.spacing), (self.window_size[0], j*self.spacing))
