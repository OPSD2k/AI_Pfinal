from pygame import draw, font

import sys

class GridElement:
    """
    GridElement used as a tile in the exercise
    """

    """
    Initialise the GridElement and assign the starting values
    """

    def __init__(self, x, y, size):
        self.position = (x, y)
        self.neighbours = []
        self.size = (size[0], size[1])
        self.parent = None
        self.distance = None
        self.score = None
        self.color = (255, 255, 255)

    """
    Overload the equals operator
    """

    def __eq__(self, other):
        return self.position == other.position
    """
    Overload the less than operator
    """

    def __lt__(self, other):
        return (self.score is not None) and (other.score is None or self.score < other.score)

    """
       Overload the hash operator
    """
    def __hash__(self):
        return hash(self.position)
    """
    Overload the string representation of the object
    """

    def __repr__(self):
        return "[%s, %s]" % (self.position, self.score)

    """
    Remove all neighbours
    """

    def reset_neighbours(self):
        self.neighbours = []

    """
    Sets the state of the GridElement 
    """

    def reset_state(self):
        self.parent = None
        self.score = None
        self.distance = None
        self.color = (255, 255, 255)

    def get_neighbours(self):
        return self.neighbours[:]

    """
     Method to calculate the Manhattan distance from a certain 
     GridElement to another GridElement of the exercise
     """

    def manhattan_distance(self, other):
        x_distance = abs(self.position[0] - other.position[0])
        y_distance = abs(self.position[1] - other.position[1])
        return x_distance + y_distance

    def null_distance(self, other):
        x_distance = abs(self.position[0] - other.position[0])
        y_distance = abs(self.position[1] - other.position[1])
        return max(x_distance ,y_distance)

    def direction(self, other):
        return other.position[0] - self.position[0], other.position[1] - self.position[1]

    def set_score(self, score):
        self.score = score

    def set_distance(self, distance):
        self.distance = distance

    def get_distance(self):
        return self.distance

    def get_score(self):
        return self.score

    def get_position(self):
        return self.position

    """
    Assign the GridElement used to reach this GridElement
    """

    def set_parent(self, parent):
        self.parent = parent
        if parent.distance is not None:
            self.distance = parent.distance+1

    def set_color(self, color):
        self.color = color

    """
    Draw the GridElement
    """

    def draw_grid_element(self, surface):
        draw.rect(surface, self.color,
                  (self.position[0] * self.size[0], self.position[1] * self.size[1], self.size[0], self.size[1]), 0)

        # discard the directions where neighbours are
        compass = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # The four directions
        for neighbour in self.neighbours:
            if self.direction(neighbour) in compass:
                compass.remove(self.direction(neighbour))

        for direction in compass:
            if direction == (0, -1):  # North
                draw.line(surface, (0, 0, 0), (self.position[0] * self.size[0], self.position[1] * self.size[1]),
                          ((self.position[0] + 1) * self.size[0], self.position[1] * self.size[1]), 2)
            if direction == (1, 0):  # East
                draw.line(surface, (0, 0, 0), ((self.position[0] + 1) * self.size[0], self.position[1] * self.size[1]),
                          ((self.position[0] + 1) * self.size[0], (self.position[1] + 1) * self.size[1]), 2)
            if direction == (0, 1):  # South
                draw.line(surface, (0, 0, 0), (self.position[0] * self.size[0], (self.position[1] + 1) * self.size[1]),
                          ((self.position[0] + 1) * self.size[0], (self.position[1] + 1) * self.size[1]), 2)
            if direction == (-1, 0):  # West
                draw.line(surface, (0, 0, 0), (self.position[0] * self.size[0], self.position[1] * self.size[1]),
                          (self.position[0] * self.size[0], (self.position[1] + 1) * self.size[1]), 2)

        # This draw an arrow to from the parent
        if self.parent is not None:

            vector = self.direction(self.parent)

            center = ((self.position[0]+0.5) * self.size[0],(self.position[1]+0.5) * self.size[1])

            if vector[0] !=0:
                left_point = (center[0]+(vector[0]-vector[1])*self.size[0]/5,center[1]+(vector[1]-vector[0])*self.size[0]/5)
                right_point = (center[0] + (vector[0] - vector[1]) * self.size[0] / 5, center[1] + (vector[1] + vector[0]) * self.size[0] / 5)
            else:
                left_point = (center[0] + (vector[0] - vector[1]) * self.size[0] / 5,
                              center[1] + (vector[1] + vector[0]) * self.size[0] / 5)
                right_point = (center[0] + (vector[0] + vector[1]) * self.size[0] / 5,
                               center[1] + (vector[1] + vector[0]) * self.size[0] / 5)
            draw.polygon(surface, (100,100,100),(center,left_point,right_point))
            entry_point= (center[0]+vector[0]*self.size[0]/2,center[1]+vector[1]*self.size[1]/2)
            end_point = (center[0] + vector[0] * self.size[0] / 5, center[1] + vector[1] * self.size[1] / 5)
            draw.line(surface, (100,100,100),end_point,entry_point,int(self.size[0]/20)+1)


    def print_neighbours(self):

        directions = []
        for neighbor in self.neighbours:
            if self.direction(neighbor) == (0, -1):  # North
                directions.append("North")
            elif self.direction(neighbor) == (1, 0):  # East
                directions.append("East")
            elif self.direction(neighbor) == (0, 1):  # South
                directions.append("South")
            elif self.direction(neighbor) == (-1, 0):  # West
                directions.append("West")
            else:
                directions.append(self.direction(neighbor))

        print(directions)
        return None

    def print_walls(self):
        # discard the directions where neighbours are
        compass = {(0, -1): "North",
                   (1, 0): "East",
                   (0, 1): "South",
                   (-1, 0): "West"}  # The four directions
        for neighbor in self.neighbours:
            compass.pop(self.direction(neighbor))

        print(list(compass.values()))
        return None
