import numpy as np

class Maze:
    def __init__(self, size):
        self.size = size
        self.maze = self._generate_ring_maze(size)

    def _generate_ring_maze(self, size):
        # Initialize a grid with walls
        grid = np.ones((size, size))

        # Logic to create a ring-shaped maze
        # ...

        return grid

    def display(self):
        for row in self.maze:
            print(" ".join(str(cell) for cell in row))

# Example usage
maze = Maze(10)
maze.display()
