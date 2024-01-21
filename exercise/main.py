import pygame
import random
import heapq
import math
import numpy as np
import cv2 as cv

# Constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WINDOW_SIZE = 1000


class Ball:
    def __init__(self, x, y, radius, colour, screen, maze):
        self.x = x
        self.y = y
        self.radius = radius
        self.colour = colour
        self.vel_y = 0  # vertical velocity
        self.acc_g = 0.5  # gravity
        self.dt = 0.1  # simulation speed
        self.screen = screen
        self.fall = True
        self.falls_number = 0
        self.maze = maze

    def update(self):
        #print(self.fall)
        self.move()
        self.draw()

    def move(self):
        self.collision()
        #self.check_for_fall()
        if self.fall:
            self.vel_y += self.acc_g * self.dt
            self.y += self.vel_y * self.dt


    def collision(self):
        # we know where the holes in the maze are.
        # if ball is in a cell with a hole, fall down until specific distance from centre
        distance_from_centre = self.distance_from_centre(self.y)

        # Calculate the current ring and sector
        current_ring = int((distance_from_centre / (WINDOW_SIZE // 2)) * self.maze.rings)
        current_ring = min(max(current_ring, 0), self.maze.rings - 1)
        angle = self.maze.rotation_angle % 360
        current_sector = int(angle / (360 / self.maze.sectors))
        current_sector = min(max(current_sector, 0), self.maze.sectors - 1)
        #print(current_ring, current_sector)
        #print(self.maze.is_wall(current_ring, 7-current_sector)) #wtf???? Why 7?
        #print(self.falls_number)

        if not self.maze.is_wall(current_ring, 7 - current_sector):
            self.fall = True

        if distance_from_centre >= (self.falls_number + 1) * (WINDOW_SIZE // 2) // 10 - 2 * self.radius:
            # inner_radius = ring * (WINDOW_SIZE // 2) // self.maze.rings
            self.fall = False
            self.falls_number += 1
            # Reset velocity when not falling
            self.vel_y = 0

    def distance_from_centre(self, y):
        return y - WINDOW_SIZE // 2

    def draw(self):
        pygame.draw.circle(self.screen, self.colour, (int(self.x), int(self.y)), self.radius)


class RingMaze:
    def __init__(self, rings, sectors, screen, path=None):
        self.rings = rings
        self.sectors = sectors
        self.maze = [[{'wall': True, 'radial_walls': [True, True]} for _ in range(sectors)] for _ in range(rings)]
        self.generate_maze()
        self.screen = screen
        self.path = path
        self.rotation_angle = 0  # Initial rotation angle

    def draw_maze(self):
        self.screen.fill(WHITE)
        center_x, center_y = WINDOW_SIZE // 2, WINDOW_SIZE // 2
        wall_thickness = max(1, WINDOW_SIZE // (8 * self.rings))

        for ring in range(self.rings):
            inner_radius = ring * (WINDOW_SIZE // 2) // self.rings
            outer_radius = (ring + 1) * (WINDOW_SIZE // 2) // self.rings

            for sector in range(self.sectors):
                sector_info = self.get_sector_info(ring, sector)
                # add rotation
                start_angle = math.radians(sector * (360 / self.sectors) + self.rotation_angle)
                end_angle = math.radians((sector + 1) * (360 / self.sectors) + self.rotation_angle)

                # Draw angular wall if sector is a wall
                if sector_info['wall']:
                    bounding_rect = [center_x - outer_radius, center_y - outer_radius, 2 * outer_radius,
                                     2 * outer_radius]
                    pygame.draw.arc(self.screen, BLACK, bounding_rect, start_angle, end_angle, wall_thickness)

                # Draw radial walls based on sector_info
                if sector_info['radial_walls'][0]:
                    self.draw_radial_wall(center_x, center_y, inner_radius, outer_radius, start_angle, wall_thickness)
                if sector_info['radial_walls'][1]:
                    self.draw_radial_wall(center_x, center_y, inner_radius, outer_radius, end_angle, wall_thickness)

    def draw_radial_wall(self, center_x, center_y, inner_radius, outer_radius, angle, thickness):
        start_point = (center_x + inner_radius * math.cos(angle), center_y + inner_radius * math.sin(angle))
        end_point = (center_x + outer_radius * math.cos(angle), center_y + outer_radius * math.sin(angle))
        pygame.draw.line(self.screen, BLACK, start_point, end_point, thickness)

    def draw_path(self):
        center = (WINDOW_SIZE // 2, WINDOW_SIZE // 2)
        # Define the path thickness
        path_thickness = WINDOW_SIZE // (4 * self.rings)
        for i in range(len(self.path) - 1):
            start = self.path[i]
            end = self.path[i + 1]
            # Calculate angles for the midpoint of each sector
            start_angle = (start[1] * (360 / self.sectors) + (180 / self.sectors)) % 360
            end_angle = (end[1] * (360 / self.sectors) + (180 / self.sectors)) % 360
            # Calculate the positions for the start and end sectors
        # start_pos = (center[0] + inner_radius * math.cos(math.radians(start_angle)),
        #               center[1] + inner_radius * math.sin(math.radians(start_angle)))
        #  end_pos = (center[0] + outer_radius * math.cos(math.radians(end_angle)),
        #             center[1] + outer_radius * math.sin(math.radians(end_angle)))
        # Draw line for the path
        # pygame.draw.line(self.screen, GREEN, start_pos, end_pos, path_thickness)

    def get_sector_info(self, ring, sector):
        return self.maze[ring][sector]

    def generate_maze(self):
        visited_cells = []
        # Example logic: mark all sectors as walls and set radial walls
        # middle right of maze is (9, 0) (ring, sector). Count up anti-clockwise. 10 is out of range
        for ring in range(self.rings):
            for sector in range(self.sectors):
                # Set every sector as a wall
                self.maze[ring][sector]['wall'] = True

                # Set radial walls for each sector
                self.maze[ring][sector]['radial_walls'] = [False, False]  # [start, end] of sector

                if ring == 0:
                    self.maze[ring][sector]['radial_walls'] = [False, False]  # no radial walls in centre

        """
        # Create a simple spiral pattern, ensuring solvability
        for ring in range(self.rings):
            # Open a sector in each ring to create a spiral path
            open_sector = (ring * 2) % self.sectors
            self.maze[ring][open_sector] = False
        """

        # self.maze[8][1]['wall'] = False

        # open a random wall in the outer ring to make an exit
        self.maze[self.rings - 1][random.randint(0, self.sectors - 1)]['wall'] = False

        # self.aldous_broder_maze()
        self.example_maze()

    def example_maze(self):
        # random holes for every ring, no radial walls. Simple case
        for ring in range(self.rings - 1):
            for _ in range(2):
                self.maze[ring][random.randint(0, 9)]['wall'] = False

        self.maze[8][1]['wall'] = False
        # self.maze[8][8]['radial_walls'] = [True, False]
        # self.maze[8][9]['radial_walls'] = [False, True]

    def aldous_broder_maze(self):
        visited_cells = []
        step_number = 0
        # Aldous-Broder random maze generation
        while len(visited_cells) < self.rings * self.sectors:  # total number of cells
            # choose a random cell
            random_ring = random.randint(0, self.rings - 1)  # 10 is out of range
            random_sector = random.randint(0, self.sectors - 1)
            visited_cells.append((random_ring, random_sector))
            # print("Start cell")
            # print(visited_cells)

            # travel to random neighbour
            # print(self.get_neighbours(random_ring, random_sector))
            i = random.randint(0, len(self.get_neighbours(random_ring, random_sector)) - 1)  # neighbour list
            random_neighbour = self.get_neighbours(random_ring, random_sector)[i]
            print("random neighbour to be added is")
            print(random_neighbour)

            # if neighbour not visited
            if random_neighbour not in visited_cells:
                visited_cells.append(random_neighbour)

                print(visited_cells)
                # remove the appropriate walls
                # if the ring number is different, remove an angular wall
                # can put this in one for loop
                if random_neighbour[0] != visited_cells[step_number][0]:
                    # self.maze[random_neighbour[0]][random_neighbour[1]]['wall'] = False
                    # print("wall changed a")
                    pass

                # if sector number is different, remove a radial wall
                # anti-clockwise wall
                elif random_neighbour[1] > visited_cells[step_number][1]:
                    self.maze[random_neighbour[0]][random_neighbour[1]]['radial_walls'] = [False, True]
                    # remove wall of previous cell as well to prevent overlap
                    self.maze[random_neighbour[0]][random_neighbour[1] - 1]['radial_walls'] = [True, False]
                    print("wall changed b")

                # clockwise wall
                elif random_neighbour[1] < visited_cells[step_number][1]:
                    self.maze[random_neighbour[0]][random_neighbour[1]]['radial_walls'] = [True, False]
                    # remove other wall in same place as well to prevent overlap
                    self.maze[random_neighbour[0]][random_neighbour[1] + 1]['radial_walls'] = [False, True]
                    print("wall changed c")

    def is_wall(self, ring, sector):
        return self.maze[ring][sector]['wall']

    def get_neighbours(self, ring, sector):
        neighbors = []
        # Check radial neighbors
        if ring > 0:
            neighbors.append((ring - 1, sector))
        if ring < self.rings - 1:
            neighbors.append((ring + 1, sector))
        # Check angular neighbors
        neighbors.append((ring, (sector - 1) % self.sectors))
        neighbors.append((ring, (sector + 1) % self.sectors))
        return neighbors

    def update(self):
        self.rotate()

    def rotate(self, direction):
        omega = 1  # degrees per step
        if direction == "left":
            self.rotation_angle -= omega
        elif direction == "right":
            self.rotation_angle += omega

        # Keep the angle within 0-360 degrees in case of multiple whole rotations
        self.rotation_angle %= 360


class AStarSolver:
    def __init__(self, maze):
        self.maze = maze

    def find_path(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.calculate_heuristic(start, end)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == end:
                return self.reconstruct_path(came_from, current)

            for neighbour in self.maze.get_neighbours(*current):
                if self.maze.is_wall(*neighbour):
                    continue
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g_score
                    f_score[neighbour] = tentative_g_score + self.calculate_heuristic(neighbour, end)
                    if neighbour not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbour], neighbour))
        return []

    def calculate_heuristic(self, cell, end):
        # Heuristic combining radial and angular distance
        radial_diff = abs(cell[0] - end[0])
        angular_diff = min(abs(cell[1] - end[1]), self.maze.sectors - abs(cell[1] - end[1]))
        return radial_diff + angular_diff

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]  # Return reversed path

class Flashlight:
    def __init__(self, screen, size, radius, alpha=200):
        self.screen = screen
        self.radius = radius
        self.alpha = alpha

        # Create a surface for the flashlight mask
        self.flashlight_mask = pygame.Surface(size, pygame.SRCALPHA)

    def draw(self, position):
        # Reset the flashlight mask
        self.flashlight_mask.fill((0, 0, 0, self.alpha))

        # Draw a transparent circle at the position on the mask
        pygame.draw.circle(self.flashlight_mask, (0, 0, 0, 0), position, self.radius)

        # Blit the flashlight mask onto the screen
        self.screen.blit(self.flashlight_mask, (0, 0))


class MazeGame:
    def __init__(self, maze, ball, screen):
        self.last_mouse_x = None  # first mouse position
        self.ball = ball
        self.maze = maze
        self.screen = screen
        pygame.display.set_caption("Ring-Shaped Maze Solver")
        self.flashlight = Flashlight(screen, (WINDOW_SIZE, WINDOW_SIZE), 70)  # Adjust radius as needed

    def run_game_loop(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                else:
                    #self.handle_keypress(event)
                    self.handle_mouse_move(event)

            # Clear the screen and draw the maze and ball
            self.screen.fill(BLACK)
            self.maze.draw_maze()
            self.maze.draw_path()
            self.ball.update()

            # Draw the flashlight effect
            self.flashlight.draw((int(self.ball.x), int(self.ball.y)))
            pygame.display.flip()

    def handle_keypress(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                self.maze.rotate("left")
            elif event.key == pygame.K_d:
                self.maze.rotate("right")

    def handle_mouse_move(self, event):
        if event.type == pygame.MOUSEMOTION:
            # Get the current mouse x position
            current_mouse_x = pygame.mouse.get_pos()[0]
            #print(current_mouse_x)

            # Initialize last_mouse_x if it's not set
            if self.last_mouse_x is None:
                self.last_mouse_x = current_mouse_x

            # Determine the rotation direction based on mouse movement
            if current_mouse_x > self.last_mouse_x:
                self.maze.rotate("right")
            elif current_mouse_x < self.last_mouse_x:
                self.maze.rotate("left")

            # Update the last mouse x position
            self.last_mouse_x = current_mouse_x


# Main function
def main():
    pygame.init()
    size = 30  # Size of the maze
    start = (0, 0)  # Start position
    end = (size - 1, size - 1)  # End position

    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))  # Create screen object

    # Create a temporary RingMaze instance for path calculation
    temp_maze = RingMaze(10, 10, screen)  # rings, sectors
    solver = AStarSolver(temp_maze)
    path = solver.find_path(start, end)  # find the path using the solver

    # Create the final RingMaze instance with the calculated path
    ring_maze = RingMaze(10, 10, screen, path)
    ball = Ball(WINDOW_SIZE // 2, WINDOW_SIZE // 2, 10, RED, screen, ring_maze)
    game = MazeGame(ring_maze, ball, screen)
    game.run_game_loop()
    pygame.quit()


if __name__ == "__main__":
    main()
