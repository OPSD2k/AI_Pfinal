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


class FingerGun:
    def __init__(self):
        self.cap = cv.VideoCapture(0)

    # read a camera frame
    def read_frame(self):
        _, img = self.cap.read()
        cv.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
        crop_img = img[100:300, 100:300]
        grey = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
        value = (35, 35)
        blurred_ = cv.GaussianBlur(grey, value, 0)
        _, thresholded = cv.threshold(blurred_, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        self.analyse_contours(thresholded)

    def analyse_contours(self, thresholded):
        self.thresholded = thresholded # pass variable
        image, contours, hierarchy = cv.findContours(self.thresholded.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        count1 = max(contours, key=lambda x: cv.contourArea(x))
        x, y, w, h = cv.boundingRect(count1)
        cv.rectangle(self.crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
        hull = cv.convexHull(count1)
        drawing = np.zeros(self.crop_img.shape, np.uint8)
        cv.drawContours(drawing, [count1], 0, (0, 255, 0), 0)
        cv.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
        hull = cv.convexHull(count1, returnPoints=False)
        defects = cv.convexityDefects(count1, hull)

        count_defects = 0
        cv.drawContours(self.thresholded, contours, -1, (0, 255, 0), 3)

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(count1[s][0])
            end = tuple(count1[e][0])
            far = tuple(count1[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            if angle <= 90:
                count_defects += 1
                cv.circle(self.crop_img, far, 1, [0, 0, 255], -1)

            cv.line(self.crop_img, start, end, [0, 255, 0], 2)

        if count_defects == 1:
            cv.putText(self.img, "2 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        elif count_defects == 2:
            str = "3 fingers"
            cv.putText(self.img, str, (5, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        elif count_defects == 3:
            cv.putText(self.img, "4 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        elif count_defects == 4:
            cv.putText(self.img, "5 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        elif count_defects == 0:
            cv.putText(self.img, "one", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))

    def show_image(self):
        cv.imshow('main window', self.img)
        all_img = np.hstack((self.drawing, self.crop_img))
        cv.imshow('Contours', all_img)
        k = cv.waitKey(10)
        if k == 27:
            return None  # "break"


class RingMaze:
    def __init__(self, rings, sectors):
        self.rings = rings
        self.sectors = sectors
        self.maze = [[False for _ in range(sectors)] for _ in range(rings)]
        self.generate_maze()

    def generate_maze(self):
        for ring in range(self.rings):
            # Initialize all sectors with walls
            for sector in range(self.sectors):
                self.maze[ring][sector] = True

            # Determine the number of gaps
            if ring == self.rings - 1:
                # Only one opening for the outermost ring
                num_gaps = 1
            else:
                # Randomly choose between 2 to 4 openings for other rings
                num_gaps = random.randint(2, 4)

            # Create gaps ensuring no two gaps are adjacent
            gaps_created = 0
            attempts = 0
            while gaps_created < num_gaps and attempts < self.sectors * 2:
                gap = random.randint(0, self.sectors - 1)
                left_neighbor = (gap - 1) % self.sectors
                right_neighbor = (gap + 1) % self.sectors
                # Place a gap only if both neighbors are walls
                if self.maze[ring][left_neighbor] and self.maze[ring][right_neighbor]:
                    self.maze[ring][gap] = False
                    gaps_created += 1
                attempts += 1

            # Special case for the last ring to ensure there's one gap
            if ring == self.rings - 1 and gaps_created == 0:
                gap = random.randint(0, self.sectors - 1)
                self.maze[ring][gap] = False

    def is_wall(self, ring, sector):
        return self.maze[ring][sector]

    def get_neighbors(self, ring, sector):
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

            for neighbor in self.maze.get_neighbors(*current):
                if self.maze.is_wall(*neighbor):
                    continue
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.calculate_heuristic(neighbor, end)
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
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


class MazeGame:
    def __init__(self, maze, path):
        self.maze = maze
        self.path = path
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Ring-Shaped Maze Solver")
        #self.finger_gun = FingerGun()  # instantiate finger gun class

    def draw_maze(self):
        center_x, center_y = WINDOW_SIZE // 2, WINDOW_SIZE // 2
        # Define the thickness of the walls
        wall_thickness = max(1, WINDOW_SIZE // (8 * self.maze.rings))

        # Draw the angular walls as arcs
        for ring in range(self.maze.rings):
            # print(ring)
            sectors_drawn = 0
            inner_radius = ring * (WINDOW_SIZE // 2) // self.maze.rings
            outer_radius = (ring + 1) * (WINDOW_SIZE // 2) // self.maze.rings

            # only one gap in outer ring
            for sector in range(self.maze.sectors):
                print("number of sectors is " + str(self.maze.sectors))
                """
                if ring == self.maze.rings - 1:  # outer wall, final ring
                    sector_is_wall = True
                else:
                    sector_is_wall = self.maze.is_wall(ring, sector)  # returns true or false
                # ensure every ring has at least 1 gap. There can't be the max number of sectors
                if sector_is_wall and sectors_drawn < self.maze.sectors - 1:"""
                if self.maze.is_wall(ring, sector):
                    start_angle = math.radians(sector * (360 / self.maze.sectors))
                    end_angle = math.radians((sector + 1) * (360 / self.maze.sectors))

                    # if (start)
                    # Define the bounding rectangle for the arc
                    bounding_rect = [center_x - outer_radius, center_y - outer_radius, 2 * outer_radius,
                                     2 * outer_radius]
                    pygame.draw.arc(self.screen, BLACK, bounding_rect, start_angle, end_angle, wall_thickness)
                    sectors_drawn += 1
            # print(sectors_drawn)

        # Draw radial walls
        for sector in range(self.maze.sectors):
            for ring in range(self.maze.rings - 1):
                if self.maze.is_wall(ring, sector) != self.maze.is_wall(ring + 1, sector):
                    angle = math.radians(sector * (360 / self.maze.sectors))
                    inner_point = (center_x + inner_radius * math.cos(angle), center_y + inner_radius * math.sin(angle))
                    outer_point = (center_x + outer_radius * math.cos(angle), center_y + outer_radius * math.sin(angle))
                    pygame.draw.line(self.screen, BLACK, inner_point, outer_point, wall_thickness)

    def draw_path(self):
        center = (WINDOW_SIZE // 2, WINDOW_SIZE // 2)
        # Define the path thickness
        path_thickness = WINDOW_SIZE // (4 * self.maze.rings)
        for i in range(len(self.path) - 1):
            start = self.path[i]
            end = self.path[i + 1]
            # Calculate angles for the midpoint of each sector
            start_angle = (start[1] * (360 / self.maze.sectors) + (180 / self.maze.sectors)) % 360
            end_angle = (end[1] * (360 / self.maze.sectors) + (180 / self.maze.sectors)) % 360
            # Calculate the positions for the start and end sectors
            start_pos = (center[0] + inner_radius * math.cos(math.radians(start_angle)),
                         center[1] + inner_radius * math.sin(math.radians(start_angle)))
            end_pos = (center[0] + outer_radius * math.cos(math.radians(end_angle)),
                       center[1] + outer_radius * math.sin(math.radians(end_angle)))
            # Draw line for the path
            pygame.draw.line(self.screen, GREEN, start_pos, end_pos, path_thickness)

    def run_game_loop(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            #self.finger_gun.read_frame()
            #self.finger_gun.analyse_contours()
            #self.finger_gun.show_image()
            # Fill the screen with white before drawing
            self.screen.fill(WHITE)
            # Draw the maze and the path
            self.draw_maze()
            self.draw_path()
            # Update the display
            pygame.display.flip()


# Main function
def main():
    pygame.init()
    size = 30  # Size of the maze
    start = (0, 0)  # Start position
    end = (size - 1, size - 1)  # End position

    ring_maze = RingMaze(10, 10)  # size, sectors
    solver = AStarSolver(ring_maze)
    path = solver.find_path(start, end)
    game = MazeGame(ring_maze, path)
    game.run_game_loop()
    pygame.quit()


if __name__ == "__main__":
    main()
