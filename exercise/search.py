class Node:
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0  # Distance from start node
        self.h = 0  # Heuristic based distance to end node
        self.f = 0  # Total cost

    def __eq__(self, other):
        return self.position == other.position

class Search:
    def __init__(self, maze):
        self.maze = maze

    def _heuristic(self, current, goal):
        # Using Manhattan distance as heuristic
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def find_path(self, start, end):
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        open_list = []
        closed_list = []

        open_list.append(start_node)

        while len(open_list) > 0:
            current_node = min(open_list, key=lambda node: node.f)
            open_list.remove(current_node)
            closed_list.append(current_node)

            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]  # Return reversed path

            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                if node_position[0] > (len(self.maze) - 1) or node_position[0] < 0 or node_position[1] > (len(self.maze[len(self.maze) - 1]) - 1) or node_position[1] < 0:
                    continue

                if self.maze[node_position[0]][node_position[1]] != 0:
                    continue

                new_node = Node(current_node, node_position)

                children.append(new_node)

            for child in children:
                if child in closed_list:
                    continue

                child.g = current_node.g + 1
                child.h = self._heuristic(child.position, end_node.position)
                child.f = child.g + child.h

                if child in open_list:
                    continue

                open_list.append(child)

        return []  # No path found

# Example usage (assuming maze is a 2D list)
# astar = AStar(maze)
# path = astar.find_path(start_position, end_position)
