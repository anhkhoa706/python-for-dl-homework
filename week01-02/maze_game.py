import torch
import random
from collections import deque

class MazeGame:
    """Class representing the Maze game logic."""
    def __init__(self, size):
        self.size = size
        self.maze = torch.full((size, size), -1)  # -1 means unfilled
        self.start_position = (0, 0)
        self.player_position = (0, 0)
        self.exit_position = (size - 1, size - 1)
        self.status_message = "Use buttons or arrow keys to move."
        
        # Generate maze
        self.create_solution_path()
        self.fill_remaining_cells()

    def create_solution_path(self):
        """
        Ensures at least two solution paths from start to exit.
        Two paths are generated using random right/down moves.
        If the two paths are identical, the second one is regenerated.
        All cells from both paths are marked as walkable (0).
        """
        def generate_path():
            x, y = 0, 0
            path = [(x, y)]
            while (x, y) != self.exit_position:
                moves = []
                if x < self.size - 1:
                    moves.append((x + 1, y))  # Move Down
                if y < self.size - 1:
                    moves.append((x, y + 1))  # Move Right
                x, y = random.choice(moves)
                path.append((x, y))
            return path

        # Generate the first solution path.
        path1 = generate_path()
        # Generate a second solution path.
        path2 = generate_path()
        # If both paths are identical, re-generate the second one until it's different.
        while path2 == path1:
            path2 = generate_path()
        
        # Mark all cells in both paths as walkable.
        for i, j in set(path1 + path2):
            self.maze[i, j] = 0

    def fill_remaining_cells(self, wall_probability=0.8):
        """
        Fills the remaining cells with walls (1) and paths (0) using a fixed wall probability.
        Args:
            wall_probability: The chance (between 0 and 1) that a cell becomes a wall.
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.maze[i, j] == -1:
                    self.maze[i, j] = 1 if random.random() < wall_probability else 0

    def move_player(self, direction):
        """Moves the player if the move is valid."""
        x, y = self.player_position
        moves = {'up': (x - 1, y), 'down': (x + 1, y), 'left': (x, y - 1), 'right': (x, y + 1)}
        
        if direction in moves:
            nx, ny = moves[direction]
            if 0 <= nx < self.size and 0 <= ny < self.size and self.maze[nx, ny] == 0:
                self.player_position = (nx, ny)
                return "Congratulations! You win!" if self.player_position == self.exit_position else "Move successful!"
        return "Invalid move! Try again."

    def find_shortest_path(self):
        """
        Finds the shortest path from the player's current position to the exit using BFS.
        Returns a list of positions representing the path.
        If no path is found, returns an empty list.
        """
        queue = deque([(self.player_position, [])])
        visited = set()
        
        while queue:
            (x, y), path = queue.popleft()
            visited.add((x, y))

            if (x, y) == self.exit_position:
                return path  # Return shortest path 

            for _, (dx, dy) in [("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.maze[nx, ny] == 0 and (nx, ny) not in visited:
                    queue.append(((nx, ny), path + [(nx, ny)]))
                    visited.add((nx, ny))

        return [] # No solution found  

    def reset(self):
        """Resets the game by generating the maze."""
        self.__init__(self.size)
