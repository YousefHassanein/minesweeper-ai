import random
from typing import List, Tuple, Optional
from enum import Enum
from collections import deque


class CellState(Enum):
    """Enumeration for cell states in the game"""
    HIDDEN = "hidden"
    REVEALED = "revealed"
    FLAGGED = "flagged"


class Cell:
    """Represents a single cell in the Minesweeper board"""
    
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.is_mine = False
        self.state = CellState.HIDDEN
        self.adjacent_mines = 0
    
    def __str__(self) -> str:
        if self.state == CellState.HIDDEN:
            return "□"
        elif self.state == CellState.FLAGGED:
            return "🚩"
        elif self.is_mine:
            return "💣"
        elif self.adjacent_mines == 0:
            return " "
        else:
            return str(self.adjacent_mines)
    
    def reveal(self) -> bool:
        """Reveal the cell. Returns True if it's a mine (game over)"""
        if self.state == CellState.FLAGGED:
            return False
        
        self.state = CellState.REVEALED
        return self.is_mine
    
    def toggle_flag(self) -> bool:
        """Toggle flag state. Returns True if flagged, False if unflagged"""
        if self.state == CellState.REVEALED:
            return False
        
        if self.state == CellState.HIDDEN:
            self.state = CellState.FLAGGED
            return True
        else:
            self.state = CellState.HIDDEN
            return False


class Board:
    """Represents the Minesweeper game board"""
    
    def __init__(self, rows: int, cols: int, num_mines: int):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.board = [[Cell(row, col) for col in range(cols)] for row in range(rows)]
        self.game_over = False
        self.first_move = True
        self.revealed_count = 0
        
    def place_mines(self, first_row: int, first_col: int):
        """Place mines randomly, avoiding the first clicked cell"""
        positions = []
        for row in range(self.rows):
            for col in range(self.cols):
                if row != first_row or col != first_col:
                    positions.append((row, col))
        
        mine_positions = random.sample(positions, self.num_mines)
        
        for row, col in mine_positions:
            self.board[row][col].is_mine = True
        
        # Calculate adjacent mine counts
        self._calculate_adjacent_mines()
    
    def _calculate_adjacent_mines(self):
        """Calculate the number of adjacent mines for each cell"""
        for row in range(self.rows):
            for col in range(self.cols):
                if not self.board[row][col].is_mine:
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            new_row, new_col = row + dr, col + dc
                            if (0 <= new_row < self.rows and 
                                0 <= new_col < self.cols and 
                                self.board[new_row][new_col].is_mine):
                                count += 1
                    self.board[row][col].adjacent_mines = count
    
    def get_adjacent_cells(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all adjacent cell positions"""
        adjacent = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    adjacent.append((new_row, new_col))
        return adjacent
    
    def reveal_cell(self, row: int, col: int) -> bool:
        """Reveal a cell and handle game logic. Returns True if mine was hit"""
        if self.game_over or not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        
        cell = self.board[row][col]
        
        # Place mines on first move
        if self.first_move:
            self.place_mines(row, col)
            self.first_move = False
        
        # Handle flagged cells
        if cell.state == CellState.FLAGGED:
            return False
        
        # Reveal the cell
        if cell.reveal():
            self.game_over = True
            return True
        
        self.revealed_count += 1
        
        # If cell has no adjacent mines, reveal adjacent cells using BFS
        if cell.adjacent_mines == 0:
            self._reveal_adjacent_cells_bfs(row, col)
        
        return False
    
    def _reveal_adjacent_cells_bfs(self, start_row: int, start_col: int):
        """Reveal adjacent cells using BFS instead of recursion"""
        queue = deque([(start_row, start_col)])
        visited = set()
        
        while queue:
            row, col = queue.popleft()
            
            if (row, col) in visited:
                continue
            visited.add((row, col))
            
            # Reveal all adjacent cells
            for adj_row, adj_col in self.get_adjacent_cells(row, col):
                adj_cell = self.board[adj_row][adj_col]
                
                if adj_cell.state == CellState.HIDDEN:
                    adj_cell.reveal()
                    self.revealed_count += 1
                    
                    # If this cell also has no adjacent mines, add to queue
                    if adj_cell.adjacent_mines == 0:
                        queue.append((adj_row, adj_col))
    
    def flag_cell(self, row: int, col: int) -> bool:
        """Flag or unflag a cell"""
        if self.game_over or not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        
        cell = self.board[row][col]
        return cell.toggle_flag()
    
    def is_won(self) -> bool:
        """Check if the game is won"""
        return self.revealed_count == (self.rows * self.cols - self.num_mines)
    
    def get_cell(self, row: int, col: int) -> Optional[Cell]:
        """Get a cell at the specified position"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.board[row][col]
        return None
    
    def display(self, show_mines: bool = False):
        """Display the current state of the board"""
        # Print column numbers
        print("   ", end="")
        for col in range(self.cols):
            print(f"{col:2}", end=" ")
        print()
        
        # Print board
        for row in range(self.rows):
            print(f"{row:2} ", end="")
            for col in range(self.cols):
                cell = self.board[row][col]
                if show_mines and cell.is_mine and cell.state != CellState.REVEALED:
                    print("💣 ", end="")
                else:
                    print(f"{cell} ", end="")
            print()


class MinesweeperGame:
    """Main game controller class with Gym-style interface for ML"""
    
    def __init__(self, rows: int = 9, cols: int = 9, num_mines: int = 10):
        self.board = Board(rows, cols, num_mines)
        self.game_over = False
        self.won = False
        self.move_count = 0
    
    def reset(self) -> dict:
        """Reset the game to initial state (Gym-style interface)"""
        self.board = Board(self.board.rows, self.board.cols, self.board.num_mines)
        self.game_over = False
        self.won = False
        self.move_count = 0
        return self.get_game_state()
    
    def step(self, action: Tuple[int, int, str]) -> Tuple[dict, float, bool, dict]:
        """Take a step in the game (Gym-style interface)
        
        Args:
            action: Tuple of (row, col, action_type) where action_type is "reveal" or "flag"
            
        Returns:
            (state, reward, done, info)
        """
        row, col, action_type = action
        
        if action_type == "reveal":
            hit_mine = self.board.reveal_cell(row, col)
            if hit_mine:
                self.game_over = True
                reward = -10.0  # Penalty for hitting mine
            else:
                reward = 1.0 if self.board.is_won() else 0.1  # Small reward for safe moves
        elif action_type == "flag":
            self.board.flag_cell(row, col)
            reward = 0.0  # No immediate reward for flagging
        else:
            reward = -1.0  # Penalty for invalid action
        
        self.move_count += 1
        
        # Check for win
        if self.board.is_won():
            self.won = True
            reward = 10.0  # Large reward for winning
        
        done = self.game_over or self.won
        info = {
            'move_count': self.move_count,
            'revealed_count': self.board.revealed_count,
            'won': self.won,
            'game_over': self.game_over
        }
        
        return self.get_game_state(), reward, done, info
    
    def make_move(self, row: int, col: int, action: str = "reveal") -> bool:
        """Legacy method for backward compatibility"""
        _, _, done, _ = self.step((row, col, action))
        return done
    
    def get_game_state(self) -> dict:
        """Get the current game state for ML training"""
        state = {
            'board': [],
            'game_over': self.game_over,
            'won': self.won,
            'rows': self.board.rows,
            'cols': self.board.cols,
            'num_mines': self.board.num_mines,
            'revealed_count': self.board.revealed_count,
            'move_count': self.move_count
        }
        
        for row in range(self.board.rows):
            board_row = []
            for col in range(self.board.cols):
                cell = self.board.get_cell(row, col)
                cell_state = {
                    'state': cell.state.value,
                    'is_mine': cell.is_mine,
                    'adjacent_mines': cell.adjacent_mines,
                    'row': cell.row,
                    'col': cell.col
                }
                board_row.append(cell_state)
            state['board'].append(board_row)
        
        return state
    
    def display(self, show_mines: bool = False):
        """Display the game board"""
        print(f"\nMinesweeper ({self.board.rows}x{self.board.cols}, {self.board.num_mines} mines)")
        print("=" * 50)
        self.board.display(show_mines)
        
        if self.game_over:
            print("\n💥 Game Over! You hit a mine!")
            self.board.display(show_mines=True)
        elif self.won:
            print("\n🎉 Congratulations! You won!")
        else:
            print(f"\nRevealed: {self.board.revealed_count}/{self.board.rows * self.board.cols - self.board.num_mines}")
            print(f"Moves: {self.move_count}")


# CLI interface separated from core logic
def run_cli_game():
    """Run the command-line interface for the game"""
    print("Welcome to Minesweeper!")
    print("Commands:")
    print("  r <row> <col> - Reveal cell")
    print("  f <row> <col> - Flag/unflag cell")
    print("  q - Quit")
    print("  s - Show all mines")
    
    # Create game
    game = MinesweeperGame(9, 9, 10)
    
    while not game.game_over and not game.won:
        game.display()
        
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'q':
                print("Goodbye!")
                break
            elif command == 's':
                game.display(show_mines=True)
                continue
            elif command.startswith('r ') or command.startswith('f '):
                parts = command.split()
                if len(parts) == 3:
                    action = parts[0]
                    row = int(parts[1])
                    col = int(parts[2])
                    
                    if action == 'r':
                        game.make_move(row, col, "reveal")
                    elif action == 'f':
                        game.make_move(row, col, "flag")
                else:
                    print("Invalid command format. Use: r <row> <col> or f <row> <col>")
            else:
                print("Invalid command. Use r, f, s, or q.")
                
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid row and column numbers.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
    
    if game.game_over or game.won:
        game.display(show_mines=True)


if __name__ == "__main__":
    run_cli_game() 