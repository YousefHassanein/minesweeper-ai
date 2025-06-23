import numpy as np
from minesweeper import MinesweeperGame, CellState
from typing import Tuple, Dict, Any

class MinesweeperEnv:
    """
    A Gym-style environment for the Minesweeper game.
    
    This environment wraps the core Minesweeper game logic and provides a
    standard interface for reinforcement learning agents.

    POMDP Definition:
    - Hidden State: The full board with all mine locations.
    - Observation: A numerical matrix of the currently visible board state.
    - Actions: An integer representing revealing or flagging a cell.
    - Reward: A scalar value indicating the outcome of an action.
    - Transition: The change in observation after an action is taken.
    """

    def __init__(self, rows: int = 9, cols: int = 9, num_mines: int = 10, discount_factor: float = 0.95):
        """
        Initialize the Minesweeper environment.
        
        Args:
            rows (int): The number of rows on the board.
            cols (int): The number of columns on the board.
            num_mines (int): The number of mines on the board.
            discount_factor (float): The discount factor (gamma) for future rewards.
        """
        self.game = MinesweeperGame(rows, cols, num_mines)
        self.discount_factor = discount_factor
        
        # Action space: (rows * cols * 2)
        # 0 to (rows*cols - 1) for revealing a cell
        # (rows*cols) to (rows*cols*2 - 1) for flagging a cell
        self.action_space_size = self.game.board.rows * self.game.board.cols * 2
        
        # Observation space: a (rows x cols) matrix
        # -2: Flagged
        # -1: Hidden
        # 0-8: Revealed with adjacent mine count
        self.observation_shape = (self.game.board.rows, self.game.board.cols)

    def _get_observation(self) -> np.ndarray:
        """
        Generate the numerical observation matrix from the current game state.
        
        Returns:
            np.ndarray: A (rows x cols) NumPy array representing the board.
        """
        observation = np.full(self.observation_shape, -1, dtype=int)
        for r in range(self.game.board.rows):
            for c in range(self.game.board.cols):
                cell = self.game.board.get_cell(r, c)
                if cell.state == CellState.REVEALED:
                    observation[r, c] = cell.adjacent_mines
                elif cell.state == CellState.FLAGGED:
                    observation[r, c] = -2
                else: # Hidden
                    observation[r, c] = -1
        return observation

    def _action_to_tuple(self, action: int) -> Tuple[int, int, str]:
        """
        Convert an integer action to a (row, col, type) tuple.
        
        Args:
            action (int): The integer action to convert.
            
        Returns:
            Tuple[int, int, str]: The (row, col, action_type) representation.
        """
        if not (0 <= action < self.action_space_size):
            raise ValueError(f"Action {action} is out of bounds for action space size {self.action_space_size}")
            
        rows, cols = self.observation_shape
        num_reveal_actions = rows * cols
        
        if action < num_reveal_actions:
            action_type = "reveal"
            cell_index = action
        else:
            action_type = "flag"
            cell_index = action - num_reveal_actions
            
        row = cell_index // cols
        col = cell_index % cols
        
        return row, col, action_type

    def reset(self) -> np.ndarray:
        """
        Reset the environment to the beginning of a new episode.
        
        Returns:
            np.ndarray: The initial observation of the new game.
        """
        self.game.reset()
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action (int): An action provided by the agent.
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: A tuple containing:
                - observation (np.ndarray): The agent's observation of the current environment.
                - reward (float): The amount of reward returned after previous action.
                - done (bool): Whether the episode has ended.
                - info (Dict): Contains auxiliary diagnostic information.
        """
        try:
            action_tuple = self._action_to_tuple(action)
        except ValueError as e:
            # Handle invalid action gracefully
            observation = self._get_observation()
            reward = -1.0 # Penalty for invalid action
            done = self.game.game_over or self.game.won
            info = {'error': str(e)}
            return observation, reward, done, info

        # The game's step method already returns the reward and done status
        _, reward, done, info = self.game.step(action_tuple)
        
        observation = self._get_observation()
        
        return observation, reward, done, info

    def render(self):
        """
        Render the environment's current state (e.g., to the console).
        """
        self.game.display()

def main():
    """
    Example usage of the MinesweeperEnv.
    This demonstrates the basic API: reset, step, and render.
    """
    print("Testing the Minesweeper Environment")
    print("=" * 40)
    
    # Create the environment
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3)
    
    # Reset the environment and get the initial observation
    obs = env.reset()
    print("Initial Observation (State):")
    print(obs)
    
    done = False
    total_reward = 0
    num_moves = 0
    
    # Run a short episode with random actions
    while not done and num_moves < 10:
        # Choose a random valid action
        action = np.random.randint(0, env.action_space_size)
        action_tuple = env._action_to_tuple(action)
        
        print(f"\n--- Move {num_moves + 1} ---")
        print(f"Action taken: {action} -> {action_tuple[2]} at ({action_tuple[0]}, {action_tuple[1]})")

        # Take the step
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        num_moves += 1
        
        print("New Observation:")
        print(obs)
        print(f"Reward: {reward:.2f}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        
    print("\n" + "=" * 40)
    print("Episode Finished!")
    print(f"Total moves: {num_moves}")
    print(f"Total reward: {total_reward:.2f}")

    if info.get('won'):
        print("Result: 🎉 You won!")
    elif info.get('game_over'):
        print("Result: 💥 Game Over!")
    else:
        print("Result: Episode ended due to move limit.")
        
    print("\nFinal Board State:")
    env.render()


if __name__ == "__main__":
    main() 