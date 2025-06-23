import numpy as np
import pickle
from typing import Any, Tuple, Optional

class QLearningAgent:
    """
    Q-Learning agent for MinesweeperEnv.
    Maintains a Q-table and supports epsilon-greedy exploration.
    """
    def __init__(self, action_space_size: int, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        self.Q = {}  # Q-table: dict mapping (state_key, action) -> value
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.action_space_size = action_space_size

    def _state_to_key(self, state: np.ndarray) -> Any:
        """
        Convert a state (observation) to a hashable key for the Q-table.
        Uses the flattened tuple of the state array.
        """
        return tuple(state.flatten())

    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.
        """
        state_key = self._state_to_key(state)
        if np.random.rand() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.action_space_size)
        else:
            # Exploit: best known action
            q_values = [self.Q.get((state_key, a), 0.0) for a in range(self.action_space_size)]
            max_q = max(q_values)
            # Break ties randomly
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Update the Q-table using the Q-learning update rule.
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        old_q = self.Q.get((state_key, action), 0.0)
        if done:
            target = reward
        else:
            next_qs = [self.Q.get((next_state_key, a), 0.0) for a in range(self.action_space_size)]
            target = reward + self.gamma * max(next_qs)
        new_q = (1 - self.alpha) * old_q + self.alpha * target
        self.Q[(state_key, action)] = new_q

    def save(self, filename: str):
        """
        Save the Q-table to a file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.Q, f)

    def load(self, filename: str):
        """
        Load the Q-table from a file.
        """
        with open(filename, 'rb') as f:
            self.Q = pickle.load(f) 