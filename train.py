import numpy as np
from minesweeper_env import MinesweeperEnv
from q_learning_agent import QLearningAgent

EPISODES = 5000
MAX_STEPS = 200
SAVE_PATH = 'q_table.pkl'

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 0.1

# Environment parameters
ROWS = 5
COLS = 5
MINES = 3

def train():
    env = MinesweeperEnv(rows=ROWS, cols=COLS, num_mines=MINES)
    agent = QLearningAgent(
        action_space_size=env.action_space_size,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON
    )

    win_count = 0
    total_reward = 0.0
    reward_history = []

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1

        total_reward += episode_reward
        reward_history.append(episode_reward)
        if info.get('won', False):
            win_count += 1

        if episode % 500 == 0:
            avg_reward = np.mean(reward_history[-500:])
            print(f"Episode {episode}/{EPISODES} | Win rate: {win_count/episode:.2%} | Avg reward (last 500): {avg_reward:.2f}")

    # Save Q-table
    agent.save(SAVE_PATH)
    print(f"\nTraining complete. Q-table saved to {SAVE_PATH}")
    print(f"Total episodes: {EPISODES}")
    print(f"Win rate: {win_count/EPISODES:.2%}")
    print(f"Average reward: {total_reward/EPISODES:.2f}")

if __name__ == "__main__":
    train() 