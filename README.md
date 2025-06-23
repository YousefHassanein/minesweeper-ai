# Minesweeper AI - Unsupervised Machine Learning Project

A well-structured Minesweeper implementation designed for unsupervised machine learning research. The project uses good OOP principles with a matrix-based board representation for easy ML integration.

## Features

- **Matrix-based Board**: Uses 2D matrices for easy numerical representation
- **Good OOP Design**: Clean separation of concerns with dedicated classes
- **ML-Ready Interface**: Built-in support for machine learning agents
- **Multiple Agent Types**: Random and heuristic agents for baseline comparison
- **Training Data Collection**: Automatic collection of game states and moves
- **Evaluation Framework**: Performance metrics for agent comparison

## Project Structure

```
minesweeper-ai/
├── minesweeper.py      # Core game logic and classes
├── ml_agent.py         # ML agent framework and implementations
├── test_minesweeper.py # Test suite and demonstrations
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Classes Overview

### Core Game Classes

- **`Cell`**: Represents individual cells with state management
- **`Board`**: Matrix-based game board with mine placement and logic
- **`MinesweeperGame`**: Main game controller with move validation

### ML Agent Classes

- **`MLAgent`**: Base class for all machine learning agents
- **`RandomAgent`**: Simple random agent for baseline
- **`SimpleHeuristicAgent`**: Rule-based agent using Minesweeper strategies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd minesweeper-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Play the Game Manually

```bash
python minesweeper.py
```

Commands:
- `r <row> <col>` - Reveal a cell
- `f <row> <col>` - Flag/unflag a cell
- `s` - Show all mines
- `q` - Quit

### Run Tests and Demonstrations

```bash
python test_minesweeper.py
```

This will run comprehensive tests including:
- Basic game functionality
- Agent behavior
- Performance evaluation
- ML data generation

### Use in Your ML Code

```python
from minesweeper import MinesweeperGame
from ml_agent import MLAgent, RandomAgent

# Create a game
game = MinesweeperGame(9, 9, 10)

# Create an agent
agent = RandomAgent(game)

# Get board state for ML
state_matrix = agent.get_board_state_matrix()
print(f"Board state shape: {state_matrix.shape}")

# Make moves
action = agent.choose_action()
if action:
    row, col, action_type = action
    agent.make_move(row, col, action_type)

# Get training data
training_data = agent.get_training_data()
```

## ML Integration Features

### Board State Matrix

The game provides a 3-channel numerical matrix:
- **Channel 0**: Revealed state (0=hidden, 1=revealed)
- **Channel 1**: Adjacent mine count (normalized 0-1, -1 for hidden)
- **Channel 2**: Flagged state (0=not flagged, 1=flagged)

### Training Data Collection

Each agent automatically collects:
- Game state matrices before each move
- Move history with coordinates and actions
- Final game outcomes and scores

### Evaluation Metrics

Built-in evaluation provides:
- Win rate percentage
- Average moves per game
- Average moves per win
- Performance comparison between agents

## Unsupervised Learning Opportunities

This framework is designed for unsupervised learning approaches:

1. **Pattern Recognition**: Learn common board patterns and safe moves
2. **Clustering**: Group similar game states and strategies
3. **Dimensionality Reduction**: Find latent representations of board states
4. **Reinforcement Learning**: Use game outcomes as reward signals
5. **Self-Supervised Learning**: Predict safe moves from board context

## Example ML Workflow

```python
from ml_agent import evaluate_agent, RandomAgent
import numpy as np

# Collect training data
results = evaluate_agent(RandomAgent, num_games=1000)
training_data = results['training_data']

# Extract features for unsupervised learning
states = np.array(training_data['states'])
moves = training_data['moves']

# Apply your unsupervised learning algorithm
# e.g., clustering, autoencoders, etc.
```

## Contributing

This project is designed to be easily extensible:

1. **Add New Agents**: Inherit from `MLAgent` and implement `choose_action()`
2. **Modify Board Representation**: Extend the state matrix channels
3. **Add New Metrics**: Extend the evaluation framework
4. **Implement ML Algorithms**: Use the provided data collection interface

## License

This project is open source and available under the MIT License.
