import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from classes import *


black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)
cyan = (0, 255, 255)
magenta = (255, 0, 255)

colors = {
            1: red,
            2: green,
            3: blue,
            4: yellow,
            5: cyan,
            6: magenta
        }

def choose_piece(color_id):
    selection = random.randint(1, 8)

    if selection == 0:
        return LargeSquare(color_id)
    elif selection == 1:
        return Square(color_id)
    elif selection == 2:
        return VerticalLine(color_id)
    elif selection == 3:
        return TwoHorizontal(color_id)
    elif selection == 4:
        return ThreeHorizontal(color_id)
    elif selection == 5:
        return FourHorizontal(color_id)
    elif selection == 6:
        return FiveHorizontal(color_id)
    elif selection == 7:
        return TwoVertical(color_id)
    elif selection == 8:
        return ThreeVertical(color_id)
    elif selection == 9:
        return FourVertical(color_id)
    
GRID_SIZE = 50  # Size of each grid cell (in pixels)
GRID_WIDTH = 8  # Width of the grid (8 columns)
GRID_HEIGHT = 8  # Height of the grid (8 rows)
WIDTH = GRID_WIDTH * GRID_SIZE
HEIGHT = GRID_HEIGHT * GRID_SIZE

# Q-learning setup
q_values = {}
epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.5

def get_state(grid):
    return tuple(grid.flatten())  # Convert grid to a hashable tuple

def get_possible_actions(piece, grid):
    actions = []
    piece_height, piece_width = piece.shape.shape
    for x in range(GRID_HEIGHT - piece_height + 1):
        for y in range(GRID_WIDTH - piece_width + 1):
            if piece.can_place_piece(piece, y, x, grid):
                actions.append((x, y))
    return actions

def choose_action(state, piece, grid):
    if random.random() < epsilon:  # Explore
        possible_actions = get_possible_actions(piece, grid)
        if possible_actions:
            return random.choice(possible_actions)
        else:
            return None
    else:  # Exploit
        max_q = -np.inf
        best_action = None
        possible_actions = get_possible_actions(piece, grid)
        if not possible_actions:
            return None
        for action in possible_actions:
            if (state, tuple(piece.shape.flatten()), action) in q_values:
                q = q_values[(state, tuple(piece.shape.flatten()), action)]
                if q > max_q:
                    max_q = q
                    best_action = action
            else:
                q_values[(state, tuple(piece.shape.flatten()), action)] = 0
                q = 0
                if q > max_q:
                    max_q = q
                    best_action = action

        return best_action
    
def count_available_spaces(grid):
    """Counts the number of empty cells on the grid."""
    return np.sum(grid == 0)

def calculate_reward(grid_before, grid_after, score_before, score_after):
    """Calculates the reward based on score change and available spaces."""
    score_reward = (score_after - score_before) * 10  # Reward for clearing lines

    return score_reward

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQLAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = state_size
        self.action_size = action_size
        self.max_piece_size = 25  # Maximum size for piece state padding
        
        # Main network and target network
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer(10000)
        
        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.target_update = 10  # Update target network every N episodes
    
    def pad_piece_state(self, piece_state):
        # Pad or truncate piece state to fixed size
        padded_state = np.zeros(self.max_piece_size)
        piece_state_flat = piece_state.flatten()
        padded_state[:len(piece_state_flat)] = piece_state_flat
        return padded_state
    
    def get_state_tensor(self, state, piece):
        # Combine board state and padded piece information
        board_state = np.array(state).flatten()
        piece_state = self.pad_piece_state(np.array(piece.shape))
        combined_state = np.concatenate([board_state, piece_state])
        return torch.FloatTensor(combined_state).unsqueeze(0).to(self.device)
    
    def choose_action(self, state_tensor, piece, grid):
        possible_actions = get_possible_actions(piece, grid)
        if not possible_actions:
            return None
            
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
            
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            # Convert actions to indices and mask impossible actions
            action_mask = torch.zeros(self.action_size, device=self.device)
            for action in possible_actions:
                idx = action[0] * GRID_WIDTH + action[1]
                action_mask[idx] = 1
            q_values = q_values * action_mask
            
            action_idx = q_values.max(1)[1].item()
            action = (action_idx // GRID_WIDTH, action_idx % GRID_WIDTH)
            return action if action in possible_actions else random.choice(possible_actions)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.cat(batch[0])
        action_batch = torch.tensor([[a] for a in batch[1]], device=self.device)
        reward_batch = torch.tensor(batch[2], device=self.device)
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.tensor(batch[4], device=self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        next_q_values[done_batch] = 0.0
        
        # Compute target Q values
        target_q_values = reward_batch + (self.gamma * next_q_values)
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_dqn():
    # Initialize environment and agent
    grid = Board()
    state_size = GRID_WIDTH * GRID_HEIGHT + 25  # Board state + max piece size
    action_size = GRID_WIDTH * GRID_HEIGHT  # All possible positions
    agent = DQLAgent(state_size, action_size)
    
    num_episodes = 1000
    scores = deque(maxlen=100)
    
    for episode in range(num_episodes):
        score = 0
        grid.reset()
        grid.board = np.array(grid.board)
        piece = choose_piece(random.randint(1, 6))
        piece.shape = np.array(piece.shape)
        state_tensor = agent.get_state_tensor(grid.board, piece)
        
        while True:
            # Choose and perform action
            action = agent.choose_action(state_tensor, piece, grid)
            if action is None:
                break
                
            # Place piece and get reward
            piece.piece_y, piece.piece_x = action
            grid_before = np.copy(grid.board)
            score_before = score
            
            piece.place_piece(grid)
            score += grid.check_rows()
            score += grid.check_columns()
            
            # Get next state and reward
            next_piece = choose_piece(random.randint(1, 6))
            next_piece.shape = np.array(next_piece.shape)
            next_state_tensor = agent.get_state_tensor(grid.board, next_piece)
            
            reward = calculate_reward(grid_before, grid.board, score_before, score)
            done = grid.is_game_over([next_piece])
            
            # Store transition and train
            action_idx = action[0] * GRID_WIDTH + action[1]
            agent.memory.push(state_tensor, action_idx, reward, next_state_tensor, done)
            agent.train()
            
            state_tensor = next_state_tensor
            piece = next_piece
            
            if done:
                break
        
        scores.append(score)
        if episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        if episode % 10 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return agent

def play_game_dqn(agent):
    game = Game()
    clock = pygame.time.Clock()
    game_over = False
    grid = Board()
    score = 0
    
    piece = choose_piece(random.randint(1, 6))
    piece.shape = np.array(piece.shape)
    
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
        
        state_tensor = agent.get_state_tensor(grid.board, piece)
        action = agent.choose_action(state_tensor, piece, grid)
        
        if action is None:
            game_over = grid.is_game_over([piece])
            if game_over:
                break
            piece = choose_piece(random.randint(1, 6))
            piece.shape = np.array(piece.shape)
            continue
        
        piece.piece_y, piece.piece_x = action
        piece.place_piece(grid)
        score += grid.check_rows()
        score += grid.check_columns()
        
        piece = choose_piece(random.randint(1, 6))
        piece.shape = np.array(piece.shape)
        
        # Draw game state
        grid.screen.fill(white)
        grid.draw_grid()
        grid.draw_pieces(colors)
        game.display_score(grid, score, black)
        pygame.display.update()
        
        clock.tick(3)
    
    print(f"Game Over! Final Score: {score}")
    pygame.quit()

# Train the DQN agent
trained_agent = train_dqn()

# Play the game using the trained agent
play_game_dqn(trained_agent)