from classes import *
from collections import deque
import numpy as np
import pygame
import random

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

def game_loop():
    game = Game()

    grid = Board()
    current_piece = None
    game_over = False
    score = 0

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.piece_x -= 1

                elif event.key == pygame.K_RIGHT:
                    current_piece.piece_x += 1
                
                elif event.key == pygame.K_UP:
                    current_piece.piece_y -= 1

                elif event.key == pygame.K_DOWN:
                    current_piece.piece_y += 1

                elif event.key == pygame.K_RETURN:  # Place the piece
                    if (0 <= current_piece.piece_x < grid.width and
                        0 <= current_piece.piece_y < grid.height and
                        current_piece.can_place_piece(current_piece, current_piece.piece_x, current_piece.piece_y, grid)):
                        current_piece.place_piece(grid)
                        score += grid.check_rows()
                        score += grid.check_columns()

                        grid.print_board()

                        current_piece = None

        if current_piece is None:
            color_id = random.randint(1, 6)
            current_piece = choose_piece(color_id)
            game_over = grid.is_game_over([current_piece])


        # Draw everything
        grid.screen.fill(white)
        grid.draw_grid()
        grid.draw_pieces(colors)
        current_piece.draw_piece(colors[color_id], current_piece, grid)

        # Display the score
        game.display_score(grid, score, black)

        pygame.display.update()

    # Game over
    game.game_over(grid, black)

# game_loop()

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

# def calculate_reward(score_before, score_after):
#     return (score_after - score_before) * 10 #reward 10 points per row/col cleared

# Main training loop
num_episodes = 500
scores = deque(maxlen=100)
score = 0
grid = Board()
for episode in range(num_episodes):
    score = 0
    grid.board = np.array(grid.board)
    piece = choose_piece(random.randint(1, 6))
    piece.shape = np.array(piece.shape)
    state = get_state(grid.board)

    while True:
        action = choose_action(state, piece, grid)
        if action is None: #no possible actions. game over
            break
        piece.piece_y, piece.piece_x = action
        grid_before = np.copy(grid.board) #create a copy of the grid before placing the piece
        score_before = score
        piece.place_piece(grid)

        score += grid.check_rows()
        score += grid.check_columns()

        # print(score)
        score_after = score
        grid_after = np.copy(grid.board)
        reward = calculate_reward(grid_before, grid_after, score_before, score_after)
        next_state = get_state(grid.board)
        next_piece = choose_piece(random.randint(1, 6))
        next_piece.shape = np.array(next_piece.shape)
        # Q-learning update
        if (state, tuple(piece.shape.flatten()), action) not in q_values:
            q_values[(state, tuple(piece.shape.flatten()), action)] = 0
        
        max_next_q = 0
        possible_next_actions = get_possible_actions(next_piece, grid)
        if possible_next_actions:
            for next_action in possible_next_actions:
                if (next_state, tuple(next_piece.shape.flatten()), next_action) in q_values:
                    q = q_values[(next_state,tuple(next_piece.shape.flatten()), next_action)]
                    max_next_q = max(max_next_q, q)
                else:
                    q_values[(next_state,tuple(next_piece.shape.flatten()), next_action)] = 0

        q_values[(state, tuple(piece.shape.flatten()), action)] += learning_rate * (
            reward + discount_factor * max_next_q - q_values[(state, tuple(piece.shape.flatten()), action)]
        )

        state = next_state
        piece = next_piece
    # print(score)
    scores.append(score)
    # if episode % 100 == 0:
    #     print(scores)
    #     average_score = np.mean(scores)
    #     print(f"Episode: {episode}, Average Score (Last 100): {average_score:.2f}")
    # epsilon = max(epsilon * 0.999, 0.01) #decay epsilon over time

print("Training finished!")
# game_loop() # You can uncomment this to play after training

def play_game(q_values):
    game = Game()
    global score
    clock = pygame.time.Clock()
    game_over = False
    grid.board = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    score = 0
    piece = choose_piece(random.randint(1, 6))
    piece.shape = np.array(piece.shape)
    # print(piece.shape)
    state = get_state(grid.board)
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        action = choose_action(state, piece, grid) #use the trained agent to choose actions
        if action is None:
            game_over = grid.is_game_over([piece])
            if game_over:
                break
            else:
                piece = choose_piece(random.randint(1, 6))
                piece.shape = np.array(piece.shape)
                continue

        piece.piece_y, piece.piece_x = action
        piece.place_piece(grid)
        score += grid.check_rows()
        score += grid.check_columns()
        state = get_state(grid.board)
        piece = choose_piece(random.randint(1, 6))
        piece.shape = np.array(piece.shape)

        grid.screen.fill(white)
        grid.draw_grid()
        grid.draw_pieces(colors)

        # Display the score
        game.display_score(grid, score, black)

        pygame.display.update()

        clock.tick(3) # Adjust speed as needed
    print(f"Game Over! Final Score: {score}")
    pygame.quit()

# Play the game using the trained Q-values
play_game(q_values)
# game_loop() # You can uncomment this to play after training