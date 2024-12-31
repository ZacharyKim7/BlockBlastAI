from classes import *
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
    selection = random.randint(0, 8)

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
    pygame.time.delay(100)  # Add a delay to control the game speed

# Game over
game.game_over(grid, black)
