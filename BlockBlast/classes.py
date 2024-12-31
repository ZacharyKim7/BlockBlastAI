import pygame

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Block Blast")
    
    def display_score(self, grid, score, color):
        font = pygame.font.Font(None, 36)
        text = font.render("Score: " + str(score), True, color)
        text_rect = text.get_rect()
        text_rect.centerx = grid.width // 2
        text_rect.top = 10
        grid.screen.blit(text, text_rect)

    def game_over(self, grid, color):
        font = pygame.font.Font(None, 48)
        text = font.render("Game Over", True, color)
        text_rect = text.get_rect()
        text_rect.center = (grid.width // 2, grid.height // 2)
        grid.screen.blit(text, text_rect)
        pygame.display.update()
        pygame.time.delay(2000)

pygame.quit()


class Board:
    def __init__(self, grid_size = 8):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.width = 400
        self.height = 400
        self.grid_size = grid_size
        self.cell_size = int(self.width / grid_size)
        self.screen = pygame.display.set_mode((self.width, self.height))
    
    def print_board(self):
        for row in self.board:
            for col in row:
                print(col, end = " ")
            print()
        print()

    def draw_grid(self, color=(0, 0, 0)):
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, color, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))

    def draw_pieces(self, colors):
        for x in range(len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y] != 0:
                    pygame.draw.rect(self.screen, colors[self.board[x][y]], (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))

    # Function to check for and remove full rows
    def check_rows(self):
        points = 0
        for y in range(self.grid_size - 1, -1, -1):
            if all(self.board[y][x] != 0 for x in range(self.grid_size)):
                for x in range(self.grid_size):
                    self.board[y][x] = 0
                points += 10
        return points

    # Function to check for and remove full columns
    def check_columns(self):
        points = 0
        for x in range(self.grid_size):
            if all(self.board[y][x] != 0 for y in range(self.grid_size)):
                for y in range(self.grid_size):
                    self.board[y][x] = 0
                points += 10
        return points

    # Function to check for game over
    def is_game_over(self, pieces):
        for piece in pieces:
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if piece.can_place_piece(piece, x, y, self):
                        return False
        print("current Piece: ", piece.name)
        return True


class Piece:
    def __init__(self, color_id):
        self.shape = []
        self.name = "Piece"
        self.color_id = color_id
        self.piece_x = 0
        self.piece_y = 0

    def draw_piece(self, color, current_piece, board):
        for x, y in current_piece.shape:
            pygame.draw.rect(board.screen, color, ((self.piece_x + x) * board.cell_size, (self.piece_y + y) * board.cell_size, board.cell_size, board.cell_size))

    # Function to check if a piece can be placed
    def can_place_piece(self, piece, piece_x, piece_y, board):
        for x, y in piece.shape:
            if piece_x + x < 0 or piece_x + x >= board.grid_size or piece_y + y < 0 or piece_y + y >= board.grid_size:
                return False
            if board.board[piece_y + y][piece_x + x] != 0:
                return False
        return True

    # Function to place the current piece on the grid
    def place_piece(self, board):
        for x, y in self.shape:
            board.board[self.piece_y + y][self.piece_x + x] = self.color_id  # Random color for each block

class Dot(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0)]
        self.name = "Dot"

class Square(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.name = "Square"

class LargeSquare(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        self.name = "Large Square"

class VerticalLine(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (1, 0), (2, 0)]
        self.name = "Vertical Line"

class TwoHorizontal(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (0, 1)]
        self.name = "Two Horizontal"

class ThreeHorizontal(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (0, 1), (0, 2)]
        self.name = "Three Horizontal"

class FourHorizontal(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (0, 1), (0, 2), (0, 3)]
        self.name = "Four Horizontal"

class FiveHorizontal(Piece):    
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        self.name = "Five Horizontal"

class TwoVertical(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (1, 0)]
        self.name = "Two Vertical"

class ThreeVertical(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (1, 0), (2, 0)]
        self.name = "Three Vertical"

class FourVertical(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self.name = "Four Vertical"

class FiveVertical(Piece):
    def __init__(self, color_id):
        super().__init__(color_id)
        self.shape = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        self.name = "Five Vertical"

# class LeftCactus(Piece):
#     def __init__(self, color_id):
#         super().__init__(color_id)
#         self.shape = [(0, 0), (1, 0), (1, 1), (2, 1)]

# class RightCactus(Piece):
#     def __init__(self, color_id):
#         super().__init__(color_id)
#         self.shape = [(0, 0), (1, -1), (1, 0), (2, 0)]

# dot
# Large Square
# 5 horizontal line
# 4 horizontal line
# 3 horizontal line
# 5 vertical line
# 2 vertical line
# 3 vertical line

# left cactus
# right cactus
# S
# Z 
# J
# left J
# upside down J
# 
# rotated L
# L
# left L
# T
# upsidedown T
# left T
# 
# br corner
# tl corner
# tr corner
# small top left corner
# vertical rectangle
# horizontal rectangle