class Board:
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
    
    def print_board(self):
        for row in self.board:
            for col in row:
                print(col, end = " ")
            print()

    def is_valid_placement(self, piece, row, col):
        for r, c in piece.shape:
            if not (0 <= row + r < 8 and 0 <= col + c < 8) or self.board[row + r][col + c] != 0:
                return False
        return True

    def place_piece(self, piece, row, col):
        if self.is_valid_placement(piece, row, col):
            for r, c in piece.shape:
                self.board[row + r][col + c] = 1
            return True
        return False

from abc import ABC, abstractmethod

class Piece(ABC):
    def __init__(self):
        self.shape = []  # List of relative coordinates for the piece's blocks

    @abstractmethod
    def rotate(self):
        """Rotate the piece (optional, depending on your game rules)."""
        pass

class Square(Piece):
    def __init__(self):
        super().__init__()
        self.shape = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def rotate(self):
        # No rotation needed for square piece
        pass

class VerticalLine(Piece):
    def __init__(self):
        super().__init__()
        self.shape = [(0, 0), (1, 0), (2, 0)]

    def rotate(self):
        # Example of rotating a vertical line
        self.shape = [(c, -r) for r, c in self.shape]
