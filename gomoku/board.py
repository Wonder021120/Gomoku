from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


Move = Tuple[int, int]


@dataclass
class Board:
    """
    Represents a Gomoku board.

    Cell values:
    - EMPTY = 0
    - BLACK = 1
    - WHITE = -1
    """

    size: int = 15

    EMPTY: int = 0
    BLACK: int = 1
    WHITE: int = -1

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("Board size must be positive.")
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.move_history: List[Tuple[Move, int]] = []

    def is_inside(self, row: int, col: int) -> bool:
        """Return True if the position is inside the board."""
        return 0 <= row < self.size and 0 <= col < self.size

    def is_empty(self, row: int, col: int) -> bool:
        """Return True if the position is inside the board and empty."""
        if not self.is_inside(row, col):
            return False
        return self.grid[row, col] == self.EMPTY

    def place_stone(self, row: int, col: int, player: int) -> None:
        """
        Place a stone for the given player.

        Raises:
            ValueError: if the player is invalid or the move is illegal.
        """
        if player not in (self.BLACK, self.WHITE):
            raise ValueError("Player must be Board.BLACK or Board.WHITE.")

        if not self.is_inside(row, col):
            raise ValueError("Move is outside the board.")

        if not self.is_empty(row, col):
            raise ValueError("Cell is already occupied.")

        self.grid[row, col] = player
        self.move_history.append(((row, col), player))

    def undo_move(self) -> None:
        """Undo the most recent move."""
        if not self.move_history:
            raise ValueError("No move to undo.")

        (row, col), _player = self.move_history.pop()
        self.grid[row, col] = self.EMPTY

    def get_legal_moves(self) -> List[Move]:
        """Return all empty positions on the board."""
        moves: List[Move] = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] == self.EMPTY:
                    moves.append((row, col))
        return moves

    def is_full(self) -> bool:
        """Return True if the board has no empty cells."""
        return not np.any(self.grid == self.EMPTY)

    def copy(self) -> "Board":
        """Return a deep copy of the board."""
        new_board = Board(size=self.size)
        new_board.grid = self.grid.copy()
        new_board.move_history = self.move_history.copy()
        return new_board

    def reset(self) -> None:
        """Clear the board."""
        self.grid.fill(self.EMPTY)
        self.move_history.clear()

    def __str__(self) -> str:
        """Return a simple text representation of the board."""
        symbols = {
            self.EMPTY: ".",
            self.BLACK: "X",
            self.WHITE: "O",
        }

        rows = []
        for row in range(self.size):
            rows.append(" ".join(symbols[self.grid[row, col]] for col in range(self.size)))

        return "\n".join(rows)