from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from gomoku.board import Board, Move
from gomoku.win_checker import get_winner


class GameStatus(Enum):
    """
    Possible states of a Gomoku game.
    """

    ONGOING = "ongoing"
    BLACK_WIN = "black_win"
    WHITE_WIN = "white_win"
    DRAW = "draw"


@dataclass
class Game:
    """
    Controls the flow of a Gomoku game.
    """

    board_size: int = 15

    def __post_init__(self) -> None:
        self.board = Board(size=self.board_size)
        self.current_player = Board.BLACK
        self.last_move: Optional[Move] = None
        self.status = GameStatus.ONGOING

    def reset(self) -> None:
        """Reset the game to the initial state."""
        self.board.reset()
        self.current_player = Board.BLACK
        self.last_move = None
        self.status = GameStatus.ONGOING

    def is_over(self) -> bool:
        """Return True if the game has ended."""
        return self.status != GameStatus.ONGOING

    def switch_player(self) -> None:
        """Switch the current player."""
        self.current_player = Board.WHITE if self.current_player == Board.BLACK else Board.BLACK

    def play_move(self, move: Move) -> GameStatus:
        """
        Play a move for the current player.

        Args:
            move: A tuple (row, col).

        Returns:
            The current game status after the move.
        """
        if self.is_over():
            raise ValueError("Cannot play a move after the game is over.")

        row, col = move
        self.board.place_stone(row, col, self.current_player)
        self.last_move = move

        winner = get_winner(self.board, self.last_move)

        if winner == Board.BLACK:
            self.status = GameStatus.BLACK_WIN
            return self.status

        if winner == Board.WHITE:
            self.status = GameStatus.WHITE_WIN
            return self.status

        if self.board.is_full():
            self.status = GameStatus.DRAW
            return self.status

        self.switch_player()
        return self.status

    def get_legal_moves(self) -> list[Move]:
        """Return all legal moves for the current position."""
        if self.is_over():
            return []
        return self.board.get_legal_moves()

    def get_winner(self) -> Optional[int]:
        """
        Return the winning player.

        Returns:
            Board.BLACK, Board.WHITE, or None.
        """
        if self.status == GameStatus.BLACK_WIN:
            return Board.BLACK
        if self.status == GameStatus.WHITE_WIN:
            return Board.WHITE
        return None

    def copy(self) -> "Game":
        """Return a copy of the game state."""
        new_game = Game(board_size=self.board_size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.last_move = self.last_move
        new_game.status = self.status
        return new_game