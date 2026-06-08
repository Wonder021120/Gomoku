from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from gomoku.board import Board, Move

if TYPE_CHECKING:
    from gomoku.game import Game


class BaseRule(ABC):
    """
    Base class for Gomoku opening and move rules.

    Different rules such as Standard, Pro, and Swap2 inherit from this.
    """

    name: str = "base"

    def get_initial_player(self) -> int:
        """
        Return the player who makes the first move.
        """
        return Board.BLACK

    @abstractmethod
    def validate_move(self, game: "Game", move: Move) -> None:
        """
        Validate whether a move is legal under this rule.

        Raises:
            ValueError: if the move is illegal.
        """
        raise NotImplementedError

    def after_move(self, game: "Game", move: Move) -> None:
        """
        Hook called after a legal move is played.

        Standard and Pro do not need this, but Swap2 may use it later.
        """
        return None

    def copy(self) -> "BaseRule":
        """
        Return a copy of the rule object.
        """
        return copy.deepcopy(self)


class StandardRule(BaseRule):
    """
    Standard Gomoku rule.

    Black moves first, players alternate turns, and any empty cell is legal.
    """

    name: str = "standard"

    def validate_move(self, game: "Game", move: Move) -> None:
        row, col = move

        if not game.board.is_inside(row, col):
            raise ValueError("Move is outside the board.")

        if not game.board.is_empty(row, col):
            raise ValueError("Cell is already occupied.")


class ProRule(StandardRule):
    """
    Pro opening rule.

    This implementation uses a common experimental interpretation:

    - Black's first move must be at the centre.
    - White's first move can be any legal empty cell.
    - Black's second move must be at least three intersections away
      from the centre, using Chebyshev distance.
    - After the first three moves, normal Standard Gomoku rules apply.
    """

    name: str = "pro"

    minimum_second_black_distance: int = 3

    def validate_move(self, game: "Game", move: Move) -> None:
        super().validate_move(game, move)

        move_count = len(game.board.move_history)
        centre = game.board.size // 2
        row, col = move

        # First move: black must play at the centre.
        if move_count == 0:
            if move != (centre, centre):
                raise ValueError("Under Pro rule, the first move must be at the centre.")
            return

        # Third move overall: black's second move must be far enough from centre.
        if move_count == 2:
            distance_from_centre = max(abs(row - centre), abs(col - centre))

            if distance_from_centre < self.minimum_second_black_distance:
                raise ValueError(
                    "Under Pro rule, black's second move must be at least "
                    f"{self.minimum_second_black_distance} intersections from the centre."
                )


def create_rule(rule_name: str) -> BaseRule:
    """
    Create a rule object by name.
    """
    if rule_name == "standard":
        return StandardRule()

    if rule_name == "pro":
        return ProRule()

    raise ValueError(f"Unsupported rule: {rule_name}")