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

    Different rules such as Standard, Pro, and Swap2 will inherit from this.
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

        Standard rule does not need this, but Pro / Swap2 may use it later.
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


def create_rule(rule_name: str) -> BaseRule:
    """
    Create a rule object by name.
    """
    if rule_name == "standard":
        return StandardRule()

    raise ValueError(f"Unsupported rule: {rule_name}")