from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from gomoku.board import Move
from gomoku.game import Game


class BaseAgent(ABC):
    """
    Base class for all Gomoku AI agents.

    Every agent must implement select_move().
    """

    name: str = "base_agent"

    @abstractmethod
    def select_move(self, game: Game) -> Move:
        """
        Select a move for the current game state.

        Args:
            game: Current game object.

        Returns:
            A legal move as (row, col).
        """
        raise NotImplementedError


@dataclass
class RandomAgent(BaseAgent):
    """
    An agent that selects a random legal move.

    This is mainly used for testing the game loop and experiment pipeline.
    """

    seed: Optional[int] = None
    name: str = "random"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_move(self, game: Game) -> Move:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        return self.rng.choice(legal_moves)