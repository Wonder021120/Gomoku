from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from gomoku.board import Board, Move
from gomoku.game import Game
from gomoku.win_checker import is_winning_move


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


@dataclass
class GreedyAgent(BaseAgent):
    """
    A simple rule-based agent.

    Strategy:
    1. Play a winning move if available.
    2. Block the opponent's immediate winning move if available.
    3. Otherwise choose a random legal move.
    """

    seed: Optional[int] = None
    name: str = "greedy"

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def select_move(self, game: Game) -> Move:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        current_player = game.current_player
        opponent = Board.WHITE if current_player == Board.BLACK else Board.BLACK

        winning_move = self._find_immediate_winning_move(game, current_player)
        if winning_move is not None:
            return winning_move

        blocking_move = self._find_immediate_winning_move(game, opponent)
        if blocking_move is not None:
            return blocking_move

        return self.rng.choice(legal_moves)

    def _find_immediate_winning_move(self, game: Game, player: int) -> Optional[Move]:
        """
        Return a move that lets the given player win immediately, if one exists.
        """
        for move in game.get_legal_moves():
            simulated_game = game.copy()
            row, col = move

            simulated_game.board.place_stone(row, col, player)

            if is_winning_move(simulated_game.board, move):
                return move

        return None