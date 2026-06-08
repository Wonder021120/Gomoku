from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from gomoku.agents import BaseAgent
from gomoku.board import Board, Move
from gomoku.game import Game, GameStatus


@dataclass
class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree.
    """

    game: Game
    parent: Optional["MCTSNode"] = None
    move: Optional[Move] = None
    children: list["MCTSNode"] = field(default_factory=list)
    untried_moves: list[Move] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0

    def __post_init__(self) -> None:
        if not self.untried_moves:
            self.untried_moves = self.game.get_legal_moves()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.game.is_over()


@dataclass
class MCTSAgent(BaseAgent):
    """
    A basic Monte Carlo Tree Search agent.

    This is the second major AI method in the project after Minimax.
    """

    simulations: int = 100
    exploration_weight: float = 1.4
    candidate_radius: int = 2
    seed: Optional[int] = None
    name: str = "mcts"

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("Number of simulations must be positive.")
        if self.exploration_weight <= 0:
            raise ValueError("Exploration weight must be positive.")
        if self.candidate_radius <= 0:
            raise ValueError("Candidate radius must be positive.")
        self.rng = random.Random(self.seed)

    def select_move(self, game: Game) -> Move:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        if len(game.board.move_history) == 0:
            centre = game.board.size // 2
            return (centre, centre)

        root_player = game.current_player
        root = MCTSNode(game=game.copy())
        root.untried_moves = self._get_candidate_moves(root.game)

        for _ in range(self.simulations):
            node = self._select(root)
            simulation_result = self._simulate(node.game.copy(), root_player)
            self._backpropagate(node, simulation_result)

        if not root.children:
            return self.rng.choice(legal_moves)

        best_child = max(root.children, key=lambda child: child.visits)

        if best_child.move is None:
            return self.rng.choice(legal_moves)

        return best_child.move

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection and expansion phase.
        """
        current = node

        while not current.is_terminal():
            if current.untried_moves:
                return self._expand(current)

            current = self._best_child(current)

        return current

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand one untried move.
        """
        move = self.rng.choice(node.untried_moves)
        node.untried_moves.remove(move)

        next_game = node.game.copy()
        next_game.play_move(move)

        child = MCTSNode(
            game=next_game,
            parent=node,
            move=move,
        )
        child.untried_moves = self._get_candidate_moves(child.game)

        node.children.append(child)
        return child

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select child with highest UCB1 score.
        """
        best_score = -math.inf
        best_children: list[MCTSNode] = []

        for child in node.children:
            if child.visits == 0:
                score = math.inf
            else:
                exploitation = child.wins / child.visits
                exploration = self.exploration_weight * math.sqrt(
                    math.log(node.visits) / child.visits
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return self.rng.choice(best_children)

    def _simulate(self, game: Game, root_player: int) -> float:
        """
        Run a rollout from the given game state.

        Returns:
            1.0 if root_player wins
            0.5 if draw
            0.0 if root_player loses
        """
        while not game.is_over():
            move = self._select_rollout_move(game)
            game.play_move(move)

        winner = game.get_winner()

        if winner == root_player:
            return 1.0

        if winner is None:
            return 0.5

        return 0.0

    def _select_rollout_move(self, game: Game) -> Move:
        """
        Lightweight rollout policy.

        Priority:
        1. Win immediately if possible.
        2. Block opponent's immediate win if possible.
        3. Otherwise choose a random candidate move.
        """
        current_player = game.current_player
        opponent = Board.WHITE if current_player == Board.BLACK else Board.BLACK

        winning_move = self._find_immediate_winning_move(game, current_player)
        if winning_move is not None:
            return winning_move

        blocking_move = self._find_immediate_winning_move(game, opponent)
        if blocking_move is not None:
            return blocking_move

        candidate_moves = self._get_candidate_moves(game)

        if candidate_moves:
            return self.rng.choice(candidate_moves)

        legal_moves = game.get_legal_moves()
        return self.rng.choice(legal_moves)

    def _find_immediate_winning_move(self, game: Game, player: int) -> Optional[Move]:
        """
        Return a move that lets player win immediately, if one exists.
        """
        for move in self._get_candidate_moves(game):
            simulated_game = game.copy()
            row, col = move
            simulated_game.board.place_stone(row, col, player)

            from gomoku.win_checker import is_winning_move

            if is_winning_move(simulated_game.board, move):
                return move

        return None

    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """
        Backpropagate rollout result up the tree.
        """
        current: Optional[MCTSNode] = node

        while current is not None:
            current.visits += 1
            current.wins += result
            current = current.parent

    def _get_candidate_moves(self, game: Game) -> list[Move]:
        """
        Generate candidate moves near existing stones.
        """
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return []

        if len(game.board.move_history) == 0:
            centre = game.board.size // 2
            return [(centre, centre)]

        candidates: set[Move] = set()
        size = game.board.size

        for stone_position, _player in game.board.move_history:
            row, col = stone_position

            for delta_row in range(-self.candidate_radius, self.candidate_radius + 1):
                for delta_col in range(-self.candidate_radius, self.candidate_radius + 1):
                    candidate_row = row + delta_row
                    candidate_col = col + delta_col

                    if (
                        0 <= candidate_row < size
                        and 0 <= candidate_col < size
                        and game.board.is_empty(candidate_row, candidate_col)
                    ):
                        candidates.add((candidate_row, candidate_col))

        if not candidates:
            return legal_moves

        centre = size // 2

        return sorted(
            candidates,
            key=lambda move: abs(move[0] - centre) + abs(move[1] - centre),
        )