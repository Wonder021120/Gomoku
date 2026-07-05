from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from gomoku.agents import BaseAgent
from gomoku.board import Board, Move
from gomoku.game import Game
from gomoku.minimax_agent import MinimaxAgent
from gomoku.win_checker import is_winning_move


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

    def is_terminal(self) -> bool:
        return self.game.is_over()


@dataclass
class MCTSAgent(BaseAgent):
    """
    A Monte Carlo Tree Search agent.

    This implementation uses:
    - root-level tactical safety
    - UCB1 selection
    - candidate move generation
    - heuristic rollout policy
    - rollout depth limit
    - pattern-based evaluation at rollout cutoff

    Root-level tactical safety:
    Before running MCTS simulations, the agent checks:
    1. whether the current player can win immediately;
    2. whether the opponent can win immediately and must be blocked.
    """

    simulations: int = 50
    exploration_weight: float = 1.4
    candidate_radius: int = 1
    rollout_depth_limit: int = 30
    seed: Optional[int] = None
    name: str = "mcts"

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("Number of simulations must be positive.")
        if self.exploration_weight <= 0:
            raise ValueError("Exploration weight must be positive.")
        if self.candidate_radius <= 0:
            raise ValueError("Candidate radius must be positive.")
        if self.rollout_depth_limit <= 0:
            raise ValueError("Rollout depth limit must be positive.")

        self.rng = random.Random(self.seed)

        # Reuse the Minimax pattern evaluator for non-terminal rollout cutoffs.
        self.evaluator = MinimaxAgent(
            depth=1,
            candidate_radius=self.candidate_radius,
            seed=self.seed,
        )

    def select_move(self, game: Game) -> Move:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        if len(game.board.move_history) == 0:
            centre = game.board.size // 2
            centre_move = (centre, centre)

            if centre_move in legal_moves:
                return centre_move

            return self.rng.choice(legal_moves)

        # Root-level tactical safety.
        # This makes MCTS directly take immediate wins and immediate blocks
        # instead of relying on rollout statistics to discover them.
        current_player = game.current_player
        opponent = self._opponent(current_player)

        winning_move = self._find_immediate_winning_move(game, current_player)
        if winning_move is not None:
            return winning_move

        blocking_move = self._find_immediate_winning_move(game, opponent)
        if blocking_move is not None:
            return blocking_move

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
                    math.log(max(node.visits, 1)) / child.visits
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
            1.0 if root_player is estimated to win
            0.5 if draw or balanced
            0.0 if root_player is estimated to lose
        """
        rollout_steps = 0

        while not game.is_over() and rollout_steps < self.rollout_depth_limit:
            move = self._select_rollout_move(game)
            game.play_move(move)
            rollout_steps += 1

        winner = game.get_winner()

        if winner == root_player:
            return 1.0

        if winner == self._opponent(root_player):
            return 0.0

        if game.is_over():
            return 0.5

        return self._evaluate_cutoff_position(game, root_player)

    def _evaluate_cutoff_position(self, game: Game, root_player: int) -> float:
        """
        Estimate a non-terminal rollout cutoff position.

        Uses the Minimax pattern evaluator and maps the score to:
        - > 100 means favourable for root_player
        - < -100 means unfavourable for root_player
        - otherwise balanced
        """
        score = self.evaluator._evaluate_board(game.board, root_player)

        if score > 100:
            return 1.0

        if score < -100:
            return 0.0

        return 0.5

    def _select_rollout_move(self, game: Game) -> Move:
        """
        Lightweight rollout policy.

        Priority:
        1. Win immediately if possible.
        2. Block opponent's immediate win if possible.
        3. Otherwise choose a random candidate move.
        """
        current_player = game.current_player
        opponent = self._opponent(current_player)

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

        The search uses all legal moves, not only candidate moves, so an
        immediate win/block cannot be lost because of candidate pruning.
        """
        for move in game.get_legal_moves():
            simulated_board = game.board.copy()
            row, col = move
            simulated_board.place_stone(row, col, player)

            if is_winning_move(simulated_board, move):
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
            centre_move = (centre, centre)

            if centre_move in legal_moves:
                return [centre_move]

            return legal_moves

        candidates: set[Move] = set()
        legal_set = set(legal_moves)
        size = game.board.size

        for stone_position, _player in game.board.move_history:
            row, col = stone_position

            for delta_row in range(-self.candidate_radius, self.candidate_radius + 1):
                for delta_col in range(-self.candidate_radius, self.candidate_radius + 1):
                    candidate_row = row + delta_row
                    candidate_col = col + delta_col
                    candidate_move = (candidate_row, candidate_col)

                    if candidate_move in legal_set:
                        candidates.add(candidate_move)

        if not candidates:
            return legal_moves

        centre = size // 2

        return sorted(
            candidates,
            key=lambda move: abs(move[0] - centre) + abs(move[1] - centre),
        )

    def _opponent(self, player: int) -> int:
        return Board.WHITE if player == Board.BLACK else Board.BLACK