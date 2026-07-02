"""
Neural-network-guided Monte Carlo Tree Search agent for Gomoku.

This agent uses a policy-value neural network to guide MCTS.

Compared with a simple one-move NN agent, NN-MCTS works as follows:

1. The neural network gives prior probabilities for candidate moves.
2. MCTS uses these priors during tree search.
3. The neural network value head evaluates leaf positions.
4. The final move is selected from MCTS visit counts.

This file also provides select_move_with_policy(), which returns both:
- the selected move
- the MCTS visit-count policy distribution over the full board

This is needed for AlphaZero-style self-play training.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from gomoku.board import Board
from gomoku.game import Game, GameStatus
from gomoku.neural_network import (
    GomokuPolicyValueNet,
    encode_board_state,
    move_to_action,
)


Move = Tuple[int, int]


@dataclass
class NNMCTSNode:
    """A node in the NN-guided MCTS search tree."""

    game: Game
    parent: Optional["NNMCTSNode"] = None
    move: Optional[Move] = None
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[Move, "NNMCTSNode"] = field(default_factory=dict)
    is_expanded: bool = False

    @property
    def mean_value(self) -> float:
        """Average value from the perspective of the player to move at this node."""

        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits

    def select_child(self, exploration_weight: float) -> "NNMCTSNode":
        """
        Select a child according to a PUCT-style score.

        Child mean values are stored from the child player's perspective.
        Since the child player is the opponent of the current node player,
        the value is negated when viewed from the current node.
        """

        if not self.children:
            raise ValueError("Cannot select a child from an unexpanded leaf node.")

        best_score = -float("inf")
        best_child: Optional[NNMCTSNode] = None

        parent_visit_term = math.sqrt(max(1, self.visits))

        for child in self.children.values():
            q_value = -child.mean_value
            exploration = (
                exploration_weight
                * child.prior
                * parent_visit_term
                / (1 + child.visits)
            )
            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("MCTS child selection failed.")

        return best_child


class NNMCTSAgent:
    """
    Neural-network-guided MCTS agent.

    Args:
        board_size:
            Gomoku board size.
        simulations:
            Number of MCTS simulations per move.
        checkpoint_path:
            Optional path to a trained policy-value network checkpoint.
        exploration_weight:
            PUCT exploration constant.
        candidate_radius:
            Only consider legal moves near existing stones within this radius.
        device:
            "auto", "cpu", or "cuda".
    """

    def __init__(
        self,
        board_size: int = 15,
        simulations: int = 25,
        checkpoint_path: Optional[str] = None,
        exploration_weight: float = 1.5,
        candidate_radius: int = 2,
        device: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        if board_size <= 0:
            raise ValueError("board_size must be positive.")

        if simulations <= 0:
            raise ValueError("simulations must be positive.")

        if candidate_radius <= 0:
            raise ValueError("candidate_radius must be positive.")

        self.board_size = board_size
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        self.candidate_radius = candidate_radius
        self.seed = seed
        self.rng = random.Random(seed)
        self.device = self._resolve_device(device)

        self.model = GomokuPolicyValueNet(board_size=board_size)
        self.model.to(self.device)
        self.model.eval()

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string."""

        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")

        return torch.device(device)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a policy-value network checkpoint."""

        path = Path(checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        checkpoint_board_size = checkpoint.get("board_size")

        if checkpoint_board_size is not None and checkpoint_board_size != self.board_size:
            raise ValueError(
                f"Checkpoint board size {checkpoint_board_size} does not match "
                f"agent board size {self.board_size}."
            )

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def select_move(self, game: Game) -> Move:
        """
        Select a move.

        This keeps the old public interface used by run_tournament.py.
        """

        move, _policy = self.select_move_with_policy(
            game=game,
            temperature=0.0,
            rng=self.rng,
        )
        return move

    def select_move_with_policy(
        self,
        game: Game,
        temperature: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> Tuple[Move, np.ndarray]:
        """
        Select a move and return the MCTS visit-count policy distribution.

        Args:
            game:
                Current game state.
            temperature:
                If 0, choose the most visited move.
                If > 0, sample from visit counts adjusted by temperature.
                This is useful for AlphaZero-style opening exploration.
            rng:
                Optional random generator used for temperature sampling.

        Returns:
            selected_move:
                The move chosen by NN-MCTS.
            policy_distribution:
                A numpy array of shape [board_size * board_size].
                It sums to 1 over legal searched moves.
        """

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        # Keep the established Gomoku opening convention:
        # on an empty board, play the centre directly.
        if not game.board.move_history:
            centre = game.board.size // 2
            centre_move = (centre, centre)

            if centre_move in legal_moves:
                policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)
                policy[move_to_action(centre_move, self.board_size)] = 1.0
                return centre_move, policy

        root = NNMCTSNode(game=game.copy())

        # Expand root before simulations so that child priors are available.
        self._expand_and_evaluate(root)

        if not root.children:
            move = legal_moves[0]
            policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)
            policy[move_to_action(move, self.board_size)] = 1.0
            return move, policy

        for _ in range(self.simulations):
            node = root

            # Selection
            while node.is_expanded and node.children:
                node = node.select_child(self.exploration_weight)

            # Evaluation / expansion
            value = self._evaluate_or_expand(node)

            # Backpropagation
            self._backpropagate(node, value)

        policy_distribution = self._build_visit_distribution(
            root=root,
            temperature=temperature,
        )

        selected_move = self._select_move_from_distribution(
            policy_distribution=policy_distribution,
            legal_moves=list(root.children.keys()),
            temperature=temperature,
            rng=rng,
        )

        return selected_move, policy_distribution

    def _evaluate_or_expand(self, node: NNMCTSNode) -> float:
        """Evaluate a terminal node or expand a non-terminal leaf."""

        if node.game.status != GameStatus.ONGOING:
            return self._terminal_value(node.game)

        return self._expand_and_evaluate(node)

    def _expand_and_evaluate(self, node: NNMCTSNode) -> float:
        """
        Expand a leaf node and return the neural network value.

        The returned value is from the perspective of the player to move
        at this node.
        """

        if node.game.status != GameStatus.ONGOING:
            return self._terminal_value(node.game)

        candidate_moves = self._get_candidate_moves(node.game)

        if not candidate_moves:
            return 0.0

        priors, value = self._predict_policy_and_value(
            game=node.game,
            candidate_moves=candidate_moves,
        )

        for move, prior in zip(candidate_moves, priors):
            child_game = node.game.copy()
            child_game.play_move(move)

            node.children[move] = NNMCTSNode(
                game=child_game,
                parent=node,
                move=move,
                prior=float(prior),
            )

        node.is_expanded = True

        return value

    def _predict_policy_and_value(
        self,
        game: Game,
        candidate_moves: List[Move],
    ) -> Tuple[np.ndarray, float]:
        """
        Use the neural network to produce priors and value.

        Priors are normalized only over candidate moves.
        """

        state = encode_board_state(game.board, game.current_player)
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            output = self.model(state_tensor)

        if hasattr(output, "policy_logits"):
            policy_logits = output.policy_logits
            value_tensor = output.value
        elif isinstance(output, tuple) and len(output) == 2:
            policy_logits, value_tensor = output
        else:
            raise TypeError("Unexpected neural network output format.")

        logits = policy_logits.squeeze(0).detach().cpu().numpy()

        actions = [move_to_action(move, self.board_size) for move in candidate_moves]
        candidate_logits = logits[actions]

        # Numerical stability.
        candidate_logits = candidate_logits - np.max(candidate_logits)
        exp_logits = np.exp(candidate_logits)

        if not np.isfinite(exp_logits).all() or exp_logits.sum() <= 0:
            priors = np.ones(len(candidate_moves), dtype=np.float32) / len(candidate_moves)
        else:
            priors = exp_logits / exp_logits.sum()

        value = float(value_tensor.squeeze().detach().cpu().item())

        # Clamp for safety.
        value = max(-1.0, min(1.0, value))

        return priors.astype(np.float32), value

    def _get_candidate_moves(self, game: Game) -> List[Move]:
        """
        Return candidate legal moves near existing stones.

        This keeps NN-MCTS computationally manageable on a 15x15 board.
        """

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return []

        if not game.board.move_history:
            centre = game.board.size // 2
            centre_move = (centre, centre)

            if centre_move in legal_moves:
                return [centre_move]

            return legal_moves

        legal_set = set(legal_moves)
        candidate_set = set()

        for (row, col), _player in game.board.move_history:
            for candidate_row in range(row - self.candidate_radius, row + self.candidate_radius + 1):
                for candidate_col in range(col - self.candidate_radius, col + self.candidate_radius + 1):
                    candidate_move = (candidate_row, candidate_col)

                    if candidate_move in legal_set:
                        candidate_set.add(candidate_move)

        if not candidate_set:
            return legal_moves

        return sorted(candidate_set)

    def _terminal_value(self, game: Game) -> float:
        """
        Return terminal value from the perspective of game.current_player.

        +1 means the current player has won.
        -1 means the current player has lost.
         0 means draw.
        """

        if game.status == GameStatus.DRAW:
            return 0.0

        if game.status == GameStatus.BLACK_WIN:
            winner = Board.BLACK
        elif game.status == GameStatus.WHITE_WIN:
            winner = Board.WHITE
        else:
            return 0.0

        return 1.0 if game.current_player == winner else -1.0

    def _backpropagate(self, node: NNMCTSNode, value: float) -> None:
        """
        Backpropagate value up the tree.

        The value is always from the perspective of the current node's player.
        Moving to the parent changes perspective, so the sign is flipped.
        """

        current_node: Optional[NNMCTSNode] = node
        current_value = value

        while current_node is not None:
            current_node.visits += 1
            current_node.value_sum += current_value

            current_value = -current_value
            current_node = current_node.parent

    def _build_visit_distribution(
        self,
        root: NNMCTSNode,
        temperature: float,
    ) -> np.ndarray:
        """
        Build a full-board policy distribution from root child visit counts.

        If temperature is 0, the most visited move receives probability 1.
        If temperature > 0, visit counts are transformed by 1 / temperature.
        """

        policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)

        if not root.children:
            return policy

        moves = list(root.children.keys())
        visits = np.array(
            [root.children[move].visits for move in moves],
            dtype=np.float64,
        )

        if visits.sum() <= 0:
            visits = np.ones_like(visits)

        if temperature <= 0.0:
            best_index = int(np.argmax(visits))
            best_move = moves[best_index]
            policy[move_to_action(best_move, self.board_size)] = 1.0
            return policy

        adjusted = np.power(visits, 1.0 / temperature)

        if not np.isfinite(adjusted).all() or adjusted.sum() <= 0:
            adjusted = np.ones_like(adjusted)

        probabilities = adjusted / adjusted.sum()

        for move, probability in zip(moves, probabilities):
            policy[move_to_action(move, self.board_size)] = float(probability)

        return policy

    def _select_move_from_distribution(
        self,
        policy_distribution: np.ndarray,
        legal_moves: List[Move],
        temperature: float,
        rng: Optional[random.Random],
    ) -> Move:
        """Select a legal move from a full-board policy distribution."""

        if not legal_moves:
            raise ValueError("No legal moves available.")

        if temperature <= 0.0:
            best_move = max(
                legal_moves,
                key=lambda move: policy_distribution[move_to_action(move, self.board_size)],
            )
            return best_move

        probabilities = np.array(
            [policy_distribution[move_to_action(move, self.board_size)] for move in legal_moves],
            dtype=np.float64,
        )

        if probabilities.sum() <= 0 or not np.isfinite(probabilities).all():
            probabilities = np.ones(len(legal_moves), dtype=np.float64) / len(legal_moves)
        else:
            probabilities = probabilities / probabilities.sum()

        random_generator = rng or self.rng
        threshold = random_generator.random()

        cumulative = 0.0
        for move, probability in zip(legal_moves, probabilities):
            cumulative += probability

            if threshold <= cumulative:
                return move

        return legal_moves[-1]