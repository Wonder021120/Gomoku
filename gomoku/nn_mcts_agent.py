from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gomoku.agents import BaseAgent
from gomoku.board import Board, Move
from gomoku.game import Game, GameStatus
from gomoku.neural_network import (
    GomokuPolicyValueNet,
    action_to_move,
    encode_board_state,
    legal_moves_mask,
    masked_softmax,
    move_to_action,
)


@dataclass
class NNMCTSNode:
    """
    A search-tree node for neural-network-guided MCTS.

    prior:
        Prior probability of selecting this move, provided by the policy network.

    visits:
        Number of times this node has been visited.

    value_sum:
        Sum of backed-up values from simulations.
    """

    game: Game
    parent: Optional["NNMCTSNode"] = None
    move: Move | None = None
    prior: float = 0.0
    children: dict[Move, "NNMCTSNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class NNMCTSAgent(BaseAgent):
    """
    Neural-network-guided MCTS agent.

    This agent uses a policy-value neural network to guide tree search:

    - policy head: gives prior probabilities for candidate moves
    - value head: evaluates leaf positions without random rollouts

    This is a lightweight AlphaZero-style MCTS implementation for this project.
    """

    def __init__(
        self,
        board_size: int = 15,
        simulations: int = 50,
        checkpoint_path: str | Path | None = None,
        exploration_weight: float = 1.5,
        candidate_radius: int | None = 2,
        device: str = "auto",
        seed: int | None = None,
        name: str = "nn_mcts",
    ) -> None:
        if simulations <= 0:
            raise ValueError("simulations must be positive.")

        if board_size <= 0:
            raise ValueError("board_size must be positive.")

        if exploration_weight < 0:
            raise ValueError("exploration_weight must be non-negative.")

        self.board_size = board_size
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        self.candidate_radius = candidate_radius
        self.seed = seed
        self.name = name

        self.random = random.Random(seed)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = GomokuPolicyValueNet(board_size=board_size).to(self.device)
        self.model.eval()

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load trained model parameters from a checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        checkpoint_board_size = checkpoint.get("board_size", self.board_size)

        if checkpoint_board_size != self.board_size:
            raise ValueError(
                f"Checkpoint board size {checkpoint_board_size} does not match "
                f"agent board size {self.board_size}."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def select_move(self, game: Game) -> Move:
        """
        Select a move using neural-network-guided MCTS.
        """
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        if len(game.board.move_history) == 0:
            centre = (game.board.size // 2, game.board.size // 2)
            if centre in legal_moves:
                return centre

        root = NNMCTSNode(game=game.copy())

        self._expand(root)

        for _ in range(self.simulations):
            node = root
            search_path = [node]

            while node.is_expanded and not node.game.is_over():
                node = self._select_child(node)
                search_path.append(node)

            if node.game.is_over():
                value = self._terminal_value(node.game)
            else:
                value = self._expand(node)

            self._backpropagate(search_path, value)

        if not root.children:
            return self.random.choice(legal_moves)

        best_child = max(
            root.children.values(),
            key=lambda child: (child.visits, child.mean_value),
        )

        if best_child.move is None:
            return self.random.choice(legal_moves)

        return best_child.move

    def _expand(self, node: NNMCTSNode) -> float:
        """
        Expand a leaf node using the neural network.

        Returns:
            The value estimate from the current player's perspective.
        """
        game = node.game

        if game.is_over():
            return self._terminal_value(game)

        legal_moves = self._get_candidate_legal_moves(game)

        if not legal_moves:
            return 0.0

        policy_probs, value = self._evaluate_game(game)

        priors = []

        for move in legal_moves:
            action = move_to_action(move, game.board.size)
            priors.append(float(policy_probs[action]))

        prior_sum = sum(priors)

        if prior_sum <= 0:
            normalised_priors = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            normalised_priors = [prior / prior_sum for prior in priors]

        for move, prior in zip(legal_moves, normalised_priors):
            child_game = game.copy()
            child_game.play_move(move)

            node.children[move] = NNMCTSNode(
                game=child_game,
                parent=node,
                move=move,
                prior=prior,
            )

        return value

    def _select_child(self, node: NNMCTSNode) -> NNMCTSNode:
        """
        Select a child using a PUCT-style score.

        score = Q + U

        Q:
            Mean value of the child.

        U:
            Exploration bonus based on neural-network prior probability.
        """
        if not node.children:
            raise ValueError("Cannot select child from an unexpanded node.")

        parent_visits = max(1, node.visits)

        def puct_score(child: NNMCTSNode) -> float:
            q_value = -child.mean_value
            exploration = (
                self.exploration_weight
                * child.prior
                * math.sqrt(parent_visits)
                / (1 + child.visits)
            )
            return q_value + exploration

        return max(node.children.values(), key=puct_score)

    def _backpropagate(self, search_path: list[NNMCTSNode], value: float) -> None:
        """
        Backpropagate value estimates through the search path.

        The value alternates sign at each level because players alternate turns.
        """
        for node in reversed(search_path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _evaluate_game(self, game: Game) -> tuple[np.ndarray, float]:
        """
        Evaluate a game state using the policy-value network.

        Returns:
            policy probabilities over all actions and scalar value.
        """
        encoded_state = encode_board_state(game.board, game.current_player)
        legal_mask_np = legal_moves_mask(game.board)

        state_tensor = torch.tensor(
            encoded_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        legal_mask_tensor = torch.tensor(
            legal_mask_np,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            policy_logits, value_tensor = self.model(state_tensor)
            policy_probs_tensor = masked_softmax(policy_logits, legal_mask_tensor)

        policy_probs = policy_probs_tensor.squeeze(0).detach().cpu().numpy()
        value = float(value_tensor.item())

        return policy_probs, value

    def _terminal_value(self, game: Game) -> float:
        """
        Return terminal value from the current player's perspective.
        """
        winner = game.get_winner()

        if winner is None:
            return 0.0

        if winner == game.current_player:
            return 1.0

        return -1.0

    def _get_candidate_legal_moves(self, game: Game) -> list[Move]:
        """
        Get legal moves, optionally restricted to cells near existing stones.

        Restricting candidate moves makes NN-MCTS much faster on 15x15 boards.
        """
        legal_moves = game.get_legal_moves()

        if self.candidate_radius is None:
            return legal_moves

        if len(game.board.move_history) == 0:
            centre = (game.board.size // 2, game.board.size // 2)
            return [centre] if centre in legal_moves else legal_moves

        occupied_positions = [
            move for move, _player in game.board.move_history
        ]

        candidates: set[Move] = set()

        for row, col in occupied_positions:
            for dr in range(-self.candidate_radius, self.candidate_radius + 1):
                for dc in range(-self.candidate_radius, self.candidate_radius + 1):
                    candidate = (row + dr, col + dc)

                    if candidate in legal_moves:
                        candidates.add(candidate)

        if not candidates:
            return legal_moves

        return sorted(candidates)