from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from gomoku.board import Board


@dataclass(frozen=True)
class NetworkOutput:
    """
    Output container for the policy-value network.

    policy_logits:
        Raw logits for all board positions.
        Shape: [batch_size, board_size * board_size]

    value:
        Estimated value of the current position from the current player's view.
        Shape: [batch_size, 1]
        Range: approximately [-1, 1]
    """

    policy_logits: torch.Tensor
    value: torch.Tensor


class GomokuPolicyValueNet(nn.Module):
    """
    A small policy-value neural network for Gomoku.

    The network has two output heads:

    1. Policy head:
       Predicts a score for every board position.

    2. Value head:
       Predicts whether the current position is favourable for the current player.

    This is a lightweight network suitable for early NN-MCTS experiments.
    It can be expanded later if stronger training is required.
    """

    def __init__(self, board_size: int = 15, num_channels: int = 64) -> None:
        super().__init__()

        if board_size <= 0:
            raise ValueError("board_size must be positive.")

        self.board_size = board_size
        self.num_actions = board_size * board_size

        self.shared_layers = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x:
                Tensor with shape [batch_size, 3, board_size, board_size].

        Returns:
            policy_logits:
                Tensor with shape [batch_size, board_size * board_size].

            value:
                Tensor with shape [batch_size, 1].
        """
        if x.ndim != 4:
            raise ValueError(
                "Input tensor must have shape [batch_size, 3, board_size, board_size]."
            )

        if x.shape[1] != 3:
            raise ValueError("Input tensor must have exactly 3 channels.")

        if x.shape[2] != self.board_size or x.shape[3] != self.board_size:
            raise ValueError(
                f"Input board size must be {self.board_size}x{self.board_size}."
            )

        features = self.shared_layers(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value


def encode_board_state(
    board: Board,
    current_player: int,
) -> np.ndarray:
    """
    Encode a board state as a 3-channel numpy array.

    Channels:
        0. Current player's stones
        1. Opponent's stones
        2. Current player indicator

    Args:
        board:
            Gomoku board.

        current_player:
            Board.BLACK or Board.WHITE.

    Returns:
        Encoded board state with shape [3, board_size, board_size].
    """
    if current_player not in (Board.BLACK, Board.WHITE):
        raise ValueError("current_player must be Board.BLACK or Board.WHITE.")

    current_stones = (board.grid == current_player).astype(np.float32)
    opponent_stones = (board.grid == -current_player).astype(np.float32)

    player_plane = np.full(
        shape=(board.size, board.size),
        fill_value=float(current_player),
        dtype=np.float32,
    )

    encoded = np.stack(
        [current_stones, opponent_stones, player_plane],
        axis=0,
    )

    return encoded.astype(np.float32)


def legal_moves_mask(board: Board) -> np.ndarray:
    """
    Create a binary mask for legal moves.

    Legal moves are empty cells.

    Returns:
        A flattened mask with shape [board_size * board_size].
        Legal positions are 1.0 and illegal positions are 0.0.
    """
    mask = (board.grid == Board.EMPTY).astype(np.float32)
    return mask.reshape(-1)


def move_to_action(move: tuple[int, int], board_size: int) -> int:
    """
    Convert a board move to a flattened action index.

    Example:
        move (row, col) becomes row * board_size + col.
    """
    row, col = move

    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"Move {move} is outside board size {board_size}.")

    return row * board_size + col


def action_to_move(action: int, board_size: int) -> tuple[int, int]:
    """
    Convert a flattened action index back to a board move.
    """
    if not (0 <= action < board_size * board_size):
        raise ValueError(f"Action {action} is invalid for board size {board_size}.")

    row = action // board_size
    col = action % board_size

    return row, col


def masked_softmax(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Apply softmax only over legal moves.

    Args:
        logits:
            Raw policy logits.

        legal_mask:
            Binary mask with 1 for legal actions and 0 for illegal actions.
            Shape must be compatible with logits.

    Returns:
        Probability distribution over legal actions.
    """
    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must have the same shape.")

    masked_logits = logits.masked_fill(legal_mask <= 0, -1e9)
    probabilities = torch.softmax(masked_logits, dim=dim)

    return probabilities