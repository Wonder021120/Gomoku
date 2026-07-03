from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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
    Policy-value neural network for Gomoku.

    By default, this class keeps the original project-compatible network:
        - board_size=15
        - input_channels=3
        - shared CNN channels: 64 -> 64 -> 64

    For the new 9x9 paper-style NN-MCTS reproduction, instantiate explicitly:

        GomokuPolicyValueNet(
            board_size=9,
            input_channels=4,
            conv_channels=(32, 64, 128),
            policy_head_channels=4,
            value_head_channels=2,
        )

    This lets the old tests and old 15x15 code keep working while allowing the
    new 9x9 reproduction model to be used by the NN-MCTS training pipeline.
    """

    def __init__(
        self,
        board_size: int = 15,
        num_channels: int = 64,
        input_channels: int = 3,
        conv_channels: Optional[Sequence[int]] = None,
        policy_head_channels: int = 2,
        value_head_channels: int = 1,
        value_hidden_size: int = 64,
    ) -> None:
        super().__init__()

        if board_size <= 0:
            raise ValueError("board_size must be positive.")

        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")

        if num_channels <= 0:
            raise ValueError("num_channels must be positive.")

        if conv_channels is None:
            conv_channels = (num_channels, num_channels, num_channels)

        if len(conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one channel size.")

        if any(ch <= 0 for ch in conv_channels):
            raise ValueError("all conv_channels values must be positive.")

        if policy_head_channels <= 0:
            raise ValueError("policy_head_channels must be positive.")

        if value_head_channels <= 0:
            raise ValueError("value_head_channels must be positive.")

        if value_hidden_size <= 0:
            raise ValueError("value_hidden_size must be positive.")

        self.board_size = board_size
        self.input_channels = input_channels
        self.conv_channels = tuple(int(ch) for ch in conv_channels)
        self.policy_head_channels = int(policy_head_channels)
        self.value_head_channels = int(value_head_channels)
        self.value_hidden_size = int(value_hidden_size)
        self.num_actions = board_size * board_size

        shared_layers = []
        in_channels = input_channels

        for out_channels in self.conv_channels:
            shared_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels

        self.shared_layers = nn.Sequential(*shared_layers)

        final_channels = self.conv_channels[-1]

        self.policy_head = nn.Sequential(
            nn.Conv2d(final_channels, self.policy_head_channels, kernel_size=1),
            nn.BatchNorm2d(self.policy_head_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.policy_head_channels * board_size * board_size, self.num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(final_channels, self.value_head_channels, kernel_size=1),
            nn.BatchNorm2d(self.value_head_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.value_head_channels * board_size * board_size, self.value_hidden_size),
            nn.ReLU(),
            nn.Linear(self.value_hidden_size, 1),
            nn.Tanh(),
        )

    @classmethod
    def create_paper_9x9(cls) -> "GomokuPolicyValueNet":
        """
        Convenience constructor for the 9x9 paper-style reproduction network.

        Architecture:
            input: 4 channels
            shared CNN: 4 -> 32 -> 64 -> 128
            policy head: 1x1 conv with 4 channels
            value head: 1x1 conv with 2 channels
        """

        return cls(
            board_size=9,
            input_channels=4,
            conv_channels=(32, 64, 128),
            policy_head_channels=4,
            value_head_channels=2,
            value_hidden_size=64,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x:
                Tensor with shape:
                    [batch_size, input_channels, board_size, board_size]

        Returns:
            policy_logits:
                Tensor with shape [batch_size, board_size * board_size].

            value:
                Tensor with shape [batch_size, 1].
        """

        if x.ndim != 4:
            raise ValueError(
                "Input tensor must have shape "
                "[batch_size, input_channels, board_size, board_size]."
            )

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Input tensor must have exactly {self.input_channels} channels, "
                f"but got {x.shape[1]}."
            )

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
    include_last_move: bool = False,
) -> np.ndarray:
    """
    Encode a board state as a numpy array.

    Default old-compatible 3-channel encoding:
        0. Current player's stones
        1. Opponent's stones
        2. Current player indicator

    Paper-style 4-channel encoding, used when include_last_move=True:
        0. Current player's stones
        1. Opponent's stones
        2. Last move on the board
        3. Current player indicator

    Args:
        board:
            Gomoku board.

        current_player:
            Board.BLACK or Board.WHITE.

        include_last_move:
            Whether to include the last-move plane.

    Returns:
        Encoded board state with shape:
            [3, board_size, board_size] when include_last_move=False
            [4, board_size, board_size] when include_last_move=True
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

    if not include_last_move:
        encoded = np.stack(
            [current_stones, opponent_stones, player_plane],
            axis=0,
        )

        return encoded.astype(np.float32)

    last_move_plane = np.zeros((board.size, board.size), dtype=np.float32)
    last_move = _get_last_move_from_board(board)

    if last_move is not None:
        row, col = last_move
        if 0 <= row < board.size and 0 <= col < board.size:
            last_move_plane[row, col] = 1.0

    encoded = np.stack(
        [current_stones, opponent_stones, last_move_plane, player_plane],
        axis=0,
    )

    return encoded.astype(np.float32)


def _get_last_move_from_board(board: Board) -> Optional[tuple[int, int]]:
    """
    Extract the most recent move from board.move_history.

    Expected project format:
        [((row, col), player), ...]
    """

    move_history = getattr(board, "move_history", None)

    if not move_history:
        return None

    last_entry = move_history[-1]

    if (
        isinstance(last_entry, tuple)
        and len(last_entry) >= 1
        and isinstance(last_entry[0], tuple)
        and len(last_entry[0]) == 2
    ):
        row, col = last_entry[0]
        return int(row), int(col)

    if isinstance(last_entry, tuple) and len(last_entry) == 2:
        row, col = last_entry
        if isinstance(row, int) and isinstance(col, int):
            return int(row), int(col)

    return None


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
