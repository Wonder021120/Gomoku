import numpy as np
import pytest
import torch

from gomoku.board import Board
from gomoku.neural_network import (
    GomokuPolicyValueNet,
    action_to_move,
    encode_board_state,
    legal_moves_mask,
    masked_softmax,
    move_to_action,
)


def test_policy_value_network_output_shapes():
    board_size = 15
    batch_size = 4

    model = GomokuPolicyValueNet(board_size=board_size)
    x = torch.zeros((batch_size, 3, board_size, board_size), dtype=torch.float32)

    policy_logits, value = model(x)

    assert policy_logits.shape == (batch_size, board_size * board_size)
    assert value.shape == (batch_size, 1)


def test_policy_value_network_value_range():
    board_size = 15

    model = GomokuPolicyValueNet(board_size=board_size)
    x = torch.randn((2, 3, board_size, board_size), dtype=torch.float32)

    _, value = model(x)

    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_encode_board_state_for_black_player():
    board = Board(size=15)
    board.place_stone(7, 7, Board.BLACK)
    board.place_stone(7, 8, Board.WHITE)

    encoded = encode_board_state(board, Board.BLACK)

    assert encoded.shape == (3, 15, 15)
    assert encoded.dtype == np.float32

    assert encoded[0, 7, 7] == 1.0
    assert encoded[0, 7, 8] == 0.0

    assert encoded[1, 7, 8] == 1.0
    assert encoded[1, 7, 7] == 0.0

    assert np.all(encoded[2] == 1.0)


def test_encode_board_state_for_white_player():
    board = Board(size=15)
    board.place_stone(7, 7, Board.BLACK)
    board.place_stone(7, 8, Board.WHITE)

    encoded = encode_board_state(board, Board.WHITE)

    assert encoded.shape == (3, 15, 15)

    assert encoded[0, 7, 8] == 1.0
    assert encoded[0, 7, 7] == 0.0

    assert encoded[1, 7, 7] == 1.0
    assert encoded[1, 7, 8] == 0.0

    assert np.all(encoded[2] == -1.0)


def test_encode_board_state_rejects_invalid_player():
    board = Board(size=15)

    with pytest.raises(ValueError):
        encode_board_state(board, 0)


def test_legal_moves_mask_marks_empty_cells():
    board = Board(size=15)
    board.place_stone(7, 7, Board.BLACK)
    board.place_stone(7, 8, Board.WHITE)

    mask = legal_moves_mask(board)

    assert mask.shape == (15 * 15,)
    assert mask[move_to_action((7, 7), 15)] == 0.0
    assert mask[move_to_action((7, 8), 15)] == 0.0
    assert mask[move_to_action((0, 0), 15)] == 1.0


def test_move_action_conversion_round_trip():
    board_size = 15
    move = (6, 9)

    action = move_to_action(move, board_size)
    recovered_move = action_to_move(action, board_size)

    assert recovered_move == move


def test_move_to_action_rejects_invalid_move():
    with pytest.raises(ValueError):
        move_to_action((-1, 0), 15)

    with pytest.raises(ValueError):
        move_to_action((15, 0), 15)


def test_action_to_move_rejects_invalid_action():
    with pytest.raises(ValueError):
        action_to_move(-1, 15)

    with pytest.raises(ValueError):
        action_to_move(225, 15)


def test_masked_softmax_assigns_zero_probability_to_illegal_actions():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    legal_mask = torch.tensor([[1.0, 0.0, 1.0]])

    probabilities = masked_softmax(logits, legal_mask)

    assert probabilities.shape == logits.shape
    assert probabilities[0, 1].item() == pytest.approx(0.0)
    assert probabilities.sum().item() == pytest.approx(1.0)