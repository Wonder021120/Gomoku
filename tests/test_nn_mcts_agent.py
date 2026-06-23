import pytest

from gomoku.board import Board
from gomoku.game import Game
from gomoku.nn_mcts_agent import NNMCTSAgent


def test_nn_mcts_agent_selects_legal_move_on_empty_board():
    game = Game(board_size=15, rule_name="standard")
    agent = NNMCTSAgent(board_size=15, simulations=5, seed=2026)

    move = agent.select_move(game)

    assert move in game.get_legal_moves()
    assert move == (7, 7)


def test_nn_mcts_agent_selects_legal_move_after_opening():
    game = Game(board_size=15, rule_name="standard")
    game.play_move((7, 7))
    game.play_move((7, 8))

    agent = NNMCTSAgent(board_size=15, simulations=5, seed=2026)

    move = agent.select_move(game)

    assert move in game.get_legal_moves()


def test_nn_mcts_agent_rejects_invalid_simulation_count():
    with pytest.raises(ValueError):
        NNMCTSAgent(board_size=15, simulations=0)


def test_nn_mcts_agent_rejects_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        NNMCTSAgent(
            board_size=15,
            simulations=5,
            checkpoint_path="missing_checkpoint.pt",
        )


def test_nn_mcts_agent_can_load_checkpoint(tmp_path):
    import torch

    from gomoku.neural_network import GomokuPolicyValueNet

    checkpoint_path = tmp_path / "test_checkpoint.pt"

    model = GomokuPolicyValueNet(board_size=15)

    torch.save(
        {
            "epoch": 1,
            "board_size": 15,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "metadata": {},
        },
        checkpoint_path,
    )

    agent = NNMCTSAgent(
        board_size=15,
        simulations=5,
        checkpoint_path=checkpoint_path,
    )

    game = Game(board_size=15, rule_name="standard")
    move = agent.select_move(game)

    assert move in game.get_legal_moves()


def test_nn_mcts_agent_checkpoint_board_size_mismatch(tmp_path):
    import torch

    from gomoku.neural_network import GomokuPolicyValueNet

    checkpoint_path = tmp_path / "test_checkpoint.pt"

    model = GomokuPolicyValueNet(board_size=9)

    torch.save(
        {
            "epoch": 1,
            "board_size": 9,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "metadata": {},
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError):
        NNMCTSAgent(
            board_size=15,
            simulations=5,
            checkpoint_path=checkpoint_path,
        )