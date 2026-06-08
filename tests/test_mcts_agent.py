import pytest

from gomoku.board import Board
from gomoku.game import Game
from gomoku.mcts_agent import MCTSAgent


def test_mcts_agent_selects_centre_on_empty_board():
    game = Game(board_size=15)
    agent = MCTSAgent(simulations=10, seed=42)

    move = agent.select_move(game)

    assert move == (7, 7)


def test_mcts_agent_selects_legal_move():
    game = Game(board_size=15)
    agent = MCTSAgent(simulations=10, seed=42)

    game.play_move((7, 7))
    move = agent.select_move(game)

    assert move in game.get_legal_moves()


def test_mcts_agent_can_play_a_move():
    game = Game(board_size=15)
    agent = MCTSAgent(simulations=10, seed=42)

    move = agent.select_move(game)
    game.play_move(move)

    assert game.board.grid[move[0], move[1]] == Board.BLACK


def test_mcts_agent_raises_error_when_no_legal_moves():
    game = Game(board_size=1)
    agent = MCTSAgent(simulations=10, seed=42)

    game.play_move((0, 0))

    with pytest.raises(ValueError):
        agent.select_move(game)


def test_mcts_agent_finds_immediate_winning_move_in_rollout_policy():
    game = Game(board_size=15)
    agent = MCTSAgent(simulations=10, seed=42)

    game.board.place_stone(7, 0, Board.BLACK)
    game.board.place_stone(7, 1, Board.BLACK)
    game.board.place_stone(7, 2, Board.BLACK)
    game.board.place_stone(7, 3, Board.BLACK)
    game.current_player = Board.BLACK

    move = agent._select_rollout_move(game)

    assert move == (7, 4)


def test_mcts_agent_blocks_immediate_opponent_win_in_rollout_policy():
    game = Game(board_size=15)
    agent = MCTSAgent(simulations=10, seed=42)

    game.board.place_stone(7, 0, Board.WHITE)
    game.board.place_stone(7, 1, Board.WHITE)
    game.board.place_stone(7, 2, Board.WHITE)
    game.board.place_stone(7, 3, Board.WHITE)
    game.current_player = Board.BLACK

    move = agent._select_rollout_move(game)

    assert move == (7, 4)


def test_mcts_rollout_depth_limit_is_positive():
    with pytest.raises(ValueError):
        MCTSAgent(simulations=10, rollout_depth_limit=0)


def test_mcts_cutoff_evaluation_returns_valid_value():
    game = Game(board_size=15)
    agent = MCTSAgent(simulations=10, rollout_depth_limit=1, seed=42)

    game.board.place_stone(7, 7, Board.BLACK)
    game.current_player = Board.WHITE

    result = agent._evaluate_cutoff_position(game, Board.BLACK)

    assert result in (0.0, 0.5, 1.0)