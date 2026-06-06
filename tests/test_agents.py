import pytest

from gomoku.agents import GreedyAgent, RandomAgent
from gomoku.board import Board
from gomoku.game import Game, GameStatus


def test_random_agent_selects_legal_move():
    game = Game(board_size=15)
    agent = RandomAgent(seed=42)

    move = agent.select_move(game)

    assert move in game.get_legal_moves()


def test_random_agent_can_play_move():
    game = Game(board_size=15)
    agent = RandomAgent(seed=42)

    move = agent.select_move(game)
    game.play_move(move)

    assert game.board.grid[move[0], move[1]] == Board.BLACK
    assert game.current_player == Board.WHITE


def test_random_agent_raises_error_when_no_legal_moves():
    game = Game(board_size=1)
    agent = RandomAgent(seed=42)

    game.play_move((0, 0))

    assert game.status == GameStatus.DRAW

    with pytest.raises(ValueError):
        agent.select_move(game)


def test_random_agent_seed_reproducibility():
    game_1 = Game(board_size=15)
    game_2 = Game(board_size=15)

    agent_1 = RandomAgent(seed=123)
    agent_2 = RandomAgent(seed=123)

    move_1 = agent_1.select_move(game_1)
    move_2 = agent_2.select_move(game_2)

    assert move_1 == move_2


def test_greedy_agent_selects_winning_move():
    game = Game(board_size=15)
    agent = GreedyAgent(seed=42)

    game.board.place_stone(7, 0, Board.BLACK)
    game.board.place_stone(7, 1, Board.BLACK)
    game.board.place_stone(7, 2, Board.BLACK)
    game.board.place_stone(7, 3, Board.BLACK)
    game.current_player = Board.BLACK

    move = agent.select_move(game)

    assert move == (7, 4)


def test_greedy_agent_blocks_opponent_winning_move():
    game = Game(board_size=15)
    agent = GreedyAgent(seed=42)

    game.board.place_stone(7, 0, Board.WHITE)
    game.board.place_stone(7, 1, Board.WHITE)
    game.board.place_stone(7, 2, Board.WHITE)
    game.board.place_stone(7, 3, Board.WHITE)
    game.current_player = Board.BLACK

    move = agent.select_move(game)

    assert move == (7, 4)


def test_greedy_agent_prefers_winning_over_blocking():
    game = Game(board_size=15)
    agent = GreedyAgent(seed=42)

    # Black can win immediately.
    game.board.place_stone(7, 0, Board.BLACK)
    game.board.place_stone(7, 1, Board.BLACK)
    game.board.place_stone(7, 2, Board.BLACK)
    game.board.place_stone(7, 3, Board.BLACK)

    # White also has a threat, but black should win first.
    game.board.place_stone(8, 0, Board.WHITE)
    game.board.place_stone(8, 1, Board.WHITE)
    game.board.place_stone(8, 2, Board.WHITE)
    game.board.place_stone(8, 3, Board.WHITE)

    game.current_player = Board.BLACK

    move = agent.select_move(game)

    assert move == (7, 4)


def test_greedy_agent_selects_legal_random_move_when_no_tactic():
    game = Game(board_size=15)
    agent = GreedyAgent(seed=42)

    move = agent.select_move(game)

    assert move in game.get_legal_moves()


def test_greedy_agent_raises_error_when_no_legal_moves():
    game = Game(board_size=1)
    agent = GreedyAgent(seed=42)

    game.play_move((0, 0))

    assert game.status == GameStatus.DRAW

    with pytest.raises(ValueError):
        agent.select_move(game)