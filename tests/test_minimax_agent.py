from gomoku.board import Board
from gomoku.game import Game
from gomoku.minimax_agent import MinimaxAgent


def test_minimax_agent_selects_centre_on_empty_board():
    game = Game(board_size=15)
    agent = MinimaxAgent(depth=1, seed=42)

    move = agent.select_move(game)

    assert move == (7, 7)


def test_minimax_agent_selects_legal_move():
    game = Game(board_size=15)
    agent = MinimaxAgent(depth=1, seed=42)

    game.play_move((7, 7))
    move = agent.select_move(game)

    assert move in game.get_legal_moves()


def test_minimax_agent_selects_immediate_winning_move():
    game = Game(board_size=15)
    agent = MinimaxAgent(depth=1, seed=42)

    game.board.place_stone(7, 0, Board.BLACK)
    game.board.place_stone(7, 1, Board.BLACK)
    game.board.place_stone(7, 2, Board.BLACK)
    game.board.place_stone(7, 3, Board.BLACK)
    game.current_player = Board.BLACK

    move = agent.select_move(game)

    assert move == (7, 4)


def test_minimax_agent_blocks_opponent_immediate_win():
    game = Game(board_size=15)
    agent = MinimaxAgent(depth=2, seed=42)

    game.board.place_stone(7, 0, Board.WHITE)
    game.board.place_stone(7, 1, Board.WHITE)
    game.board.place_stone(7, 2, Board.WHITE)
    game.board.place_stone(7, 3, Board.WHITE)
    game.current_player = Board.BLACK

    move = agent.select_move(game)

    assert move == (7, 4)


def test_minimax_agent_can_play_a_move_in_game():
    game = Game(board_size=15)
    agent = MinimaxAgent(depth=1, seed=42)

    move = agent.select_move(game)
    game.play_move(move)

    assert game.board.grid[move[0], move[1]] == Board.BLACK