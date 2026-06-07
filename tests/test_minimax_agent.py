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

def test_minimax_scores_open_four_higher_than_blocked_four():
    agent = MinimaxAgent(depth=1, seed=42)

    open_four_board = Board(size=15)
    open_four_board.place_stone(7, 5, Board.BLACK)
    open_four_board.place_stone(7, 6, Board.BLACK)
    open_four_board.place_stone(7, 7, Board.BLACK)
    open_four_board.place_stone(7, 8, Board.BLACK)

    blocked_four_board = Board(size=15)
    blocked_four_board.place_stone(7, 4, Board.WHITE)
    blocked_four_board.place_stone(7, 5, Board.BLACK)
    blocked_four_board.place_stone(7, 6, Board.BLACK)
    blocked_four_board.place_stone(7, 7, Board.BLACK)
    blocked_four_board.place_stone(7, 8, Board.BLACK)

    open_score = agent._score_player_patterns(open_four_board, Board.BLACK)
    blocked_score = agent._score_player_patterns(blocked_four_board, Board.BLACK)

    assert open_score > blocked_score


def test_minimax_scores_open_three_higher_than_blocked_three():
    agent = MinimaxAgent(depth=1, seed=42)

    open_three_board = Board(size=15)
    open_three_board.place_stone(7, 5, Board.BLACK)
    open_three_board.place_stone(7, 6, Board.BLACK)
    open_three_board.place_stone(7, 7, Board.BLACK)

    blocked_three_board = Board(size=15)
    blocked_three_board.place_stone(7, 4, Board.WHITE)
    blocked_three_board.place_stone(7, 5, Board.BLACK)
    blocked_three_board.place_stone(7, 6, Board.BLACK)
    blocked_three_board.place_stone(7, 7, Board.BLACK)

    open_score = agent._score_player_patterns(open_three_board, Board.BLACK)
    blocked_score = agent._score_player_patterns(blocked_three_board, Board.BLACK)

    assert open_score > blocked_score