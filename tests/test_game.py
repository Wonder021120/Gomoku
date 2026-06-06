import pytest

from gomoku.board import Board
from gomoku.game import Game, GameStatus


def test_game_initialisation():
    game = Game(board_size=15)

    assert game.board.size == 15
    assert game.current_player == Board.BLACK
    assert game.last_move is None
    assert game.status == GameStatus.ONGOING
    assert not game.is_over()


def test_first_move_is_black():
    game = Game(board_size=15)

    game.play_move((7, 7))

    assert game.board.grid[7, 7] == Board.BLACK
    assert game.last_move == (7, 7)
    assert game.current_player == Board.WHITE
    assert game.status == GameStatus.ONGOING


def test_players_alternate_turns():
    game = Game(board_size=15)

    game.play_move((7, 7))
    game.play_move((7, 8))

    assert game.board.grid[7, 7] == Board.BLACK
    assert game.board.grid[7, 8] == Board.WHITE
    assert game.current_player == Board.BLACK


def test_black_horizontal_win():
    game = Game(board_size=15)

    moves = [
        (7, 0),  # Black
        (8, 0),  # White
        (7, 1),  # Black
        (8, 1),  # White
        (7, 2),  # Black
        (8, 2),  # White
        (7, 3),  # Black
        (8, 3),  # White
        (7, 4),  # Black wins
    ]

    for move in moves:
        status = game.play_move(move)

    assert status == GameStatus.BLACK_WIN
    assert game.is_over()
    assert game.get_winner() == Board.BLACK


def test_white_vertical_win():
    game = Game(board_size=15)

    moves = [
        (0, 0),  # Black
        (0, 7),  # White
        (1, 0),  # Black
        (1, 7),  # White
        (2, 0),  # Black
        (2, 7),  # White
        (3, 0),  # Black
        (3, 7),  # White
        (5, 5),  # Black
        (4, 7),  # White wins
    ]

    for move in moves:
        status = game.play_move(move)

    assert status == GameStatus.WHITE_WIN
    assert game.is_over()
    assert game.get_winner() == Board.WHITE


def test_cannot_play_after_game_over():
    game = Game(board_size=15)

    moves = [
        (7, 0),
        (8, 0),
        (7, 1),
        (8, 1),
        (7, 2),
        (8, 2),
        (7, 3),
        (8, 3),
        (7, 4),
    ]

    for move in moves:
        game.play_move(move)

    with pytest.raises(ValueError):
        game.play_move((10, 10))


def test_illegal_move_raises_error():
    game = Game(board_size=15)

    game.play_move((7, 7))

    with pytest.raises(ValueError):
        game.play_move((7, 7))


def test_reset_game():
    game = Game(board_size=15)

    game.play_move((7, 7))
    game.reset()

    assert game.board.grid[7, 7] == Board.EMPTY
    assert game.current_player == Board.BLACK
    assert game.last_move is None
    assert game.status == GameStatus.ONGOING
    assert not game.is_over()


def test_copy_game():
    game = Game(board_size=15)

    game.play_move((7, 7))
    copied = game.copy()

    assert copied.board.grid[7, 7] == Board.BLACK
    assert copied.current_player == game.current_player
    assert copied.last_move == game.last_move
    assert copied.status == game.status

    copied.play_move((7, 8))

    assert game.board.grid[7, 8] == Board.EMPTY
    assert copied.board.grid[7, 8] == Board.WHITE


def test_get_legal_moves_returns_empty_when_game_over():
    game = Game(board_size=15)

    moves = [
        (7, 0),
        (8, 0),
        (7, 1),
        (8, 1),
        (7, 2),
        (8, 2),
        (7, 3),
        (8, 3),
        (7, 4),
    ]

    for move in moves:
        game.play_move(move)

    assert game.is_over()
    assert game.get_legal_moves() == []