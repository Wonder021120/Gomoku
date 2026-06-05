import pytest

from gomoku.board import Board


def test_board_initialisation():
    board = Board(size=15)

    assert board.size == 15
    assert board.grid.shape == (15, 15)
    assert len(board.get_legal_moves()) == 225


def test_place_black_stone():
    board = Board(size=15)

    board.place_stone(7, 7, Board.BLACK)

    assert board.grid[7, 7] == Board.BLACK
    assert not board.is_empty(7, 7)
    assert len(board.move_history) == 1


def test_place_white_stone():
    board = Board(size=15)

    board.place_stone(3, 4, Board.WHITE)

    assert board.grid[3, 4] == Board.WHITE
    assert not board.is_empty(3, 4)


def test_cannot_place_outside_board():
    board = Board(size=15)

    with pytest.raises(ValueError):
        board.place_stone(15, 0, Board.BLACK)

    with pytest.raises(ValueError):
        board.place_stone(-1, 0, Board.BLACK)


def test_cannot_place_on_occupied_cell():
    board = Board(size=15)

    board.place_stone(7, 7, Board.BLACK)

    with pytest.raises(ValueError):
        board.place_stone(7, 7, Board.WHITE)


def test_invalid_player_raises_error():
    board = Board(size=15)

    with pytest.raises(ValueError):
        board.place_stone(7, 7, 99)


def test_undo_move():
    board = Board(size=15)

    board.place_stone(7, 7, Board.BLACK)
    board.undo_move()

    assert board.grid[7, 7] == Board.EMPTY
    assert len(board.move_history) == 0


def test_undo_without_move_raises_error():
    board = Board(size=15)

    with pytest.raises(ValueError):
        board.undo_move()


def test_copy_board():
    board = Board(size=15)
    board.place_stone(7, 7, Board.BLACK)

    copied = board.copy()

    assert copied.size == board.size
    assert copied.grid[7, 7] == Board.BLACK
    assert copied.move_history == board.move_history

    copied.place_stone(7, 8, Board.WHITE)

    assert board.grid[7, 8] == Board.EMPTY
    assert copied.grid[7, 8] == Board.WHITE


def test_reset_board():
    board = Board(size=15)
    board.place_stone(7, 7, Board.BLACK)

    board.reset()

    assert board.grid[7, 7] == Board.EMPTY
    assert len(board.move_history) == 0
    assert len(board.get_legal_moves()) == 225