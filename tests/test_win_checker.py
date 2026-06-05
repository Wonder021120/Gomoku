from gomoku.board import Board
from gomoku.win_checker import get_winner, is_winning_move


def test_no_winner_on_empty_board():
    board = Board(size=15)

    assert get_winner(board, None) is None


def test_horizontal_win():
    board = Board(size=15)

    for col in range(5):
        board.place_stone(7, col, Board.BLACK)

    assert is_winning_move(board, (7, 4))
    assert get_winner(board, (7, 4)) == Board.BLACK


def test_vertical_win():
    board = Board(size=15)

    for row in range(5):
        board.place_stone(row, 7, Board.WHITE)

    assert is_winning_move(board, (4, 7))
    assert get_winner(board, (4, 7)) == Board.WHITE


def test_diagonal_down_right_win():
    board = Board(size=15)

    for i in range(5):
        board.place_stone(i, i, Board.BLACK)

    assert is_winning_move(board, (4, 4))
    assert get_winner(board, (4, 4)) == Board.BLACK


def test_diagonal_down_left_win():
    board = Board(size=15)

    moves = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]

    for row, col in moves:
        board.place_stone(row, col, Board.WHITE)

    assert is_winning_move(board, (4, 0))
    assert get_winner(board, (4, 0)) == Board.WHITE


def test_four_in_a_row_is_not_win():
    board = Board(size=15)

    for col in range(4):
        board.place_stone(7, col, Board.BLACK)

    assert not is_winning_move(board, (7, 3))
    assert get_winner(board, (7, 3)) is None


def test_gap_between_stones_is_not_win():
    board = Board(size=15)

    board.place_stone(7, 0, Board.BLACK)
    board.place_stone(7, 1, Board.BLACK)
    board.place_stone(7, 2, Board.BLACK)
    board.place_stone(7, 4, Board.BLACK)
    board.place_stone(7, 5, Board.BLACK)

    assert not is_winning_move(board, (7, 5))
    assert get_winner(board, (7, 5)) is None


def test_win_can_be_detected_from_middle_move():
    board = Board(size=15)

    board.place_stone(7, 5, Board.BLACK)
    board.place_stone(7, 6, Board.BLACK)
    board.place_stone(7, 7, Board.BLACK)
    board.place_stone(7, 8, Board.BLACK)
    board.place_stone(7, 9, Board.BLACK)

    assert is_winning_move(board, (7, 7))
    assert get_winner(board, (7, 7)) == Board.BLACK


def test_empty_last_move_is_not_win():
    board = Board(size=15)

    assert not is_winning_move(board, (7, 7))
    assert get_winner(board, (7, 7)) is None


def test_outside_last_move_is_not_win():
    board = Board(size=15)

    assert not is_winning_move(board, (15, 15))