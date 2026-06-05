from __future__ import annotations

from typing import Optional, Tuple

from gomoku.board import Board, Move


DIRECTIONS = [
    (0, 1),   # horizontal
    (1, 0),   # vertical
    (1, 1),   # diagonal down-right
    (1, -1),  # diagonal down-left
]


def count_stones_in_direction(
    board: Board,
    row: int,
    col: int,
    player: int,
    delta_row: int,
    delta_col: int,
) -> int:
    """
    Count consecutive stones belonging to player from a starting position
    in one direction.

    The starting position itself is not counted.
    """
    count = 0
    current_row = row + delta_row
    current_col = col + delta_col

    while board.is_inside(current_row, current_col):
        if board.grid[current_row, current_col] != player:
            break

        count += 1
        current_row += delta_row
        current_col += delta_col

    return count


def is_winning_move(board: Board, last_move: Move) -> bool:
    """
    Return True if the last move creates five or more consecutive stones.

    Args:
        board: Current board.
        last_move: The most recent move as (row, col).
    """
    row, col = last_move

    if not board.is_inside(row, col):
        return False

    player = int(board.grid[row, col])

    if player == Board.EMPTY:
        return False

    for delta_row, delta_col in DIRECTIONS:
        count_forward = count_stones_in_direction(
            board, row, col, player, delta_row, delta_col
        )
        count_backward = count_stones_in_direction(
            board, row, col, player, -delta_row, -delta_col
        )

        total_count = 1 + count_forward + count_backward

        if total_count >= 5:
            return True

    return False


def get_winner(board: Board, last_move: Optional[Move]) -> Optional[int]:
    """
    Return the winning player after the last move.

    Returns:
        Board.BLACK if black wins.
        Board.WHITE if white wins.
        None if there is no winner.
    """
    if last_move is None:
        return None

    if is_winning_move(board, last_move):
        row, col = last_move
        return int(board.grid[row, col])

    return None