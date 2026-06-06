from __future__ import annotations

import sys
from pathlib import Path

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from gomoku.agents import RandomAgent
from gomoku.board import Board
from gomoku.game import Game, GameStatus


def format_winner(status: GameStatus) -> str:
    if status == GameStatus.BLACK_WIN:
        return "black"
    if status == GameStatus.WHITE_WIN:
        return "white"
    if status == GameStatus.DRAW:
        return "draw"
    return "none"


def run_random_vs_random(board_size: int = 15, seed: int = 42) -> Game:
    game = Game(board_size=board_size)

    black_agent = RandomAgent(seed=seed)
    white_agent = RandomAgent(seed=seed + 1)

    move_count = 0

    while not game.is_over():
        if game.current_player == Board.BLACK:
            move = black_agent.select_move(game)
        else:
            move = white_agent.select_move(game)

        game.play_move(move)
        move_count += 1

    print("Game finished")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Status: {game.status.value}")
    print(f"Winner: {format_winner(game.status)}")
    print(f"Moves: {move_count}")
    print()
    print("Final board:")
    print(game.board)

    return game


if __name__ == "__main__":
    run_random_vs_random(board_size=15, seed=42)