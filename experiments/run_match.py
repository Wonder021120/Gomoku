from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from gomoku.agents import BaseAgent, RandomAgent
from gomoku.board import Board
from gomoku.game import Game, GameStatus


@dataclass
class MatchResult:
    """
    Stores the result of a single Gomoku match.
    """

    board_size: int
    black_agent: str
    white_agent: str
    black_agent_config: str
    white_agent_config: str
    status: str
    winner: str
    first_player_win: bool
    moves: int
    black_total_time: float
    white_total_time: float
    black_avg_time: float
    white_avg_time: float


def format_winner(status: GameStatus) -> str:
    if status == GameStatus.BLACK_WIN:
        return "black"
    if status == GameStatus.WHITE_WIN:
        return "white"
    if status == GameStatus.DRAW:
        return "draw"
    return "none"


def get_agent_config(agent: BaseAgent) -> str:
    """
    Return a compact text description of an agent's configuration.

    This is used in CSV results so that experiments remain traceable.
    """
    if agent.name == "minimax":
        return f"depth={agent.depth},candidate_radius={agent.candidate_radius}"

    if agent.name == "mcts":
        return (
            f"simulations={agent.simulations},"
            f"exploration_weight={agent.exploration_weight},"
            f"candidate_radius={agent.candidate_radius},"
            f"rollout_depth_limit={agent.rollout_depth_limit}"
        )

    if hasattr(agent, "seed"):
        return "seeded"

    return "default"


def run_match(
    black_agent: BaseAgent,
    white_agent: BaseAgent,
    board_size: int = 15,
    verbose: bool = True,
) -> MatchResult:
    """
    Run a single AI-vs-AI Gomoku match.

    Args:
        black_agent: Agent playing black stones.
        white_agent: Agent playing white stones.
        board_size: Board size.
        verbose: Whether to print the result.

    Returns:
        MatchResult containing match statistics.
    """
    game = Game(board_size=board_size)

    move_count = 0
    black_total_time = 0.0
    white_total_time = 0.0
    black_moves = 0
    white_moves = 0

    while not game.is_over():
        if game.current_player == Board.BLACK:
            agent = black_agent
        else:
            agent = white_agent

        start_time = time.perf_counter()
        move = agent.select_move(game)
        elapsed_time = time.perf_counter() - start_time

        if game.current_player == Board.BLACK:
            black_total_time += elapsed_time
            black_moves += 1
        else:
            white_total_time += elapsed_time
            white_moves += 1

        game.play_move(move)
        move_count += 1

    winner = format_winner(game.status)

    result = MatchResult(
        board_size=board_size,
        black_agent=black_agent.name,
        white_agent=white_agent.name,
        black_agent_config=get_agent_config(black_agent),
        white_agent_config=get_agent_config(white_agent),
        status=game.status.value,
        winner=winner,
        first_player_win=winner == "black",
        moves=move_count,
        black_total_time=black_total_time,
        white_total_time=white_total_time,
        black_avg_time=black_total_time / black_moves if black_moves > 0 else 0.0,
        white_avg_time=white_total_time / white_moves if white_moves > 0 else 0.0,
    )

    if verbose:
        print_match_result(result)
        print()
        print("Final board:")
        print(game.board)

    return result


def print_match_result(result: MatchResult) -> None:
    print("Game finished")
    print(f"Board size: {result.board_size}x{result.board_size}")
    print(f"Black agent: {result.black_agent}")
    print(f"Black agent config: {result.black_agent_config}")
    print(f"White agent: {result.white_agent}")
    print(f"White agent config: {result.white_agent_config}")
    print(f"Status: {result.status}")
    print(f"Winner: {result.winner}")
    print(f"First-player win: {result.first_player_win}")
    print(f"Moves: {result.moves}")
    print(f"Black total decision time: {result.black_total_time:.6f}s")
    print(f"White total decision time: {result.white_total_time:.6f}s")
    print(f"Black average decision time: {result.black_avg_time:.6f}s")
    print(f"White average decision time: {result.white_avg_time:.6f}s")


def run_random_vs_random(board_size: int = 15, seed: int = 42) -> MatchResult:
    black_agent = RandomAgent(seed=seed)
    white_agent = RandomAgent(seed=seed + 1)

    return run_match(
        black_agent=black_agent,
        white_agent=white_agent,
        board_size=board_size,
        verbose=True,
    )


if __name__ == "__main__":
    run_random_vs_random(board_size=15, seed=42)