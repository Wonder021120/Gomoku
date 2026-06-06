from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from experiments.run_match import MatchResult, run_match
from gomoku.agents import RandomAgent
from gomoku.game import GameStatus


def save_results_to_csv(results: list[MatchResult], output_path: Path, rule_name: str) -> None:
    """
    Save match results to a CSV file.

    Args:
        results: List of MatchResult objects.
        output_path: CSV output path.
        rule_name: Name of the rule used in the tournament.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for game_id, result in enumerate(results, start=1):
        row = asdict(result)
        row["game_id"] = game_id
        row["rule"] = rule_name
        rows.append(row)

    fieldnames = [
        "game_id",
        "rule",
        "board_size",
        "black_agent",
        "white_agent",
        "status",
        "winner",
        "first_player_win",
        "moves",
        "black_total_time",
        "white_total_time",
        "black_avg_time",
        "white_avg_time",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise_results(results: list[MatchResult]) -> dict[str, float | int]:
    """
    Calculate summary statistics for a list of match results.
    """
    total_games = len(results)

    black_wins = sum(1 for result in results if result.winner == "black")
    white_wins = sum(1 for result in results if result.winner == "white")
    draws = sum(1 for result in results if result.winner == "draw")

    first_player_wins = sum(1 for result in results if result.first_player_win)

    avg_moves = mean(result.moves for result in results) if results else 0.0
    avg_black_time = mean(result.black_avg_time for result in results) if results else 0.0
    avg_white_time = mean(result.white_avg_time for result in results) if results else 0.0

    return {
        "total_games": total_games,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "first_player_wins": first_player_wins,
        "first_player_win_rate": first_player_wins / total_games if total_games > 0 else 0.0,
        "avg_moves": avg_moves,
        "avg_black_time": avg_black_time,
        "avg_white_time": avg_white_time,
    }


def print_summary(summary: dict[str, float | int], rule_name: str) -> None:
    print("Tournament finished")
    print(f"Rule: {rule_name}")
    print(f"Games: {summary['total_games']}")
    print(f"Black wins: {summary['black_wins']}")
    print(f"White wins: {summary['white_wins']}")
    print(f"Draws: {summary['draws']}")
    print(f"First-player wins: {summary['first_player_wins']}")
    print(f"First-player win rate: {summary['first_player_win_rate']:.2%}")
    print(f"Average moves: {summary['avg_moves']:.2f}")
    print(f"Average black decision time: {summary['avg_black_time']:.6f}s")
    print(f"Average white decision time: {summary['avg_white_time']:.6f}s")


def run_random_tournament(
    games: int = 100,
    board_size: int = 15,
    rule_name: str = "standard",
    output_path: Path = Path("results/raw/random_vs_random_standard.csv"),
) -> list[MatchResult]:
    """
    Run a RandomAgent vs RandomAgent tournament.
    """
    if rule_name != "standard":
        raise NotImplementedError("Only the standard rule is currently supported.")

    results: list[MatchResult] = []

    for game_index in range(games):
        black_agent = RandomAgent(seed=1000 + game_index * 2)
        white_agent = RandomAgent(seed=1000 + game_index * 2 + 1)

        result = run_match(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=board_size,
            verbose=False,
        )

        results.append(result)

    save_results_to_csv(results, output_path=output_path, rule_name=rule_name)

    summary = summarise_results(results)
    print_summary(summary, rule_name=rule_name)
    print(f"CSV saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_random_tournament(
        games=100,
        board_size=15,
        rule_name="standard",
        output_path=Path("results/raw/random_vs_random_standard.csv"),
    )