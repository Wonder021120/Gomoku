from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from experiments.run_match import MatchResult, run_match
from gomoku.agents import BaseAgent, GreedyAgent, RandomAgent
from gomoku.minimax_agent import MinimaxAgent


def create_agent(
    agent_name: str,
    seed: int,
    minimax_depth: int,
    candidate_radius: int,
) -> BaseAgent:
    """
    Create an agent by name.
    """
    if agent_name == "random":
        return RandomAgent(seed=seed)

    if agent_name == "greedy":
        return GreedyAgent(seed=seed)

    if agent_name == "minimax":
        return MinimaxAgent(
            depth=minimax_depth,
            candidate_radius=candidate_radius,
            seed=seed,
        )

    raise ValueError(f"Unsupported agent: {agent_name}")


def save_results_to_csv(results: list[MatchResult], output_path: Path, rule_name: str) -> None:
    """
    Save match results to a CSV file.
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
        "black_agent_config",
        "white_agent_config",
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


def print_summary(
    summary: dict[str, float | int],
    rule_name: str,
    black_agent_name: str,
    white_agent_name: str,
    output_path: Path,
) -> None:
    print("Tournament finished")
    print(f"Rule: {rule_name}")
    print(f"Black agent: {black_agent_name}")
    print(f"White agent: {white_agent_name}")
    print(f"Games: {summary['total_games']}")
    print(f"Black wins: {summary['black_wins']}")
    print(f"White wins: {summary['white_wins']}")
    print(f"Draws: {summary['draws']}")
    print(f"First-player wins: {summary['first_player_wins']}")
    print(f"First-player win rate: {summary['first_player_win_rate']:.2%}")
    print(f"Average moves: {summary['avg_moves']:.2f}")
    print(f"Average black decision time: {summary['avg_black_time']:.6f}s")
    print(f"Average white decision time: {summary['avg_white_time']:.6f}s")
    print(f"CSV saved to: {output_path}")


def run_tournament(
    games: int,
    board_size: int,
    rule_name: str,
    black_agent_name: str,
    white_agent_name: str,
    seed: int,
    output_path: Path,
    minimax_depth: int,
    candidate_radius: int,
) -> list[MatchResult]:
    """
    Run an AI-vs-AI tournament.

    Currently only the standard rule is supported.
    """
    if games <= 0:
        raise ValueError("Number of games must be positive.")

    if board_size <= 0:
        raise ValueError("Board size must be positive.")

    if minimax_depth <= 0:
        raise ValueError("Minimax depth must be positive.")

    if candidate_radius <= 0:
        raise ValueError("Candidate radius must be positive.")

    if rule_name != "standard":
        raise NotImplementedError("Only the standard rule is currently supported.")

    results: list[MatchResult] = []

    for game_index in range(games):
        black_agent = create_agent(
            agent_name=black_agent_name,
            seed=seed + game_index * 2,
            minimax_depth=minimax_depth,
            candidate_radius=candidate_radius,
        )
        white_agent = create_agent(
            agent_name=white_agent_name,
            seed=seed + game_index * 2 + 1,
            minimax_depth=minimax_depth,
            candidate_radius=candidate_radius,
        )

        result = run_match(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=board_size,
            verbose=False,
        )

        results.append(result)

    save_results_to_csv(results, output_path=output_path, rule_name=rule_name)

    summary = summarise_results(results)
    print_summary(
        summary=summary,
        rule_name=rule_name,
        black_agent_name=black_agent_name,
        white_agent_name=white_agent_name,
        output_path=output_path,
    )

    return results


def format_agent_name_for_filename(
    agent_name: str,
    minimax_depth: int,
    candidate_radius: int,
) -> str:
    """
    Format agent name for output filenames.
    """
    if agent_name == "minimax":
        return f"minimax_d{minimax_depth}_r{candidate_radius}"

    return agent_name


def build_default_output_path(
    black_agent: str,
    white_agent: str,
    rule_name: str,
    board_size: int,
    minimax_depth: int,
    candidate_radius: int,
) -> Path:
    black_name = format_agent_name_for_filename(
        agent_name=black_agent,
        minimax_depth=minimax_depth,
        candidate_radius=candidate_radius,
    )
    white_name = format_agent_name_for_filename(
        agent_name=white_agent,
        minimax_depth=minimax_depth,
        candidate_radius=candidate_radius,
    )

    filename = f"{black_name}_vs_{white_name}_{rule_name}_{board_size}x{board_size}.csv"
    return Path("results/raw") / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Gomoku AI-vs-AI tournament."
    )

    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to run.",
    )

    parser.add_argument(
        "--board-size",
        type=int,
        default=15,
        help="Board size. Default is 15.",
    )

    parser.add_argument(
        "--rule",
        type=str,
        default="standard",
        choices=["standard"],
        help="Opening rule. Currently only standard is supported.",
    )

    parser.add_argument(
        "--black",
        type=str,
        default="random",
        choices=["random", "greedy", "minimax"],
        help="Agent playing black.",
    )

    parser.add_argument(
        "--white",
        type=str,
        default="random",
        choices=["random", "greedy", "minimax"],
        help="Agent playing white.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Base random seed.",
    )

    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=1,
        help="Search depth for MinimaxAgent.",
    )

    parser.add_argument(
        "--candidate-radius",
        type=int,
        default=2,
        help="Candidate move radius for MinimaxAgent.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output CSV file. If omitted, a default filename is used.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_path = args.output
    if output_path is None:
        output_path = build_default_output_path(
            black_agent=args.black,
            white_agent=args.white,
            rule_name=args.rule,
            board_size=args.board_size,
            minimax_depth=args.minimax_depth,
            candidate_radius=args.candidate_radius,
        )

    run_tournament(
        games=args.games,
        board_size=args.board_size,
        rule_name=args.rule,
        black_agent_name=args.black,
        white_agent_name=args.white,
        seed=args.seed,
        output_path=output_path,
        minimax_depth=args.minimax_depth,
        candidate_radius=args.candidate_radius,
    )