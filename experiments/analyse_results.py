from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def summarise_file(csv_path: Path) -> dict[str, object]:
    """
    Summarise one raw tournament CSV file.
    """
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    total_games = len(df)

    black_wins = int((df["winner"] == "black").sum())
    white_wins = int((df["winner"] == "white").sum())
    draws = int((df["winner"] == "draw").sum())
    first_player_wins = int(df["first_player_win"].sum())

    black_agent = str(df["black_agent"].iloc[0])
    white_agent = str(df["white_agent"].iloc[0])
    black_agent_config = str(df["black_agent_config"].iloc[0])
    white_agent_config = str(df["white_agent_config"].iloc[0])
    rule = str(df["rule"].iloc[0])
    board_size = int(df["board_size"].iloc[0])

    avg_moves = float(df["moves"].mean())
    avg_black_time = float(df["black_avg_time"].mean())
    avg_white_time = float(df["white_avg_time"].mean())

    return {
        "source_file": csv_path.name,
        "rule": rule,
        "board_size": board_size,
        "black_agent": black_agent,
        "white_agent": white_agent,
        "black_agent_config": black_agent_config,
        "white_agent_config": white_agent_config,
        "games": total_games,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "black_win_rate": black_wins / total_games,
        "white_win_rate": white_wins / total_games,
        "draw_rate": draws / total_games,
        "first_player_wins": first_player_wins,
        "first_player_win_rate": first_player_wins / total_games,
        "avg_moves": avg_moves,
        "avg_black_time": avg_black_time,
        "avg_white_time": avg_white_time,
        "avg_decision_time": (avg_black_time + avg_white_time) / 2,
    }


def analyse_results(input_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Analyse all CSV files in input_dir and save a summary CSV.
    """
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    summaries = []

    for csv_path in csv_files:
        try:
            summaries.append(summarise_file(csv_path))
        except Exception as error:
            print(f"Skipping {csv_path}: {error}")

    if not summaries:
        raise ValueError("No valid CSV files could be summarised.")

    summary_df = pd.DataFrame(summaries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8")

    return summary_df


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """
    Print a compact summary table to the terminal.
    """
    columns = [
        "source_file",
        "rule",
        "board_size",
        "black_agent",
        "white_agent",
        "games",
        "black_win_rate",
        "white_win_rate",
        "draw_rate",
        "first_player_win_rate",
        "avg_moves",
        "avg_decision_time",
    ]

    display_df = summary_df[columns].copy()

    percentage_columns = [
        "black_win_rate",
        "white_win_rate",
        "draw_rate",
        "first_player_win_rate",
    ]

    for column in percentage_columns:
        display_df[column] = display_df[column].map(lambda value: f"{value:.2%}")

    display_df["avg_moves"] = display_df["avg_moves"].map(lambda value: f"{value:.2f}")
    display_df["avg_decision_time"] = display_df["avg_decision_time"].map(
        lambda value: f"{value:.6f}s"
    )

    print(display_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse Gomoku tournament CSV results."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw"),
        help="Directory containing raw tournament CSV files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/processed/summary.csv"),
        help="Path to save the processed summary CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    summary = analyse_results(
        input_dir=args.input_dir,
        output_path=args.output,
    )

    print("Analysis finished")
    print(f"Summary saved to: {args.output}")
    print()
    print_summary_table(summary)