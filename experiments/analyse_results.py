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

    rule = _get_first_value(df, "rule", "unknown")
    board_size = _get_first_value(df, "board_size", "unknown")
    black_agent = _get_first_value(df, "black_agent", "unknown")
    white_agent = _get_first_value(df, "white_agent", "unknown")
    black_agent_config = _get_first_value(df, "black_agent_config", "unknown")
    white_agent_config = _get_first_value(df, "white_agent_config", "unknown")

    swap2_choices = "none"
    swap2_opening_template = "none"

    if "swap2_choice" in df.columns:
        swap2_choices = _format_value_counts(df["swap2_choice"])

    if "swap2_opening_template" in df.columns:
        swap2_opening_template = _format_value_counts(df["swap2_opening_template"])

    return {
        "source_file": csv_path.name,
        "rule": rule,
        "board_size": board_size,
        "black_agent": black_agent,
        "white_agent": white_agent,
        "black_agent_config": black_agent_config,
        "white_agent_config": white_agent_config,
        "total_games": total_games,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "black_win_rate": black_wins / total_games,
        "white_win_rate": white_wins / total_games,
        "draw_rate": draws / total_games,
        "first_player_wins": first_player_wins,
        "first_player_win_rate": first_player_wins / total_games,
        "avg_moves": float(df["moves"].mean()),
        "min_moves": int(df["moves"].min()),
        "max_moves": int(df["moves"].max()),
        "avg_black_decision_time": float(df["black_avg_time"].mean()),
        "avg_white_decision_time": float(df["white_avg_time"].mean()),
        "avg_decision_time_overall": float(
            (df["black_avg_time"].mean() + df["white_avg_time"].mean()) / 2
        ),
        "swap2_choices": swap2_choices,
        "swap2_opening_template": swap2_opening_template,
    }


def analyse_results(input_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Analyse all raw tournament CSV files in input_dir.
    """
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    summaries = []

    for csv_path in csv_files:
        try:
            summary = summarise_file(csv_path)
            summaries.append(summary)
        except Exception as error:
            print(f"Skipping {csv_path}: {error}")

    if not summaries:
        raise ValueError("No valid CSV files could be analysed.")

    summary_df = pd.DataFrame(summaries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8")

    return summary_df


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """
    Print a readable summary table to the terminal.
    """
    display_columns = [
        "source_file",
        "rule",
        "black_agent",
        "white_agent",
        "total_games",
        "black_win_rate",
        "white_win_rate",
        "draw_rate",
        "first_player_win_rate",
        "avg_moves",
        "avg_decision_time_overall",
        "swap2_choices",
    ]

    available_columns = [
        column for column in display_columns if column in summary_df.columns
    ]

    printable_df = summary_df[available_columns].copy()

    rate_columns = [
        "black_win_rate",
        "white_win_rate",
        "draw_rate",
        "first_player_win_rate",
    ]

    for column in rate_columns:
        if column in printable_df.columns:
            printable_df[column] = printable_df[column].map(lambda value: f"{value:.2%}")

    if "avg_moves" in printable_df.columns:
        printable_df["avg_moves"] = printable_df["avg_moves"].map(lambda value: f"{value:.2f}")

    if "avg_decision_time_overall" in printable_df.columns:
        printable_df["avg_decision_time_overall"] = printable_df[
            "avg_decision_time_overall"
        ].map(lambda value: f"{value:.6f}s")

    print(printable_df.to_string(index=False))


def _get_first_value(df: pd.DataFrame, column: str, default: object) -> object:
    """
    Safely get the first value of a column.
    """
    if column not in df.columns:
        return default

    if df[column].empty:
        return default

    return df[column].iloc[0]


def _format_value_counts(series: pd.Series) -> str:
    """
    Format value counts as a compact string.
    """
    counts = series.fillna("none").value_counts()

    return "; ".join(f"{key}:{value}" for key, value in counts.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse raw Gomoku tournament result CSV files."
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
        help="Path to save the summary CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    summary = analyse_results(
        input_dir=args.input_dir,
        output_path=args.output,
    )

    print_summary_table(summary)
    print()
    print(f"Summary saved to: {args.output}")