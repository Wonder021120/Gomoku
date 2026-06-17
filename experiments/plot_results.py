from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def get_same_agent_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only same-agent matches.

    Same-agent matches are the main evidence for rule fairness because the
    playing strength is controlled within each comparison.
    """
    return summary_df[
        summary_df["black_agent"] == summary_df["white_agent"]
    ].copy()


def plot_same_agent_first_player_win_rate(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot first-player win rate by rule for same-agent matches only.
    """
    same_agent_df = get_same_agent_df(summary_df)

    if same_agent_df.empty:
        print("No same-agent rows found. Skipping same-agent first-player plot.")
        return

    same_agent_df["agent"] = same_agent_df["black_agent"]

    pivot_df = same_agent_df.pivot_table(
        index="agent",
        columns="rule",
        values="first_player_win_rate",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 6))
    pivot_df.plot(kind="bar", ax=plt.gca())
    plt.ylim(0, 1)
    plt.xlabel("AI Agent")
    plt.ylabel("First-player Win Rate")
    plt.title("First-player Win Rate by Rule for Same-Agent Matches")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = output_dir / "same_agent_first_player_win_rate_by_rule.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_same_agent_black_white_draw_rates(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot average black, white, and draw rates by rule for same-agent matches.
    """
    same_agent_df = get_same_agent_df(summary_df)

    if same_agent_df.empty:
        print("No same-agent rows found. Skipping win/draw rate plot.")
        return

    grouped = (
        same_agent_df.groupby("rule", as_index=False)[
            ["black_win_rate", "white_win_rate", "draw_rate"]
        ]
        .mean()
        .sort_values("rule")
    )

    grouped = grouped.set_index("rule")

    plt.figure(figsize=(10, 6))
    grouped.plot(kind="bar", ax=plt.gca())
    plt.ylim(0, 1)
    plt.xlabel("Opening Rule")
    plt.ylabel("Rate")
    plt.title("Black / White / Draw Rates by Rule for Same-Agent Matches")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = output_dir / "same_agent_black_white_draw_rates_by_rule.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_same_agent_average_moves_by_rule(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot average moves by rule for same-agent matches only.
    """
    same_agent_df = get_same_agent_df(summary_df)

    if same_agent_df.empty:
        print("No same-agent rows found. Skipping same-agent average moves plot.")
        return

    grouped = (
        same_agent_df.groupby("rule", as_index=False)["avg_moves"]
        .mean()
        .sort_values("rule")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(grouped["rule"], grouped["avg_moves"])
    plt.xlabel("Opening Rule")
    plt.ylabel("Average Moves")
    plt.title("Average Moves by Rule for Same-Agent Matches")
    plt.tight_layout()

    output_path = output_dir / "same_agent_avg_moves_by_rule.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_agent_decision_time(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot average decision time by agent.

    This combines black-side and white-side timing records so that the result
    represents the agent rather than only the black_agent column.
    """
    black_time_df = summary_df[["black_agent", "avg_black_decision_time"]].copy()
    black_time_df.columns = ["agent", "avg_decision_time"]

    white_time_df = summary_df[["white_agent", "avg_white_decision_time"]].copy()
    white_time_df.columns = ["agent", "avg_decision_time"]

    combined_df = pd.concat([black_time_df, white_time_df], ignore_index=True)

    grouped = (
        combined_df.groupby("agent", as_index=False)["avg_decision_time"]
        .mean()
        .sort_values("avg_decision_time")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(grouped["agent"], grouped["avg_decision_time"])
    plt.xlabel("AI Agent")
    plt.ylabel("Average Decision Time (seconds)")
    plt.title("Average Decision Time by AI Agent")
    plt.tight_layout()

    output_path = output_dir / "agent_average_decision_time.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_swap2_choice_frequency(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot Swap2 choice frequency from summary rows.

    The summary CSV stores compact value-count strings such as:
    add_two_then_slicer_choose_black:3
    """
    if "swap2_choices" not in summary_df.columns:
        print("No swap2_choices column found. Skipping Swap2 choice plot.")
        return

    swap2_df = summary_df[summary_df["rule"] == "swap2"].copy()

    if swap2_df.empty:
        print("No swap2 rows found. Skipping Swap2 choice plot.")
        return

    counts: dict[str, int] = {}

    for value in swap2_df["swap2_choices"].dropna():
        if value == "none":
            continue

        parts = str(value).split(";")

        for part in parts:
            part = part.strip()

            if not part or ":" not in part:
                continue

            key, count_text = part.rsplit(":", 1)

            try:
                count = int(count_text)
            except ValueError:
                continue

            counts[key] = counts.get(key, 0) + count

    if not counts:
        print("No valid Swap2 choice counts found. Skipping Swap2 choice plot.")
        return

    choice_df = pd.DataFrame(
        {
            "choice": list(counts.keys()),
            "count": list(counts.values()),
        }
    ).sort_values("choice")

    plt.figure(figsize=(10, 5))
    plt.bar(choice_df["choice"], choice_df["count"])
    plt.xlabel("Swap2 Choice")
    plt.ylabel("Count")
    plt.title("Swap2 Choice Frequency")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = output_dir / "swap2_choice_frequency.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_plots(summary_path: Path, output_dir: Path) -> None:
    """
    Generate result plots from summary CSV.
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)

    if summary_df.empty:
        raise ValueError(f"Summary CSV is empty: {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_same_agent_first_player_win_rate(summary_df, output_dir)
    plot_same_agent_black_white_draw_rates(summary_df, output_dir)
    plot_same_agent_average_moves_by_rule(summary_df, output_dir)
    plot_agent_decision_time(summary_df, output_dir)
    plot_swap2_choice_frequency(summary_df, output_dir)

    print(f"Figures saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots from Gomoku experiment summary results."
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/processed/summary.csv"),
        help="Path to summary CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to save generated figures.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate_plots(
        summary_path=args.summary,
        output_dir=args.output_dir,
    )