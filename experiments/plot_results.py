from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow this script to be run directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def plot_first_player_win_rate_by_rule(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot average first-player win rate for each rule.
    """
    grouped = (
        summary_df.groupby("rule", as_index=False)["first_player_win_rate"]
        .mean()
        .sort_values("rule")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(grouped["rule"], grouped["first_player_win_rate"])
    plt.ylim(0, 1)
    plt.xlabel("Opening Rule")
    plt.ylabel("First-player Win Rate")
    plt.title("Average First-player Win Rate by Opening Rule")
    plt.tight_layout()

    output_path = output_dir / "first_player_win_rate_by_rule.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_avg_moves_by_rule(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot average number of moves for each rule.
    """
    grouped = (
        summary_df.groupby("rule", as_index=False)["avg_moves"]
        .mean()
        .sort_values("rule")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(grouped["rule"], grouped["avg_moves"])
    plt.xlabel("Opening Rule")
    plt.ylabel("Average Moves")
    plt.title("Average Number of Moves by Opening Rule")
    plt.tight_layout()

    output_path = output_dir / "avg_moves_by_rule.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_avg_decision_time_by_agent(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot average decision time grouped by black agent.

    This is a secondary metric used to understand computational cost.
    """
    grouped = (
        summary_df.groupby("black_agent", as_index=False)["avg_decision_time_overall"]
        .mean()
        .sort_values("avg_decision_time_overall")
    )

    plt.figure(figsize=(8, 5))
    plt.bar(grouped["black_agent"], grouped["avg_decision_time_overall"])
    plt.xlabel("Agent")
    plt.ylabel("Average Decision Time (seconds)")
    plt.title("Average Decision Time by Agent")
    plt.tight_layout()

    output_path = output_dir / "avg_decision_time_by_agent.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_first_player_win_rate_by_agent_and_rule(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot first-player win rate for same-agent matches only.

    Same-agent matches are the main evidence for rule fairness because AI
    strength is controlled within each match-up.
    """
    same_agent_df = summary_df[
        summary_df["black_agent"] == summary_df["white_agent"]
    ].copy()

    if same_agent_df.empty:
        print("No same-agent rows found. Skipping same-agent plot.")
        return

    same_agent_df["agent_pair"] = (
        same_agent_df["black_agent"] + " vs " + same_agent_df["white_agent"]
    )

    pivot_df = same_agent_df.pivot_table(
        index="agent_pair",
        columns="rule",
        values="first_player_win_rate",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 6))
    pivot_df.plot(kind="bar", ax=plt.gca())
    plt.ylim(0, 1)
    plt.xlabel("Agent Pair")
    plt.ylabel("First-player Win Rate")
    plt.title("First-player Win Rate by Rule for Same-Agent Matches")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = output_dir / "first_player_win_rate_by_agent_and_rule.png"
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

    plot_first_player_win_rate_by_rule(summary_df, output_dir)
    plot_avg_moves_by_rule(summary_df, output_dir)
    plot_avg_decision_time_by_agent(summary_df, output_dir)
    plot_first_player_win_rate_by_agent_and_rule(summary_df, output_dir)

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