from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


def make_player_name(agent: str, config: str, use_config: bool) -> str:
    agent = str(agent)
    config = str(config) if pd.notna(config) else ""
    if not use_config or config in {"", "nan", "none"}:
        return agent
    compact = config.replace(";", ",")
    return f"{agent}({compact})"


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def actual_scores(winner: str) -> tuple[float, float]:
    winner = str(winner).strip().lower()
    if winner == "black":
        return 1.0, 0.0
    if winner == "white":
        return 0.0, 1.0
    return 0.5, 0.5


def iter_csv_files(patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(path) for path in glob.glob(pattern))
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Elo-style ratings from Gomoku tournament CSV files.")
    parser.add_argument("--input-glob", action="append", required=True, help="Input CSV glob. Can be repeated.")
    parser.add_argument("--output", type=Path, default=Path("results/formal/processed/elo_ratings.csv"), help="Output rating CSV path.")
    parser.add_argument("--pair-output", type=Path, default=Path("results/formal/processed/elo_pair_summary.csv"), help="Output pair summary CSV path.")
    parser.add_argument("--initial-rating", type=float, default=1000.0, help="Initial Elo rating.")
    parser.add_argument("--k-factor", type=float, default=32.0, help="Elo K factor.")
    parser.add_argument("--use-config", action="store_true", help="Treat different agent configs as different Elo players.")
    parser.add_argument("--rule", type=str, default=None, help="Optional rule filter, e.g. standard.")
    args = parser.parse_args()

    files = iter_csv_files(args.input_glob)
    if not files:
        raise FileNotFoundError(f"No CSV files matched: {args.input_glob}")

    ratings: dict[str, float] = defaultdict(lambda: float(args.initial_rating))
    games_played: dict[str, int] = defaultdict(int)
    score_sum: dict[str, float] = defaultdict(float)
    pair_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"games": 0, "a_score": 0.0, "b_score": 0.0, "draws": 0})

    rows_processed = 0

    for file in files:
        df = pd.read_csv(file)
        if args.rule is not None and "rule" in df.columns:
            df = df[df["rule"].astype(str).str.lower() == args.rule.lower()]
        for _, row in df.iterrows():
            black = make_player_name(row.get("black_agent", "black"), row.get("black_agent_config", ""), args.use_config)
            white = make_player_name(row.get("white_agent", "white"), row.get("white_agent_config", ""), args.use_config)
            winner = str(row.get("winner", "draw")).strip().lower()
            black_score, white_score = actual_scores(winner)

            black_expected = expected_score(ratings[black], ratings[white])
            white_expected = 1.0 - black_expected

            ratings[black] += args.k_factor * (black_score - black_expected)
            ratings[white] += args.k_factor * (white_score - white_expected)

            games_played[black] += 1
            games_played[white] += 1
            score_sum[black] += black_score
            score_sum[white] += white_score

            key = tuple(sorted([black, white]))
            a, b = key
            if black == a:
                a_score, b_score = black_score, white_score
            else:
                a_score, b_score = white_score, black_score
            pair_stats[key]["games"] += 1
            pair_stats[key]["a_score"] += a_score
            pair_stats[key]["b_score"] += b_score
            if winner not in {"black", "white"}:
                pair_stats[key]["draws"] += 1

            rows_processed += 1

    rating_rows = []
    for player, rating in sorted(ratings.items(), key=lambda item: item[1], reverse=True):
        gp = games_played[player]
        rating_rows.append({
            "player": player,
            "elo": round(rating, 2),
            "games": gp,
            "score": score_sum[player],
            "score_rate": round(score_sum[player] / gp, 4) if gp else 0.0,
        })

    pair_rows = []
    for (a, b), stats in sorted(pair_stats.items()):
        games = int(stats["games"])
        pair_rows.append({
            "player_a": a,
            "player_b": b,
            "games": games,
            "player_a_score": stats["a_score"],
            "player_b_score": stats["b_score"],
            "draws": int(stats["draws"]),
            "player_a_score_rate": round(stats["a_score"] / games, 4) if games else 0.0,
            "player_b_score_rate": round(stats["b_score"] / games, 4) if games else 0.0,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pair_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rating_rows).to_csv(args.output, index=False)
    pd.DataFrame(pair_rows).to_csv(args.pair_output, index=False)

    print(f"Processed files: {len(files)}")
    print(f"Processed games: {rows_processed}")
    print(f"Saved ratings: {args.output}")
    print(f"Saved pair summary: {args.pair_output}")
    print(pd.DataFrame(rating_rows).to_string(index=False))


if __name__ == "__main__":
    main()
