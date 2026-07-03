"""
Run a simplified paper-style NN-MCTS training loop for 9x9 Gomoku.

This script implements the project reproduction version of the paper's
best-player/current-player loop:

1. Use the current best player to generate NN-MCTS self-play data.
2. Train a current player from that self-play data.
3. Evaluate current player against best player.
4. Update best player only if current player wins enough games.
5. Save a JSON/CSV summary for each round.

It reuses:
- experiments/generate_self_play_data.py
- experiments/train_nn.py

The neural-network path is:
- board_size = 9
- model_variant = paper_9x9
- 4-channel state encoding
- MCTS visit-distribution policy targets
- symmetry augmentation during training
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys

import torch
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gomoku.board import Board
from gomoku.game import Game, GameStatus
from gomoku.nn_mcts_agent import NNMCTSAgent


@dataclass
class EvaluationResult:
    games: int
    current_wins: int
    best_wins: int
    draws: int
    current_win_rate: float
    best_win_rate: float
    draw_rate: float
    accepted: bool


@dataclass
class RoundSummary:
    round_index: int
    best_checkpoint_before: Optional[str]
    self_play_data_path: str
    current_checkpoint_path: str
    current_best_checkpoint_path: str
    best_checkpoint_after: Optional[str]
    accepted: bool
    evaluation: Dict
    config: Dict


def run_command(command: List[str]) -> None:
    print()
    print("Running command:")
    print("  " + " ".join(command))
    print()
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def status_to_winner(status: GameStatus) -> int:
    if status == GameStatus.BLACK_WIN:
        return Board.BLACK
    if status == GameStatus.WHITE_WIN:
        return Board.WHITE
    return Board.EMPTY


def create_nn_mcts_agent(
    board_size: int,
    checkpoint_path: Optional[Path],
    simulations: int,
    exploration_weight: float,
    candidate_radius: int,
    device: str,
    seed: int,
) -> NNMCTSAgent:
    return NNMCTSAgent(
        board_size=board_size,
        simulations=simulations,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        exploration_weight=exploration_weight,
        candidate_radius=candidate_radius,
        device=device,
        seed=seed,
        model_variant="paper_9x9",
    )


def play_evaluation_game(
    current_agent: NNMCTSAgent,
    best_agent: NNMCTSAgent,
    current_as_black: bool,
    board_size: int,
    rule: str,
    max_moves: int,
) -> int:
    """
    Return:
        1  current wins
        -1 best wins
        0  draw
    """

    game = Game(board_size=board_size, rule_name=rule)

    while game.status == GameStatus.ONGOING and len(game.board.move_history) < max_moves:
        if game.current_player == Board.BLACK:
            agent = current_agent if current_as_black else best_agent
        else:
            agent = best_agent if current_as_black else current_agent

        move = agent.select_move(game)
        game.play_move(move)

    winner = status_to_winner(game.status)

    if winner == Board.EMPTY:
        return 0
    if current_as_black and winner == Board.BLACK:
        return 1
    if (not current_as_black) and winner == Board.WHITE:
        return 1
    return -1


def evaluate_current_vs_best(
    current_checkpoint: Path,
    best_checkpoint: Optional[Path],
    games: int,
    board_size: int,
    rule: str,
    max_moves: int,
    simulations: int,
    exploration_weight: float,
    candidate_radius: int,
    device: str,
    seed: int,
) -> EvaluationResult:
    current_wins = 0
    best_wins = 0
    draws = 0

    for game_index in range(games):
        current_as_black = game_index % 2 == 0
        game_seed = seed + game_index

        current_agent = create_nn_mcts_agent(
            board_size=board_size,
            checkpoint_path=current_checkpoint,
            simulations=simulations,
            exploration_weight=exploration_weight,
            candidate_radius=candidate_radius,
            device=device,
            seed=game_seed,
        )

        best_agent = create_nn_mcts_agent(
            board_size=board_size,
            checkpoint_path=best_checkpoint,
            simulations=simulations,
            exploration_weight=exploration_weight,
            candidate_radius=candidate_radius,
            device=device,
            seed=game_seed + 100_000,
        )

        result = play_evaluation_game(
            current_agent=current_agent,
            best_agent=best_agent,
            current_as_black=current_as_black,
            board_size=board_size,
            rule=rule,
            max_moves=max_moves,
        )

        if result > 0:
            current_wins += 1
            result_text = "current"
        elif result < 0:
            best_wins += 1
            result_text = "best"
        else:
            draws += 1
            result_text = "draw"

        colour_text = "black" if current_as_black else "white"
        print(
            f"Evaluation game {game_index + 1}/{games}: "
            f"current_as={colour_text}, result={result_text}"
        )

    return EvaluationResult(
        games=games,
        current_wins=current_wins,
        best_wins=best_wins,
        draws=draws,
        current_win_rate=current_wins / games,
        best_win_rate=best_wins / games,
        draw_rate=draws / games,
        accepted=False,
    )


def save_round_summary_json(summary: RoundSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def append_round_summary_csv(summary: RoundSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "round_index": summary.round_index,
        "best_checkpoint_before": summary.best_checkpoint_before,
        "self_play_data_path": summary.self_play_data_path,
        "current_checkpoint_path": summary.current_checkpoint_path,
        "current_best_checkpoint_path": summary.current_best_checkpoint_path,
        "best_checkpoint_after": summary.best_checkpoint_after,
        "accepted": summary.accepted,
        "evaluation_games": summary.evaluation["games"],
        "current_wins": summary.evaluation["current_wins"],
        "best_wins": summary.evaluation["best_wins"],
        "draws": summary.evaluation["draws"],
        "current_win_rate": summary.evaluation["current_win_rate"],
        "best_win_rate": summary.evaluation["best_win_rate"],
        "draw_rate": summary.evaluation["draw_rate"],
    }

    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_generate_self_play_command(
    args: argparse.Namespace,
    round_index: int,
    best_checkpoint: Optional[Path],
    data_path: Path,
) -> List[str]:
    command = [
        sys.executable,
        "experiments/generate_self_play_data.py",
        "--games",
        str(args.self_play_games),
        "--board-size",
        str(args.board_size),
        "--agent",
        "nn_mcts",
        "--model-variant",
        "paper_9x9",
        "--rule",
        args.rule,
        "--max-moves",
        str(args.max_moves),
        "--nn-mcts-simulations",
        str(args.self_play_simulations),
        "--nn-mcts-exploration-weight",
        str(args.exploration_weight),
        "--nn-mcts-candidate-radius",
        str(args.candidate_radius),
        "--nn-mcts-device",
        args.device,
        "--policy-target-mode",
        "visit_distribution",
        "--temperature-moves",
        str(args.temperature_moves),
        "--temperature",
        str(args.temperature),
        "--policy-temperature",
        str(args.policy_temperature),
        "--seed",
        str(args.seed + round_index * 10_000),
        "--output",
        str(data_path),
    ]

    if best_checkpoint is not None:
        command.extend(["--nn-mcts-checkpoint", str(best_checkpoint)])

    return command


def get_checkpoint_epoch(checkpoint_path: Optional[Path]) -> int:
    """Read the epoch number stored in a checkpoint."""

    if checkpoint_path is None:
        return 0

    if not checkpoint_path.exists():
        return 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        return int(checkpoint.get("epoch", 0))

    return 0


def get_target_epochs_for_round(
    args: argparse.Namespace,
    best_checkpoint: Optional[Path],
) -> int:
    """
    Convert per-round epochs into train_nn.py's absolute --epochs value.

    train_nn.py resumes from checkpoint epoch + 1.
    So if the best checkpoint is at epoch 5 and we want 5 more epochs,
    this loop must call train_nn.py with --epochs 10, not --epochs 5.
    """

    if best_checkpoint is not None and args.resume_from_best:
        return get_checkpoint_epoch(best_checkpoint) + args.epochs

    return args.epochs


def build_train_command(
    args: argparse.Namespace,
    data_path: Path,
    current_checkpoint: Path,
    current_best_checkpoint: Path,
    best_checkpoint: Optional[Path],
) -> List[str]:
    target_epochs = get_target_epochs_for_round(
        args=args,
        best_checkpoint=best_checkpoint,
    )

    command = [
        sys.executable,
        "experiments/train_nn.py",
        "--data",
        str(data_path),
        "--model-variant",
        "paper_9x9",
        "--augment-symmetries",
        "--epochs",
        str(target_epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--value-loss-weight",
        str(args.value_loss_weight),
        "--validation-split",
        str(args.validation_split),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--output",
        str(current_checkpoint),
        "--best-output",
        str(current_best_checkpoint),
    ]

    if best_checkpoint is not None and args.resume_from_best:
        command.extend(["--resume-checkpoint", str(best_checkpoint)])

    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-style 9x9 NN-MCTS best/current training loop."
    )

    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--rule", choices=["standard", "pro"], default="standard")
    parser.add_argument("--max-moves", type=int, default=81)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--work-dir", type=Path, default=Path("experiments/nn_mcts_training_loop"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/self_play/nn_mcts_training_loop"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/nn_mcts_training_loop"))

    parser.add_argument(
        "--initial-best-checkpoint",
        type=Path,
        default=None,
        help="Optional starting best checkpoint. If omitted, round 1 uses an untrained paper_9x9 network.",
    )

    parser.add_argument(
        "--resume-from-best",
        action="store_true",
        help="Resume current training from previous best checkpoint when available.",
    )

    parser.add_argument(
        "--accept-first-round",
        action="store_true",
        default=True,
        help="Bootstrap by accepting round 1 if no initial best checkpoint was supplied.",
    )
    parser.add_argument(
        "--no-accept-first-round",
        dest="accept_first_round",
        action="store_false",
    )

    parser.add_argument("--self-play-games", type=int, default=20)
    parser.add_argument("--self-play-simulations", type=int, default=100)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--validation-split", type=float, default=0.2)

    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-simulations", type=int, default=100)
    parser.add_argument("--accept-win-rate", type=float, default=0.55)

    parser.add_argument("--exploration-weight", type=float, default=1.5)
    parser.add_argument("--candidate-radius", type=int, default=2)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.board_size != 9:
        raise ValueError("This reproduction loop currently requires --board-size 9.")

    positive_int_fields = [
        "rounds",
        "max_moves",
        "self_play_games",
        "self_play_simulations",
        "epochs",
        "batch_size",
        "eval_games",
        "eval_simulations",
        "candidate_radius",
    ]

    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")

    if not 0.0 <= args.validation_split < 1.0:
        raise ValueError("--validation-split must be in [0, 1).")
    if not 0.0 <= args.accept_win_rate <= 1.0:
        raise ValueError("--accept-win-rate must be between 0 and 1.")
    if args.temperature_moves < 0:
        raise ValueError("--temperature-moves must be non-negative.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.value_loss_weight < 0.0:
        raise ValueError("--value-loss-weight must be non-negative.")
    if args.initial_best_checkpoint is not None and not args.initial_best_checkpoint.exists():
        raise FileNotFoundError(f"Initial best checkpoint not found: {args.initial_best_checkpoint}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.work_dir / "round_summaries.csv"
    best_checkpoint: Optional[Path] = args.initial_best_checkpoint

    print("Paper-style NN-MCTS training loop configuration:")
    print(f"  rounds: {args.rounds}")
    print(f"  board_size: {args.board_size}")
    print(f"  rule: {args.rule}")
    print("  model_variant: paper_9x9")
    print(f"  initial_best_checkpoint: {best_checkpoint}")
    print(f"  self_play_games: {args.self_play_games}")
    print(f"  self_play_simulations: {args.self_play_simulations}")
    print(f"  epochs: {args.epochs}")
    print(f"  eval_games: {args.eval_games}")
    print(f"  eval_simulations: {args.eval_simulations}")
    print(f"  accept_win_rate: {args.accept_win_rate}")
    print(f"  accept_first_round: {args.accept_first_round}")
    print(f"  resume_from_best: {args.resume_from_best}")
    print()

    for round_index in range(1, args.rounds + 1):
        print("=" * 80)
        print(f"Round {round_index}/{args.rounds}")
        print("=" * 80)

        best_before = best_checkpoint

        data_path = args.data_dir / f"round_{round_index:03d}_self_play.npz"
        current_checkpoint = args.checkpoint_dir / f"round_{round_index:03d}_current_final.pt"
        current_best_checkpoint = args.checkpoint_dir / f"round_{round_index:03d}_current_best.pt"
        accepted_best_checkpoint = args.checkpoint_dir / "best.pt"

        run_command(
            build_generate_self_play_command(
                args=args,
                round_index=round_index,
                best_checkpoint=best_checkpoint,
                data_path=data_path,
            )
        )

        run_command(
            build_train_command(
                args=args,
                data_path=data_path,
                current_checkpoint=current_checkpoint,
                current_best_checkpoint=current_best_checkpoint,
                best_checkpoint=best_checkpoint,
            )
        )

        checkpoint_for_evaluation = current_best_checkpoint if current_best_checkpoint.exists() else current_checkpoint

        evaluation = evaluate_current_vs_best(
            current_checkpoint=checkpoint_for_evaluation,
            best_checkpoint=best_before,
            games=args.eval_games,
            board_size=args.board_size,
            rule=args.rule,
            max_moves=args.max_moves,
            simulations=args.eval_simulations,
            exploration_weight=args.exploration_weight,
            candidate_radius=args.candidate_radius,
            device=args.device,
            seed=args.seed + round_index * 20_000,
        )

        accepted = evaluation.current_win_rate >= args.accept_win_rate

        if round_index == 1 and best_before is None and args.accept_first_round:
            accepted = True
            print(
                "Round 1 bootstrap: accepting current checkpoint as best "
                "because no initial best checkpoint was supplied."
            )

        evaluation.accepted = accepted

        if accepted:
            shutil.copy2(checkpoint_for_evaluation, accepted_best_checkpoint)
            best_checkpoint = accepted_best_checkpoint
            print(f"Accepted current player as new best: {best_checkpoint}")
        else:
            best_checkpoint = best_before
            print("Rejected current player; keeping previous best checkpoint.")

        summary = RoundSummary(
            round_index=round_index,
            best_checkpoint_before=str(best_before) if best_before else None,
            self_play_data_path=str(data_path),
            current_checkpoint_path=str(current_checkpoint),
            current_best_checkpoint_path=str(current_best_checkpoint),
            best_checkpoint_after=str(best_checkpoint) if best_checkpoint else None,
            accepted=accepted,
            evaluation=asdict(evaluation),
            config={
                "board_size": args.board_size,
                "rule": args.rule,
                "model_variant": "paper_9x9",
                "self_play_games": args.self_play_games,
                "self_play_simulations": args.self_play_simulations,
                "temperature_moves": args.temperature_moves,
                "temperature": args.temperature,
                "policy_temperature": args.policy_temperature,
                "epochs_per_round": args.epochs,
                "target_epochs_this_round": get_target_epochs_for_round(args, best_before),
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "eval_games": args.eval_games,
                "eval_simulations": args.eval_simulations,
                "accept_win_rate": args.accept_win_rate,
                "resume_from_best": args.resume_from_best,
            },
        )

        summary_json = args.work_dir / f"round_{round_index:03d}_summary.json"
        save_round_summary_json(summary, summary_json)
        append_round_summary_csv(summary, summary_csv)

        print()
        print("Round summary:")
        print(json.dumps(asdict(summary), indent=2))
        print()

    print("=" * 80)
    print("Training loop finished.")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Summary CSV: {summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
