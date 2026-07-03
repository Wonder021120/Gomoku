"""
Swap2 opening utilities for Gomoku.

This module keeps the original project interface:

    result, black_agent, white_agent = run_swap2_opening(...)

but makes the fixed Swap2 opening template depend on board_size.

For 15x15:
    B(7, 7), W(7, 8), B(8, 7)
    optional add-two: W(6, 7), B(7, 6)

For 9x9:
    B(4, 4), W(4, 5), B(5, 4)
    optional add-two: W(3, 4), B(4, 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


try:
    from .board import Board, BLACK as MODULE_BLACK, WHITE as MODULE_WHITE
except ImportError:  # pragma: no cover
    Board = None
    MODULE_BLACK = 1
    MODULE_WHITE = -1


BLACK = getattr(Board, "BLACK", MODULE_BLACK)
WHITE = getattr(Board, "WHITE", MODULE_WHITE)

Move = Tuple[int, int]
OpeningMove = Tuple[int, Move]


class Swap2Choice(str, Enum):
    """Internal simplified Swap2 choices."""

    CHOOSER_CHOOSE_BLACK = "chooser_choose_black"
    CHOOSER_CHOOSE_WHITE = "chooser_choose_white"
    ADD_TWO_THEN_SLICER_CHOOSE_BLACK = "add_two_then_slicer_choose_black"
    ADD_TWO_THEN_SLICER_CHOOSE_WHITE = "add_two_then_slicer_choose_white"


@dataclass
class Swap2OpeningResult:
    """Result object returned by run_swap2_opening."""

    choice: str
    opening_template: str
    opening_moves: List[OpeningMove]
    opening_moves_text: str
    board_size: int
    next_player: int = WHITE

    @property
    def num_opening_stones(self) -> int:
        return len(self.opening_moves)

    @property
    def swap2_choice(self) -> str:
        return self.choice

    def to_dict(self) -> Dict[str, Any]:
        return {
            "choice": self.choice,
            "swap2_choice": self.swap2_choice,
            "opening_template": self.opening_template,
            "opening_moves": self.opening_moves,
            "opening_moves_text": self.opening_moves_text,
            "board_size": self.board_size,
            "next_player": self.next_player,
            "num_opening_stones": self.num_opening_stones,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def items(self):
        return self.to_dict().items()


def get_swap2_initial_moves(board_size: int) -> List[OpeningMove]:
    """Return the first three fixed central Swap2 stones as (colour, move)."""

    _validate_board_size_for_swap2(board_size)
    center = board_size // 2

    return [
        (BLACK, (center, center)),
        (WHITE, (center, center + 1)),
        (BLACK, (center + 1, center)),
    ]


def get_swap2_additional_moves(board_size: int) -> List[OpeningMove]:
    """Return the optional two extra stones for the add-two branch."""

    _validate_board_size_for_swap2(board_size)
    center = board_size // 2

    return [
        (WHITE, (center - 1, center)),
        (BLACK, (center, center - 1)),
    ]


def get_swap2_opening_moves(board_size: int, add_two: bool = False) -> List[OpeningMove]:
    """Return the full fixed Swap2 opening template."""

    moves = get_swap2_initial_moves(board_size)
    if add_two:
        moves.extend(get_swap2_additional_moves(board_size))
    return moves


def run_swap2_opening(
    game: Any,
    slicer_agent: Any,
    chooser_agent: Any,
    choice_threshold: float = 0.5,
    choice: Optional[str | Swap2Choice] = None,
    **kwargs: Any,
) -> Tuple[Swap2OpeningResult, Any, Any]:
    """Run the simplified Swap2 opening."""

    board_size = _get_board_size(game)
    _ensure_empty_board(game)

    resolved_choice = _resolve_choice(choice=choice, choice_threshold=choice_threshold, kwargs=kwargs)
    add_two = resolved_choice in {
        Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_BLACK,
        Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_WHITE,
    }

    opening_moves = get_swap2_opening_moves(board_size=board_size, add_two=add_two)

    for colour, move in opening_moves:
        _place_stone(game=game, move=move, player=colour)

    if hasattr(game, "current_player"):
        game.current_player = WHITE

    black_agent, white_agent = _assign_agents(
        resolved_choice=resolved_choice,
        slicer_agent=slicer_agent,
        chooser_agent=chooser_agent,
    )

    opening_template = "central_5_stone_template" if add_two else "central_3_stone_template"

    result = Swap2OpeningResult(
        choice=resolved_choice.value,
        opening_template=opening_template,
        opening_moves=opening_moves,
        opening_moves_text=_format_opening_moves_text(opening_moves),
        board_size=board_size,
        next_player=WHITE,
    )

    return result, black_agent, white_agent


def apply_swap2_opening(
    game: Any,
    choice: Optional[str | Swap2Choice] = None,
) -> Swap2OpeningResult:
    """Apply Swap2 opening and return only the result object."""

    class _PlaceholderAgent:
        pass

    result, _, _ = run_swap2_opening(
        game=game,
        slicer_agent=_PlaceholderAgent(),
        chooser_agent=_PlaceholderAgent(),
        choice=choice,
    )
    return result


def setup_swap2_opening(
    game: Any,
    choice: Optional[str | Swap2Choice] = None,
) -> Swap2OpeningResult:
    """Backward-compatible alias."""

    return apply_swap2_opening(game=game, choice=choice)


def initialise_swap2_opening(
    game: Any,
    choice: Optional[str | Swap2Choice] = None,
) -> Swap2OpeningResult:
    """British spelling alias."""

    return apply_swap2_opening(game=game, choice=choice)


def initialize_swap2_opening(
    game: Any,
    choice: Optional[str | Swap2Choice] = None,
) -> Swap2OpeningResult:
    """American spelling alias."""

    return apply_swap2_opening(game=game, choice=choice)


def apply_swap2(
    game: Any,
    choice: Optional[str | Swap2Choice] = None,
) -> Swap2OpeningResult:
    """Short alias."""

    return apply_swap2_opening(game=game, choice=choice)


def _resolve_choice(
    choice: Optional[str | Swap2Choice],
    choice_threshold: float,
    kwargs: Dict[str, Any],
) -> Swap2Choice:
    """Resolve the old project choice strings into Swap2Choice."""

    if choice is None:
        choice = kwargs.get("swap2_choice", kwargs.get("second_player_choice"))

    if isinstance(choice, Swap2Choice):
        return choice

    if choice is not None:
        value = str(choice).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "chooser_choose_black": Swap2Choice.CHOOSER_CHOOSE_BLACK,
            "choose_black": Swap2Choice.CHOOSER_CHOOSE_BLACK,
            "swap": Swap2Choice.CHOOSER_CHOOSE_BLACK,
            "black": Swap2Choice.CHOOSER_CHOOSE_BLACK,
            "chooser_choose_white": Swap2Choice.CHOOSER_CHOOSE_WHITE,
            "choose_white": Swap2Choice.CHOOSER_CHOOSE_WHITE,
            "stay": Swap2Choice.CHOOSER_CHOOSE_WHITE,
            "white": Swap2Choice.CHOOSER_CHOOSE_WHITE,
            "add_two_then_slicer_choose_black": Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_BLACK,
            "add_two_black": Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_BLACK,
            "add_two_then_slicer_choose_white": Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_WHITE,
            "add_two_white": Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_WHITE,
            "add_two": Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_BLACK,
            "add2": Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_BLACK,
        }

        if value not in aliases:
            valid = ", ".join(sorted(aliases))
            raise ValueError(f"Unknown Swap2 choice: {choice!r}. Valid choices include: {valid}")

        return aliases[value]

    # Existing tests rely on this deterministic behaviour.
    if choice_threshold >= 1.0:
        return Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_BLACK

    return Swap2Choice.CHOOSER_CHOOSE_WHITE


def _assign_agents(resolved_choice: Swap2Choice, slicer_agent: Any, chooser_agent: Any) -> Tuple[Any, Any]:
    """Return black_agent, white_agent for the simplified Swap2 result."""

    if resolved_choice == Swap2Choice.CHOOSER_CHOOSE_BLACK:
        return chooser_agent, slicer_agent

    if resolved_choice == Swap2Choice.CHOOSER_CHOOSE_WHITE:
        return slicer_agent, chooser_agent

    if resolved_choice == Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_BLACK:
        return slicer_agent, chooser_agent

    if resolved_choice == Swap2Choice.ADD_TWO_THEN_SLICER_CHOOSE_WHITE:
        return chooser_agent, slicer_agent

    raise ValueError(f"Unhandled Swap2 choice: {resolved_choice}")


def _get_board_size(game: Any) -> int:
    """Extract board size from a Game-like object."""

    if hasattr(game, "board") and hasattr(game.board, "size"):
        return int(game.board.size)

    if hasattr(game, "board_size"):
        return int(game.board_size)

    raise AttributeError("Cannot determine board size from game. Expected game.board.size.")


def _ensure_empty_board(game: Any) -> None:
    """Swap2 opening should only be applied to an empty board."""

    board = getattr(game, "board", None)
    if board is None:
        raise AttributeError("game must have a board attribute.")

    move_history = getattr(board, "move_history", None)
    if move_history is not None and len(move_history) > 0:
        raise ValueError("Swap2 opening can only be applied to an empty board.")

    grid = getattr(board, "grid", None)
    if grid is not None:
        try:
            has_stones = bool((grid != 0).any())
        except Exception:
            has_stones = any(any(cell != 0 for cell in row) for row in grid)

        if has_stones:
            raise ValueError("Swap2 opening can only be applied to an empty board.")


def _place_stone(game: Any, move: Move, player: int) -> None:
    """Place an opening stone directly on the board."""

    board = getattr(game, "board", None)
    if board is None:
        raise AttributeError("game must have a board attribute.")

    if not hasattr(board, "place_stone"):
        raise AttributeError("game.board must provide place_stone(row, col, player).")

    row, col = move
    result = board.place_stone(row, col, player)

    if result is False:
        raise ValueError(f"Failed to place Swap2 opening stone at {move} for player {player}.")


def _format_opening_moves_text(opening_moves: Iterable[OpeningMove]) -> str:
    """Format opening moves as a readable string."""

    parts = []
    for colour, (row, col) in opening_moves:
        symbol = "B" if colour == BLACK else "W"
        parts.append(f"{symbol}({row}, {col})")
    return ", ".join(parts)


def _validate_board_size_for_swap2(board_size: int) -> None:
    """Validate board size for this fixed central template."""

    if board_size < 5:
        raise ValueError("Swap2 opening requires board_size >= 5.")

    if board_size % 2 == 0:
        raise ValueError(
            "This Swap2 template expects an odd board size so that the board has a clear centre."
        )


__all__ = [
    "Swap2Choice",
    "Swap2OpeningResult",
    "get_swap2_initial_moves",
    "get_swap2_additional_moves",
    "get_swap2_opening_moves",
    "run_swap2_opening",
    "apply_swap2_opening",
    "setup_swap2_opening",
    "initialise_swap2_opening",
    "initialize_swap2_opening",
    "apply_swap2",
]
