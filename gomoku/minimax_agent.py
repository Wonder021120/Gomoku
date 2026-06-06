from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from gomoku.agents import BaseAgent
from gomoku.board import Board, Move
from gomoku.game import Game, GameStatus


WIN_SCORE = 1_000_000


@dataclass
class MinimaxAgent(BaseAgent):
    """
    A Minimax agent with Alpha-Beta pruning.

    This is the first classical AI baseline for the project.
    """

    depth: int = 2
    candidate_radius: int = 2
    seed: Optional[int] = None
    name: str = "minimax"

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("Search depth must be positive.")
        if self.candidate_radius <= 0:
            raise ValueError("Candidate radius must be positive.")
        self.rng = random.Random(self.seed)

    def select_move(self, game: Game) -> Move:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        # On an empty board, play the centre. This reduces the opening branching factor.
        if len(game.board.move_history) == 0:
            centre = game.board.size // 2
            return (centre, centre)

        root_player = game.current_player
        best_score = -math.inf
        best_moves: list[Move] = []

        candidate_moves = self._get_candidate_moves(game)

        for move in candidate_moves:
            simulated_game = game.copy()
            simulated_game.play_move(move)

            score = self._minimax(
                game=simulated_game,
                depth=self.depth - 1,
                alpha=-math.inf,
                beta=math.inf,
                root_player=root_player,
            )

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return self.rng.choice(best_moves)

    def _minimax(
        self,
        game: Game,
        depth: int,
        alpha: float,
        beta: float,
        root_player: int,
    ) -> float:
        """
        Minimax search with alpha-beta pruning.
        """
        if game.is_over() or depth == 0:
            return self._evaluate_game(game, root_player)

        candidate_moves = self._get_candidate_moves(game)

        if not candidate_moves:
            return self._evaluate_game(game, root_player)

        maximizing = game.current_player == root_player

        if maximizing:
            value = -math.inf

            for move in candidate_moves:
                simulated_game = game.copy()
                simulated_game.play_move(move)

                value = max(
                    value,
                    self._minimax(
                        game=simulated_game,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                        root_player=root_player,
                    ),
                )

                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return value

        value = math.inf

        for move in candidate_moves:
            simulated_game = game.copy()
            simulated_game.play_move(move)

            value = min(
                value,
                self._minimax(
                    game=simulated_game,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    root_player=root_player,
                ),
            )

            beta = min(beta, value)

            if alpha >= beta:
                break

        return value

    def _evaluate_game(self, game: Game, player: int) -> float:
        """
        Evaluate the current game state from the perspective of player.
        """
        winner = game.get_winner()

        if winner == player:
            return WIN_SCORE

        if winner == self._opponent(player):
            return -WIN_SCORE

        if game.status == GameStatus.DRAW:
            return 0.0

        return self._evaluate_board(game.board, player)

    def _evaluate_board(self, board: Board, player: int) -> float:
        """
        Evaluate board using simple five-cell window patterns.
        """
        opponent = self._opponent(player)

        player_score = self._score_player_windows(board, player, opponent)
        opponent_score = self._score_player_windows(board, opponent, player)

        return player_score - opponent_score

    def _score_player_windows(self, board: Board, player: int, opponent: int) -> float:
        """
        Score all length-5 windows for one player.
        """
        total_score = 0.0

        lines = self._collect_lines(board)

        for line in lines:
            if len(line) < 5:
                continue

            for start in range(len(line) - 4):
                window = line[start : start + 5]

                player_count = window.count(player)
                opponent_count = window.count(opponent)

                # Blocked window: both players appear, so it is not useful.
                if player_count > 0 and opponent_count > 0:
                    continue

                total_score += self._score_window(player_count)

        return total_score

    def _score_window(self, player_count: int) -> float:
        """
        Score a five-cell window based on how many stones it contains.
        """
        if player_count == 5:
            return WIN_SCORE
        if player_count == 4:
            return 10_000
        if player_count == 3:
            return 1_000
        if player_count == 2:
            return 100
        if player_count == 1:
            return 10
        return 0

    def _collect_lines(self, board: Board) -> list[list[int]]:
        """
        Collect all rows, columns, and diagonals.
        """
        lines: list[list[int]] = []

        size = board.size

        # Rows
        for row in range(size):
            lines.append([int(board.grid[row, col]) for col in range(size)])

        # Columns
        for col in range(size):
            lines.append([int(board.grid[row, col]) for row in range(size)])

        # Diagonals: top-left to bottom-right
        for start_col in range(size):
            line = []
            row, col = 0, start_col
            while row < size and col < size:
                line.append(int(board.grid[row, col]))
                row += 1
                col += 1
            lines.append(line)

        for start_row in range(1, size):
            line = []
            row, col = start_row, 0
            while row < size and col < size:
                line.append(int(board.grid[row, col]))
                row += 1
                col += 1
            lines.append(line)

        # Diagonals: top-right to bottom-left
        for start_col in range(size):
            line = []
            row, col = 0, start_col
            while row < size and col >= 0:
                line.append(int(board.grid[row, col]))
                row += 1
                col -= 1
            lines.append(line)

        for start_row in range(1, size):
            line = []
            row, col = start_row, size - 1
            while row < size and col >= 0:
                line.append(int(board.grid[row, col]))
                row += 1
                col -= 1
            lines.append(line)

        return lines

    def _get_candidate_moves(self, game: Game) -> list[Move]:
        """
        Generate candidate moves near existing stones.

        Searching every empty cell on a 15x15 board is expensive.
        This method only considers empty cells within candidate_radius
        of an existing stone.
        """
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return []

        if len(game.board.move_history) == 0:
            centre = game.board.size // 2
            return [(centre, centre)]

        candidates: set[Move] = set()
        size = game.board.size

        for (stone_position, _player) in game.board.move_history:
            row, col = stone_position

            for delta_row in range(-self.candidate_radius, self.candidate_radius + 1):
                for delta_col in range(-self.candidate_radius, self.candidate_radius + 1):
                    candidate_row = row + delta_row
                    candidate_col = col + delta_col

                    if (
                        0 <= candidate_row < size
                        and 0 <= candidate_col < size
                        and game.board.is_empty(candidate_row, candidate_col)
                    ):
                        candidates.add((candidate_row, candidate_col))

        if not candidates:
            return legal_moves

        centre = size // 2

        return sorted(
            candidates,
            key=lambda move: abs(move[0] - centre) + abs(move[1] - centre),
        )

    def _opponent(self, player: int) -> int:
        return Board.WHITE if player == Board.BLACK else Board.BLACK