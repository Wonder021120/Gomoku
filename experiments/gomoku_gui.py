from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gomoku.board import Board
from gomoku.game import Game
from gomoku.agents import RandomAgent, GreedyAgent
from gomoku.minimax_agent import MinimaxAgent
from gomoku.mcts_agent import MCTSAgent
from gomoku.nn_mcts_agent import NNMCTSAgent
from gomoku.swap2 import run_swap2_opening


BOARD_SIZE = 9
DEFAULT_RULE_NAME = "standard"

DEFAULT_MCTS_SIMULATIONS = 200
DEFAULT_NN_MCTS_SIMULATIONS = 200
DEFAULT_MINIMAX_DEPTH = 2

NN_MCTS_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "nn_mcts_training_loop_tactical" / "best.pt"

BLACK = Board.BLACK
WHITE = Board.WHITE
EMPTY = Board.EMPTY


class GomokuGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Gomoku AI Demo")
        self.root.resizable(False, False)

        self.board_size = BOARD_SIZE
        self.margin = 35
        self.cell = 50
        self.canvas_size = self.margin * 2 + self.cell * (self.board_size - 1)

        self.game: Game | None = None
        self.mode = "human"
        self.rule_name = DEFAULT_RULE_NAME
        self.human_player = BLACK
        self.agents: dict[int, object | None] = {}

        self.busy = False
        self.auto_play = False
        self.auto_job: str | None = None
        self.seed_counter = 0

        # Human Swap2 is intentionally implemented as a simple first version:
        # AI places the initial B-W-B opening, then the human chooses Black or White.
        # The "add two stones" option can be added later.
        self.human_swap2_phase = "none"

        self.status_var = tk.StringVar(value="Ready")
        self.info_var = tk.StringVar(value="9x9 Standard Gomoku")
        self.move_var = tk.StringVar(value="Moves: 0")
        self.rule_label_var = tk.StringVar(value="Rule: Standard")

        self._build_layout()
        self._new_human_game()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------
    # Layout
    # -----------------------------

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        board_frame = ttk.Frame(main)
        board_frame.grid(row=0, column=0, padx=(0, 12), sticky="n")

        self.canvas = tk.Canvas(
            board_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#d9b36c",
            highlightthickness=1,
            highlightbackground="#6f4e1e",
        )
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self._on_board_click)

        status_frame = ttk.Frame(board_frame)
        status_frame.grid(row=1, column=0, pady=(8, 0), sticky="ew")

        ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(status_frame, textvariable=self.info_var, wraplength=self.canvas_size).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(status_frame, textvariable=self.move_var).grid(row=2, column=0, sticky="w")

        control_frame = ttk.Frame(main)
        control_frame.grid(row=0, column=1, sticky="n")

        ttk.Label(control_frame, text="Gomoku GUI", font=("Arial", 14, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        ttk.Label(control_frame, text="Board: 9x9").grid(row=1, column=0, sticky="w")
        ttk.Label(control_frame, textvariable=self.rule_label_var).grid(
            row=2, column=0, sticky="w", pady=(0, 10)
        )

        self.notebook = ttk.Notebook(control_frame)
        self.notebook.grid(row=3, column=0, sticky="nsew")

        self._build_human_tab()
        self._build_ai_tab()

    def _build_human_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Human vs AI")

        ttk.Label(tab, text="Rule").grid(row=0, column=0, sticky="w")

        self.human_rule_var = tk.StringVar(value="Standard")
        self.human_rule_combo = ttk.Combobox(
            tab,
            textvariable=self.human_rule_var,
            values=["Standard", "Pro", "Swap2"],
            state="readonly",
            width=22,
        )
        self.human_rule_combo.grid(row=1, column=0, sticky="w", pady=(2, 10))

        ttk.Label(tab, text="Difficulty").grid(row=2, column=0, sticky="w")

        self.difficulty_var = tk.StringVar(value="Easy - NN-MCTS")
        self.difficulty_combo = ttk.Combobox(
            tab,
            textvariable=self.difficulty_var,
            values=[
                "Easy - NN-MCTS",
                "Medium - MCTS",
                "Hard - Minimax",
            ],
            state="readonly",
            width=22,
        )
        self.difficulty_combo.grid(row=3, column=0, sticky="w", pady=(2, 10))

        ttk.Label(tab, text="Your colour").grid(row=4, column=0, sticky="w")

        self.human_colour_var = tk.StringVar(value="Black - first")
        self.human_colour_combo = ttk.Combobox(
            tab,
            textvariable=self.human_colour_var,
            values=[
                "Black - first",
                "White - second",
            ],
            state="readonly",
            width=22,
        )
        self.human_colour_combo.grid(row=5, column=0, sticky="w", pady=(2, 10))

        ttk.Button(tab, text="New Human Game", command=self._new_human_game).grid(
            row=6, column=0, sticky="ew", pady=(0, 8)
        )

        self.swap2_human_frame = ttk.LabelFrame(tab, text="Human Swap2 choice", padding=8)
        self.swap2_human_frame.grid(row=7, column=0, sticky="ew", pady=(4, 8))

        self.take_black_button = ttk.Button(
            self.swap2_human_frame,
            text="Take Black",
            command=lambda: self._human_swap2_take_colour(BLACK),
        )
        self.take_black_button.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self.take_white_button = ttk.Button(
            self.swap2_human_frame,
            text="Take White",
            command=lambda: self._human_swap2_take_colour(WHITE),
        )
        self.take_white_button.grid(row=1, column=0, sticky="ew")

        ttk.Label(
            self.swap2_human_frame,
            text="Swap2 first version:\nAI places B-W-B.\nHuman chooses Black or White.",
            justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        ttk.Label(
            tab,
            text=(
                "Human mode supports:\n"
                "Standard\n"
                "Pro\n"
                "Swap2\n\n"
                "Difficulty mapping:\n"
                "Easy    = NN-MCTS\n"
                "Medium  = MCTS\n"
                "Hard    = Minimax"
            ),
            justify="left",
        ).grid(row=8, column=0, sticky="w", pady=(8, 0))

        self._set_swap2_choice_buttons_enabled(False)

    def _build_ai_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="AI vs AI")

        agent_values = ["Random", "Greedy", "NN-MCTS", "MCTS", "Minimax"]

        ttk.Label(tab, text="Rule").grid(row=0, column=0, sticky="w")
        self.ai_rule_var = tk.StringVar(value="Standard")
        self.ai_rule_combo = ttk.Combobox(
            tab,
            textvariable=self.ai_rule_var,
            values=["Standard", "Pro", "Swap2"],
            state="readonly",
            width=22,
        )
        self.ai_rule_combo.grid(row=1, column=0, sticky="w", pady=(2, 10))

        ttk.Label(tab, text="Black / opening agent").grid(row=2, column=0, sticky="w")
        self.black_ai_var = tk.StringVar(value="MCTS")
        self.black_ai_combo = ttk.Combobox(
            tab,
            textvariable=self.black_ai_var,
            values=agent_values,
            state="readonly",
            width=22,
        )
        self.black_ai_combo.grid(row=3, column=0, sticky="w", pady=(2, 10))

        ttk.Label(tab, text="White / choosing agent").grid(row=4, column=0, sticky="w")
        self.white_ai_var = tk.StringVar(value="NN-MCTS")
        self.white_ai_combo = ttk.Combobox(
            tab,
            textvariable=self.white_ai_var,
            values=agent_values,
            state="readonly",
            width=22,
        )
        self.white_ai_combo.grid(row=5, column=0, sticky="w", pady=(2, 10))

        ttk.Button(tab, text="New AI Match", command=self._new_ai_match).grid(
            row=6, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(tab, text="Step One Move", command=self._step_ai_once).grid(
            row=7, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(tab, text="Auto Play", command=self._start_auto_play).grid(
            row=8, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(tab, text="Stop Auto", command=self._stop_auto_play).grid(
            row=9, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(tab, text="Reset AI Match", command=self._new_ai_match).grid(
            row=10, column=0, sticky="ew", pady=(0, 6)
        )

        ttk.Label(
            tab,
            text=(
                "AI mode supports:\n"
                "Standard\n"
                "Pro\n"
                "Swap2\n\n"
                "In Swap2 mode:\n"
                "Black agent = opening agent\n"
                "White agent = choosing agent"
            ),
            justify="left",
        ).grid(row=11, column=0, sticky="w", pady=(8, 0))

    # -----------------------------
    # Game setup
    # -----------------------------

    def _new_game(self, rule_name: str | None = None) -> None:
        self._stop_auto_play()

        if rule_name is not None:
            self.rule_name = rule_name

        # Swap2 opening is handled manually in the GUI or by run_swap2_opening,
        # so the underlying Game object starts with standard move legality.
        game_rule = "standard" if self.rule_name == "swap2" else self.rule_name
        self.game = Game(board_size=self.board_size, rule_name=game_rule)

        self.busy = False
        self.human_swap2_phase = "none"
        self._set_swap2_choice_buttons_enabled(False)

        self.rule_label_var.set(f"Rule: {self._display_rule_name(self.rule_name)}")
        self._draw_board()
        self._update_status("New game started.")

    def _new_human_game(self) -> None:
        self.mode = "human"

        selected_rule = self.human_rule_var.get().lower()
        self._new_game(rule_name=selected_rule)

        ai_agent_name = self._difficulty_to_agent_name(self.difficulty_var.get())

        try:
            ai_agent = self._create_agent(ai_agent_name)
        except Exception as exc:
            messagebox.showerror("Agent error", str(exc))
            self._update_status("Failed to create AI agent.")
            return

        if selected_rule == "swap2":
            self._setup_human_swap2_game(ai_agent_name=ai_agent_name, ai_agent=ai_agent)
            return

        self.human_player = BLACK if self.human_colour_var.get().startswith("Black") else WHITE
        self.agents = {
            self.human_player: None,
            -self.human_player: ai_agent,
        }

        human_label = "Black" if self.human_player == BLACK else "White"
        ai_label = "White" if self.human_player == BLACK else "Black"
        rule_label = self._display_rule_name(selected_rule)

        self.info_var.set(
            f"Human vs AI | Rule: {rule_label} | Human: {human_label} | AI: {ai_label} {ai_agent_name}"
        )

        if self.game is not None and self.game.current_player != self.human_player:
            self.root.after(300, self._request_ai_move)

    def _setup_human_swap2_game(self, ai_agent_name: str, ai_agent) -> None:
        if self.game is None:
            return

        try:
            self._place_fixed_swap2_opening()
        except Exception as exc:
            messagebox.showerror("Swap2 error", str(exc))
            self._update_status("Failed to create Swap2 opening.")
            return

        # Human is the chooser in this first version.
        # Human chooses whether to take Black or White after the B-W-B opening.
        self.agents = {
            BLACK: None,
            WHITE: None,
        }
        self.pending_human_swap2_ai_agent = ai_agent
        self.pending_human_swap2_ai_name = ai_agent_name
        self.human_swap2_phase = "choose_colour"

        self.info_var.set(
            "Human vs AI | Rule: Swap2 | AI placed opening stones | "
            "Choose Take Black or Take White."
        )
        self._draw_board()
        self._set_swap2_choice_buttons_enabled(True)
        self._update_status("Swap2 opening ready. Please choose Black or White.")

    def _place_fixed_swap2_opening(self) -> None:
        if self.game is None:
            return

        centre = self.board_size // 2
        opening_moves = [
            (centre, centre),      # Black
            (centre, centre + 1),  # White
            (centre + 1, centre),  # Black
        ]

        for move in opening_moves:
            self.game.play_move(move)

    def _human_swap2_take_colour(self, colour: int) -> None:
        if self.game is None:
            return

        if self.mode != "human" or self.rule_name != "swap2":
            return

        if self.human_swap2_phase != "choose_colour":
            return

        ai_agent = getattr(self, "pending_human_swap2_ai_agent", None)
        ai_name = getattr(self, "pending_human_swap2_ai_name", "AI")

        if ai_agent is None:
            messagebox.showerror("Swap2 error", "No pending AI agent found.")
            return

        self.human_player = colour

        self.agents = {
            self.human_player: None,
            -self.human_player: ai_agent,
        }

        self.human_swap2_phase = "normal"
        self._set_swap2_choice_buttons_enabled(False)

        human_label = "Black" if self.human_player == BLACK else "White"
        ai_label = "White" if self.human_player == BLACK else "Black"

        self.info_var.set(
            f"Human vs AI | Rule: Swap2 | Human took {human_label} | AI: {ai_label} {ai_name}"
        )
        self._draw_board()
        self._update_status(f"Human chose {human_label}.")

        if self.game.current_player != self.human_player and not self.game.is_over():
            self.root.after(300, self._request_ai_move)

    def _new_ai_match(self) -> None:
        self.mode = "ai"

        selected_rule = self.ai_rule_var.get().lower()
        self._new_game(rule_name=selected_rule)

        black_name = self.black_ai_var.get()
        white_name = self.white_ai_var.get()

        try:
            black_agent = self._create_agent(black_name)
            white_agent = self._create_agent(white_name)
        except Exception as exc:
            messagebox.showerror("Agent error", str(exc))
            self._update_status("Failed to create AI agents.")
            return

        if selected_rule == "swap2":
            self._setup_swap2_ai_match(
                opening_agent_name=black_name,
                choosing_agent_name=white_name,
                opening_agent=black_agent,
                choosing_agent=white_agent,
            )
            return

        self.agents = {
            BLACK: black_agent,
            WHITE: white_agent,
        }

        rule_label = self._display_rule_name(selected_rule)
        self.info_var.set(f"AI vs AI | Rule: {rule_label} | Black: {black_name} | White: {white_name}")
        self._update_status("AI match ready.")

    def _setup_swap2_ai_match(
        self,
        opening_agent_name: str,
        choosing_agent_name: str,
        opening_agent,
        choosing_agent,
    ) -> None:
        if self.game is None:
            return

        try:
            result, actual_black_agent, actual_white_agent = run_swap2_opening(
                game=self.game,
                slicer_agent=opening_agent,
                chooser_agent=choosing_agent,
            )
        except Exception as exc:
            messagebox.showerror("Swap2 error", str(exc))
            self._update_status("Swap2 opening failed.")
            return

        self.agents = {
            BLACK: actual_black_agent,
            WHITE: actual_white_agent,
        }

        choice = getattr(result, "choice", "unknown")
        opening_text = getattr(result, "opening_moves_text", "")
        extra = f" | Opening moves: {opening_text}" if opening_text else ""

        self.info_var.set(
            "AI vs AI | Rule: Swap2 | "
            f"Opening: {opening_agent_name} | Chooser: {choosing_agent_name} | "
            f"{choice}{extra}"
        )

        self._draw_board()
        self._update_status(f"Swap2 opening completed. {choice}")

    # -----------------------------
    # Agent creation
    # -----------------------------

    def _difficulty_to_agent_name(self, value: str) -> str:
        if value.startswith("Easy"):
            return "NN-MCTS"
        if value.startswith("Medium"):
            return "MCTS"
        return "Minimax"

    def _create_agent(self, name: str):
        self.seed_counter += 1
        seed = 2026 + self.seed_counter

        if name == "Random":
            return self._try_create(
                name,
                [
                    lambda: RandomAgent(seed=seed),
                    lambda: RandomAgent(),
                ],
            )

        if name == "Greedy":
            return self._try_create(
                name,
                [
                    lambda: GreedyAgent(seed=seed),
                    lambda: GreedyAgent(),
                ],
            )

        if name == "MCTS":
            return self._try_create(
                name,
                [
                    lambda: MCTSAgent(simulations=DEFAULT_MCTS_SIMULATIONS, seed=seed),
                    lambda: MCTSAgent(simulations=DEFAULT_MCTS_SIMULATIONS),
                    lambda: MCTSAgent(num_simulations=DEFAULT_MCTS_SIMULATIONS),
                    lambda: MCTSAgent(),
                ],
            )

        if name == "Minimax":
            return self._try_create(
                name,
                [
                    lambda: MinimaxAgent(depth=DEFAULT_MINIMAX_DEPTH, candidate_radius=1, seed=seed),
                    lambda: MinimaxAgent(depth=DEFAULT_MINIMAX_DEPTH, candidate_radius=1),
                    lambda: MinimaxAgent(depth=DEFAULT_MINIMAX_DEPTH),
                    lambda: MinimaxAgent(),
                ],
            )

        if name == "NN-MCTS":
            if not NN_MCTS_CHECKPOINT.exists():
                raise FileNotFoundError(
                    f"NN-MCTS checkpoint not found:\n{NN_MCTS_CHECKPOINT}"
                )

            return self._try_create(
                name,
                [
                    lambda: NNMCTSAgent(
                        board_size=BOARD_SIZE,
                        simulations=DEFAULT_NN_MCTS_SIMULATIONS,
                        checkpoint_path=str(NN_MCTS_CHECKPOINT),
                        device="auto",
                        model_variant="paper_9x9",
                    ),
                    lambda: NNMCTSAgent(
                        board_size=BOARD_SIZE,
                        simulations=DEFAULT_NN_MCTS_SIMULATIONS,
                        checkpoint_path=str(NN_MCTS_CHECKPOINT),
                        device="auto",
                    ),
                ],
            )

        raise ValueError(f"Unknown agent: {name}")

    def _try_create(self, name: str, constructors):
        last_error: Exception | None = None

        for constructor in constructors:
            try:
                return constructor()
            except TypeError as exc:
                last_error = exc

        if last_error is not None:
            raise RuntimeError(f"Could not create agent {name}: {last_error}")

        raise RuntimeError(f"Could not create agent {name}")

    # -----------------------------
    # User input and AI moves
    # -----------------------------

    def _on_board_click(self, event) -> None:
        if self.game is None:
            return

        if self.mode != "human":
            return

        if self.busy or self.game.is_over():
            return

        if self.human_swap2_phase == "choose_colour":
            self._update_status("Please choose Take Black or Take White first.")
            return

        if self.game.current_player != self.human_player:
            return

        move = self._event_to_move(event)
        if move is None:
            return

        legal_moves = set(self.game.get_legal_moves())
        if move not in legal_moves:
            self._update_status("Illegal move.")
            return

        try:
            self.game.play_move(move)
        except Exception as exc:
            messagebox.showerror("Move error", str(exc))
            return

        self._draw_board()
        self._update_status("Human moved.")

        if self.game.is_over():
            self._update_status_for_game_end()
            return

        self.root.after(250, self._request_ai_move)

    def _event_to_move(self, event):
        col = round((event.x - self.margin) / self.cell)
        row = round((event.y - self.margin) / self.cell)

        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return None

        x, y = self._board_to_canvas(row, col)
        if abs(event.x - x) > self.cell * 0.45 or abs(event.y - y) > self.cell * 0.45:
            return None

        return row, col

    def _request_ai_move(self) -> None:
        if self.game is None:
            return

        if self.busy or self.game.is_over():
            return

        player = self.game.current_player
        agent = self.agents.get(player)

        if agent is None:
            return

        self.busy = True
        player_label = "Black" if player == BLACK else "White"
        self._update_status(f"{player_label} AI is thinking...")

        game_copy = self.game.copy()

        def worker() -> None:
            move = None
            error_text = None

            try:
                move = agent.select_move(game_copy)
            except Exception:
                error_text = traceback.format_exc()

            self.root.after(0, lambda: self._finish_ai_move(player, move, error_text))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_ai_move(self, player: int, move, error_text: str | None) -> None:
        if self.game is None:
            return

        self.busy = False

        if error_text is not None:
            messagebox.showerror("AI error", error_text)
            self._update_status("AI move failed.")
            return

        if self.game.is_over():
            self._update_status_for_game_end()
            return

        if player != self.game.current_player:
            self._update_status("AI move ignored because turn changed.")
            return

        legal_moves = set(self.game.get_legal_moves())
        if move not in legal_moves:
            messagebox.showerror("AI error", f"AI returned illegal move: {move}")
            self._update_status("AI returned illegal move.")
            return

        try:
            self.game.play_move(move)
        except Exception as exc:
            messagebox.showerror("Move error", str(exc))
            self._update_status("AI move failed.")
            return

        player_label = "Black" if player == BLACK else "White"
        self._draw_board()
        self._update_status(f"{player_label} AI moved: {move}")

        if self.game.is_over():
            self._update_status_for_game_end()
            return

        if self.mode == "ai" and self.auto_play:
            self.auto_job = self.root.after(250, self._auto_step)

    def _step_ai_once(self) -> None:
        if self.mode != "ai":
            self._new_ai_match()

        self._stop_auto_play()

        if self.game is None or self.game.is_over():
            return

        self._request_ai_move()

    def _start_auto_play(self) -> None:
        if self.mode != "ai":
            self._new_ai_match()

        if self.game is None or self.game.is_over():
            return

        self.auto_play = True
        self._auto_step()

    def _auto_step(self) -> None:
        if not self.auto_play:
            return

        if self.game is None:
            return

        if self.game.is_over():
            self._stop_auto_play()
            self._update_status_for_game_end()
            return

        if not self.busy:
            self._request_ai_move()

    def _stop_auto_play(self) -> None:
        self.auto_play = False

        if self.auto_job is not None:
            try:
                self.root.after_cancel(self.auto_job)
            except Exception:
                pass
            self.auto_job = None

    # -----------------------------
    # Drawing and board helpers
    # -----------------------------

    def _draw_board(self) -> None:
        self.canvas.delete("all")

        for i in range(self.board_size):
            x0, y0 = self._board_to_canvas(i, 0)
            x1, y1 = self._board_to_canvas(i, self.board_size - 1)
            self.canvas.create_line(x0, y0, x1, y1, width=1)

            x0, y0 = self._board_to_canvas(0, i)
            x1, y1 = self._board_to_canvas(self.board_size - 1, i)
            self.canvas.create_line(x0, y0, x1, y1, width=1)

        centre = self.board_size // 2
        x, y = self._board_to_canvas(centre, centre)
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black", outline="black")

        if self.game is None:
            return

        last_move = self._get_last_move()

        for row in range(self.board_size):
            for col in range(self.board_size):
                value = self._get_cell(row, col)

                if value == EMPTY:
                    continue

                x, y = self._board_to_canvas(row, col)
                radius = self.cell * 0.38

                if value == BLACK:
                    self.canvas.create_oval(
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                        fill="black",
                        outline="black",
                    )
                elif value == WHITE:
                    self.canvas.create_oval(
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                        fill="white",
                        outline="black",
                        width=2,
                    )

                if last_move == (row, col):
                    mark_radius = self.cell * 0.16
                    self.canvas.create_rectangle(
                        x - mark_radius,
                        y - mark_radius,
                        x + mark_radius,
                        y + mark_radius,
                        outline="red",
                        width=2,
                    )

    def _board_to_canvas(self, row: int, col: int):
        x = self.margin + col * self.cell
        y = self.margin + row * self.cell
        return x, y

    def _get_cell(self, row: int, col: int) -> int:
        if self.game is None:
            return EMPTY

        board = self.game.board

        grid = getattr(board, "grid", None)
        if grid is not None:
            return int(grid[row][col])

        cells = getattr(board, "board", None)
        if cells is not None:
            return int(cells[row][col])

        raise AttributeError("Could not find board grid.")

    def _get_last_move(self):
        if self.game is None:
            return None

        history = getattr(self.game.board, "move_history", None)
        if not history:
            return None

        item = history[-1]

        if isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[0], int) and isinstance(item[1], int):
                return item

            if isinstance(item[1], tuple) and len(item[1]) == 2:
                return item[1]

        return None

    def _get_move_count(self) -> int:
        if self.game is None:
            return 0

        history = getattr(self.game.board, "move_history", None)
        if history is not None:
            return len(history)

        count = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self._get_cell(row, col) != EMPTY:
                    count += 1
        return count

    # -----------------------------
    # Status helpers
    # -----------------------------

    def _set_swap2_choice_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"

        if hasattr(self, "take_black_button"):
            self.take_black_button.configure(state=state)

        if hasattr(self, "take_white_button"):
            self.take_white_button.configure(state=state)

    def _update_status(self, message: str) -> None:
        if self.game is None:
            self.status_var.set(message)
            self.move_var.set("Moves: 0")
            return

        if self.game.is_over():
            self.status_var.set(message)
        elif self.human_swap2_phase == "choose_colour":
            self.status_var.set(message)
        else:
            current = "Black" if self.game.current_player == BLACK else "White"
            self.status_var.set(f"{message} Current turn: {current}")

        self.move_var.set(f"Moves: {self._get_move_count()}")

    def _update_status_for_game_end(self) -> None:
        if self.game is None:
            return

        winner = self.game.get_winner()

        if winner == BLACK:
            self.status_var.set("Game over: Black wins.")
        elif winner == WHITE:
            self.status_var.set("Game over: White wins.")
        else:
            self.status_var.set("Game over: Draw.")

        self.move_var.set(f"Moves: {self._get_move_count()}")

    def _display_rule_name(self, rule_name: str) -> str:
        if rule_name == "swap2":
            return "Swap2"
        return rule_name.capitalize()

    def _on_close(self) -> None:
        self._stop_auto_play()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    GomokuGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
