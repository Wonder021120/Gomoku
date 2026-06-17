from gomoku.agents import RandomAgent
from gomoku.board import Board
from gomoku.game import Game
from gomoku.swap2 import run_swap2_opening


def test_swap2_opening_places_three_or_five_stones():
    game = Game(board_size=15, rule_name="standard")
    slicer = RandomAgent(seed=1)
    chooser = RandomAgent(seed=2)

    result, black_agent, white_agent = run_swap2_opening(
        game=game,
        slicer_agent=slicer,
        chooser_agent=chooser,
    )

    assert len(game.board.move_history) in (3, 5)
    assert result.choice in (
        "chooser_choose_black",
        "chooser_choose_white",
        "add_two_then_slicer_choose_black",
        "add_two_then_slicer_choose_white",
    )
    assert black_agent is not None
    assert white_agent is not None
    assert game.current_player == Board.WHITE


def test_swap2_opening_uses_fixed_central_initial_template():
    game = Game(board_size=15, rule_name="standard")
    slicer = RandomAgent(seed=1)
    chooser = RandomAgent(seed=2)

    result, _, _ = run_swap2_opening(
        game=game,
        slicer_agent=slicer,
        chooser_agent=chooser,
        choice_threshold=0.0,
    )

    assert result.opening_moves[0] == (Board.BLACK, (7, 7))
    assert result.opening_moves[1] == (Board.WHITE, (7, 8))
    assert result.opening_moves[2] == (Board.BLACK, (8, 7))


def test_swap2_opening_can_force_add_two_path():
    game = Game(board_size=15, rule_name="standard")
    slicer = RandomAgent(seed=1)
    chooser = RandomAgent(seed=2)

    result, black_agent, white_agent = run_swap2_opening(
        game=game,
        slicer_agent=slicer,
        chooser_agent=chooser,
        choice_threshold=999999999.0,
    )

    black_stones = sum(1 for _, colour in game.board.move_history if colour == Board.BLACK)
    white_stones = sum(1 for _, colour in game.board.move_history if colour == Board.WHITE)

    assert len(game.board.move_history) == 5
    assert black_stones == 3
    assert white_stones == 2
    assert result.opening_template == "central_5_stone_template"
    assert result.choice in (
        "add_two_then_slicer_choose_black",
        "add_two_then_slicer_choose_white",
    )
    assert black_agent is not None
    assert white_agent is not None
    assert game.current_player == Board.WHITE


def test_swap2_opening_uses_fixed_central_five_stone_template_when_add_two():
    game = Game(board_size=15, rule_name="standard")
    slicer = RandomAgent(seed=1)
    chooser = RandomAgent(seed=2)

    result, _, _ = run_swap2_opening(
        game=game,
        slicer_agent=slicer,
        chooser_agent=chooser,
        choice_threshold=999999999.0,
    )

    assert result.opening_moves[0] == (Board.BLACK, (7, 7))
    assert result.opening_moves[1] == (Board.WHITE, (7, 8))
    assert result.opening_moves[2] == (Board.BLACK, (8, 7))
    assert result.opening_moves[3] == (Board.WHITE, (6, 7))
    assert result.opening_moves[4] == (Board.BLACK, (7, 6))


def test_swap2_opening_moves_are_recorded_as_text():
    game = Game(board_size=15, rule_name="standard")
    slicer = RandomAgent(seed=1)
    chooser = RandomAgent(seed=2)

    result, _, _ = run_swap2_opening(
        game=game,
        slicer_agent=slicer,
        chooser_agent=chooser,
    )

    assert "B(" in result.opening_moves_text
    assert "W(" in result.opening_moves_text