# Formal Experiment Plan

## 1. Research Question

The main research question of this project is:

**Do different opening rules affect or reduce first-player advantage in Gomoku?**

The project focuses on comparing first-player advantage under three opening rules:

- Standard
- Pro
- Swap2

AI agents are used as experimental tools to evaluate these rules under controlled conditions.

---

## 2. Main Experiment: Rule Fairness Evaluation

The main experiment uses same-agent self-play.

This means that the same AI agent plays both black and white under each rule. This controls the playing strength of both sides, so that the effect of the opening rule can be observed more clearly.

### Main Experiment Matrix

| Agent | Standard | Pro | Swap2 |
|---|---|---|---|
| Minimax vs Minimax | Required | Required | Required |
| MCTS vs MCTS | Required | Required | Required |
| NN-MCTS vs NN-MCTS | Required after implementation | Required after implementation | Required after implementation |

### Primary Metrics

The primary metrics are:

- First-player win rate
- Black win rate
- White win rate
- Draw rate

These metrics directly address whether opening rules affect first-player advantage.

### Secondary Metrics

The secondary metrics are:

- Average number of moves
- Average decision time
- Swap2 choice frequency

These metrics are used to understand game length, computational cost, and the behaviour of the Swap2 opening protocol.

---

## 3. Auxiliary Experiment: AI Strength Comparison

The auxiliary experiment compares different AI agents against each other.

This is not the main evidence for opening-rule fairness. Instead, it is used to validate the relative strength levels of the implemented agents.

### Auxiliary Experiment Matrix

| Match-up | Purpose |
|---|---|
| Greedy vs Random / Random vs Greedy | Validate simple baseline improvement |
| Minimax vs Greedy / Greedy vs Minimax | Validate Minimax as a stronger search baseline |
| MCTS vs Minimax / Minimax vs MCTS | Compare search-based AI methods |
| NN-MCTS vs MCTS / MCTS vs NN-MCTS | Evaluate whether NN-MCTS improves over standard MCTS |

Both directions should be tested where possible, because swapping black and white helps separate AI strength from first-player advantage.

---

## 4. Baseline Agent Parameters

### Minimax

The main Minimax baseline will use:

- Depth: 2
- Candidate radius: 1

This setting is selected to balance playing strength and computational cost.

If runtime allows, a smaller supplementary experiment may also use:

- Depth: 3
- Candidate radius: 1

### MCTS

The main MCTS baseline will initially use:

- Simulations: 50
- Rollout depth limit: 30
- Candidate radius: 1
- Exploration weight: 1.4

If runtime allows, a stronger supplementary setting may be tested:

- Simulations: 100
- Rollout depth limit: 40
- Candidate radius: 1
- Exploration weight: 1.4

The same parameter settings must be used across Standard, Pro, and Swap2 for fair comparison.

---

## 5. Number of Games

The current validation experiments use a small number of games only to verify the pipeline.

For formal experiments, the recommended minimum is:

- 30 games per rule per agent setting

If runtime allows, the target should be:

- 50 to 100 games per rule per agent setting

For NN-MCTS, the number of games may be adjusted depending on training and inference cost.

---

## 6. Example Formal Experiment Commands

### Minimax vs Minimax

Command for Standard:

python experiments\run_tournament.py --games 50 --black minimax --white minimax --rule standard --minimax-depth 2 --minimax-candidate-radius 1

Command for Pro:

python experiments\run_tournament.py --games 50 --black minimax --white minimax --rule pro --minimax-depth 2 --minimax-candidate-radius 1

Command for Swap2:

python experiments\run_tournament.py --games 50 --black minimax --white minimax --rule swap2 --minimax-depth 2 --minimax-candidate-radius 1

### MCTS vs MCTS

Command for Standard:

python experiments\run_tournament.py --games 50 --black mcts --white mcts --rule standard --mcts-simulations 50 --mcts-rollout-depth 30 --mcts-candidate-radius 1

Command for Pro:

python experiments\run_tournament.py --games 50 --black mcts --white mcts --rule pro --mcts-simulations 50 --mcts-rollout-depth 30 --mcts-candidate-radius 1

Command for Swap2:

python experiments\run_tournament.py --games 50 --black mcts --white mcts --rule swap2 --mcts-simulations 50 --mcts-rollout-depth 30 --mcts-candidate-radius 1

---

## 7. Planned NN-MCTS Experiments

After NN-MCTS is implemented, it will be added to the same rule-comparison framework.

### NN-MCTS Main Matrix

| Agent | Standard | Pro | Swap2 |
|---|---|---|---|
| NN-MCTS vs NN-MCTS | Required | Required | Required |

The purpose is to investigate whether the trends observed using Minimax and MCTS remain under a stronger self-learning AI agent.

If training on the full 15x15 board is too expensive, reduced board sizes such as 9x9 or 11x11 may be used first for the neural self-play experiments.

---

## 8. Planned Figures

The final dissertation should prioritise the following figures:

### Main Figures

- First-player win rate by rule for same-agent matches
- Black / white / draw rates by rule for same-agent matches

### Supporting Figures

- Average number of moves by rule
- Average decision time by AI agent
- Swap2 choice frequency

The current generated figures are validation plots only. Formal figures should be regenerated after final experiment parameters are fixed and formal experiments are rerun.

---

## 9. Notes on Current Validation Data

The current CSV files and plots are used for pipeline validation only.

They should not be treated as final experimental results because:

- The number of games is small.
- Minimax and MCTS parameters are lightweight.
- NN-MCTS has not yet been implemented.
- Some experiment combinations may be missing.

Before final evaluation, the raw and processed result folders should be cleaned or archived, and formal experiments should be rerun with fixed parameters and seeds.