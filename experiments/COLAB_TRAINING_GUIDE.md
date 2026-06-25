# Colab Training Guide

## Purpose

This guide explains how to run longer NN-MCTS training on Google Colab.

The local project has already validated the full NN-MCTS pipeline:

* Generate self-play data
* Train a policy-value neural network
* Save a checkpoint
* Load the checkpoint into NN-MCTS
* Run NN-MCTS in tournaments
* Analyse and plot results

Colab will mainly be used to train a stronger checkpoint with more self-play data and more training epochs.

---

## 1. Open Colab and Enable GPU

In Google Colab:

1. Open a new notebook.
2. Go to `Runtime`.
3. Select `Change runtime type`.
4. Set hardware accelerator to `GPU`.
5. Save.

Then check GPU availability:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

---

## 2. Clone the Project

Replace the URL below with the GitHub repository URL.

```python
!git clone YOUR_GITHUB_REPOSITORY_URL
%cd Gomoku
```

If the repository folder name is different, change `Gomoku` to the correct folder name.

---

## 3. Install Dependencies

```python
!pip install -r requirements.txt
```

Then check that the project imports correctly:

```python
from gomoku.game import Game
from gomoku.nn_mcts_agent import NNMCTSAgent

print("Project imported successfully.")
```

---

## 4. Quick Pipeline Test on Colab

Before long training, run a very small test first.

### Generate Greedy Self-play Data

```python
!python experiments/generate_self_play_data.py --games 2 --agent greedy --rule standard --max-moves 60
```

Expected output:

```text
Saved dataset to: data/self_play/...
states shape: ...
policy_targets shape: ...
value_targets shape: ...
```

### Train an Initial Checkpoint

```python
!python experiments/train_nn.py \
  --dataset data/self_play/self_play_greedy_standard_15x15_2games_seed2026.npz \
  --epochs 3 \
  --batch-size 16 \
  --output checkpoints/gomoku_policy_value_net_colab_initial.pt
```

Expected output:

```text
Epoch 1/3 ...
Epoch 2/3 ...
Epoch 3/3 ...
Saved checkpoint to: checkpoints/gomoku_policy_value_net_colab_initial.pt
```

### Test NN-MCTS With the Checkpoint

```python
!python experiments/run_tournament.py \
  --games 1 \
  --black nn_mcts \
  --white nn_mcts \
  --rule standard \
  --nn-mcts-simulations 10 \
  --nn-mcts-checkpoint checkpoints/gomoku_policy_value_net_colab_initial.pt
```

If this runs successfully, the Colab environment is working.

---

## 5. Generate NN-MCTS Self-play Data

After the quick test, generate NN-MCTS self-play data.

Start with a small or medium setting:

```python
!python experiments/generate_self_play_data.py \
  --games 50 \
  --agent nn_mcts \
  --rule standard \
  --max-moves 150 \
  --nn-mcts-simulations 25 \
  --nn-mcts-checkpoint checkpoints/gomoku_policy_value_net_colab_initial.pt \
  --output data/self_play/self_play_nn_mcts_standard_15x15_50games.npz
```

If this is too slow, reduce:

```text
games: 10 or 20
nn-mcts-simulations: 10
max-moves: 100
```

If it runs well, increase later:

```text
games: 100 / 300 / 500
nn-mcts-simulations: 25 / 50
max-moves: 150 / 225
```

---

## 6. Train a Stronger Checkpoint

Train the policy-value network using the NN-MCTS self-play data:

```python
!python experiments/train_nn.py \
  --dataset data/self_play/self_play_nn_mcts_standard_15x15_50games.npz \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --output checkpoints/gomoku_policy_value_net_colab_final.pt
```

If training is stable and time allows, try:

```text
epochs: 30 / 50
batch-size: 64 / 128
```

---

## 7. Validate the Final Checkpoint

Run one small tournament:

```python
!python experiments/run_tournament.py \
  --games 1 \
  --black nn_mcts \
  --white nn_mcts \
  --rule standard \
  --nn-mcts-simulations 25 \
  --nn-mcts-checkpoint checkpoints/gomoku_policy_value_net_colab_final.pt
```

If this works, the final checkpoint can be downloaded and used locally.

---

## 8. Download the Checkpoint

In Colab, download the checkpoint:

```python
from google.colab import files

files.download("checkpoints/gomoku_policy_value_net_colab_final.pt")
```

Then place the downloaded file in the local project folder:

```text
checkpoints/gomoku_policy_value_net_colab_final.pt
```

The checkpoint file should not be committed to Git.

---

## 9. Local Verification After Download

Back on the local machine, test the downloaded checkpoint:

```cmd
python -c "from gomoku.game import Game; from gomoku.nn_mcts_agent import NNMCTSAgent; g=Game(board_size=15, rule_name='standard'); a=NNMCTSAgent(board_size=15, simulations=10, checkpoint_path='checkpoints/gomoku_policy_value_net_colab_final.pt'); print(a.select_move(g))"
```

Then run one tournament:

```cmd
python experiments\run_tournament.py --games 1 --black nn_mcts --white nn_mcts --rule standard --nn-mcts-simulations 25 --nn-mcts-checkpoint checkpoints\gomoku_policy_value_net_colab_final.pt
```

If both commands work, the checkpoint is ready for formal experiments.

---

## 10. Formal Experiment After Training

After the final checkpoint is ready, clear old validation results before formal experiments:

```cmd
del results\raw\*.csv
del results\processed\*.csv
del results\figures\*.png
```

Then run the formal rule-comparison experiments:

```cmd
python experiments\run_tournament.py --games 50 --black nn_mcts --white nn_mcts --rule standard --nn-mcts-simulations 50 --nn-mcts-checkpoint checkpoints\gomoku_policy_value_net_colab_final.pt
python experiments\run_tournament.py --games 50 --black nn_mcts --white nn_mcts --rule pro --nn-mcts-simulations 50 --nn-mcts-checkpoint checkpoints\gomoku_policy_value_net_colab_final.pt
python experiments\run_tournament.py --games 50 --black nn_mcts --white nn_mcts --rule swap2 --nn-mcts-simulations 50 --nn-mcts-checkpoint checkpoints\gomoku_policy_value_net_colab_final.pt
```

Then analyse and plot:

```cmd
python experiments\analyse_results.py
python experiments\plot_results.py
```

---

## Notes

The current NN-MCTS checkpoint is only for validation.

The final dissertation should only use results generated after:

* A stronger checkpoint has been trained.
* Validation CSV files have been cleared or archived.
* Formal experiments have been rerun with fixed parameters.
* The full Standard / Pro / Swap2 matrix has been completed.
