import numpy as np
import torch

board = np.zeros((15, 15), dtype=int)
board[7, 7] = 1

print(board)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Gomoku project environment is ready.")