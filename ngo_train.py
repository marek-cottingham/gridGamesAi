from pathlib import Path
import os
import math
from time import time

from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import NGO_MODELS_DIR
import matplotlib.pyplot as plt
import numpy as np

from gridGamesAi.ngo.gameState import NgoGameRunner, NgoGameState
from gridGamesAi.ngo.modelManager import ModelManager


alpha_model_dir = NGO_MODELS_DIR / "alpha_4x4_with_rotation"
alpha_runner = NgoGameRunner(2, 4, True)

beta_model_dir = NGO_MODELS_DIR / "beta_4x4_no_rotation"
beta_runner = NgoGameRunner(2, 4, False)

model = ModelManager(
    # alpha_model_dir, alpha_runner
    beta_model_dir, beta_runner
)

# try:
#     model.load_latest_model()
# except FileNotFoundError:
#     model.new_model()


# model.ml_agent.plot_training_total_moves()
# model.train()

for _ in range(10):
    gs = NgoGameState.init_as_winning_position(beta_runner, 1, 3)
    print(gs.grid)
    print(gs.winPlayer)