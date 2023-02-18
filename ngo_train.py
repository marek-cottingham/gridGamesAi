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
from gridGamesAi.ngo.resolvedPositions import ResolvedPositions


alpha_model_dir = NGO_MODELS_DIR / "alpha_4x4_with_rotation"
alpha_runner = NgoGameRunner(2, 4, True)

beta_model_dir = NGO_MODELS_DIR / "beta_4x4_with_rotation"
beta_runner = NgoGameRunner(2, 4, True)

# model_dir, runner = alpha_model_dir, alpha_runner
model_dir, runner = beta_model_dir, beta_runner

model = ModelManager(
    model_dir, runner
)

try:
    model.load_latest_model()
    try:
        model.ml_agent.plot_training_total_moves()
    except (TypeError) as e:
        print(e)
except (IndexError):
    model.new_model()

if model.base_path == beta_model_dir:
    model.ml_agent.td_model.batch_size = 20
    model.new_save_file_after_training_calls = 2000

model.training_model_moves_at_start = 10
model.training_random_moves = 3
model.train()