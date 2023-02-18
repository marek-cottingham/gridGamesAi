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

resolved_positions = ResolvedPositions(runner)
print("Generating resolved positions...")
resolved_positions.add_winning_positions(10, 0, 3)
print("Halfway there...")
resolved_positions.add_winning_positions(10, 1, 3)
print("Finished generating resolved positions.")

for path in model.get_sorted_model_paths():
    model.current_model_path = path
    model.load_model(True)
    score = model.rate_against_resolved_positions(resolved_positions)
    print(score)