from time import sleep

import matplotlib.pyplot as plt
from pathlib import Path

from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game
from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import NGO_MODELS_DIR
from gridGamesAi.ngo.gameState import NgoGameRunner, NgoGameState
from gridGamesAi.ngo.render import NgoRender, UserNgoAgent
from gridGamesAi.ngo.scoringAgent import NgoNaiveScoringAgent
from gridGamesAi.ngo.modelManager import ModelManager

alpha_model_dir = NGO_MODELS_DIR / "alpha_4x4_with_rotation"
alpha_runner = NgoGameRunner(2, 4, True)

model_dir, ngoRunner = alpha_model_dir, alpha_runner
model_1 = ModelManager(model_dir, ngoRunner)
model_2 = ModelManager(model_dir, ngoRunner)

starting_positions = [NgoGameState.init_with_n_random_placements(2, ngoRunner) for _ in range(30)]

version_paths = model_1.get_sorted_model_paths()
for i in range(len(version_paths)-1):
    model_1.current_model_path = version_paths[i]
    model_2.current_model_path = version_paths[i+1]
    model_1.load_model()
    model_2.load_model()
    model_1_score = 0
    model_2_score = 0
    for starting_position in starting_positions:
        g = Game(
            [MinimaxAgent(model_1.ml_agent, 0), MinimaxAgent(model_2.ml_agent, 0)],
            starting_position
        )
        while not g.current_game_state.isEnd:
            g.moveWithCurrentPlayer()
        if g.current_game_state.winPlayer == 0:
            model_1_score += 1
        elif g.current_game_state.winPlayer == 1:
            model_2_score += 1
    for starting_position in starting_positions:
        g = Game(
            [MinimaxAgent(model_2.ml_agent, 0), MinimaxAgent(model_1.ml_agent, 0)],
            starting_position
        )
        while not g.current_game_state.isEnd:
            g.moveWithCurrentPlayer()
        if g.current_game_state.winPlayer == 0:
            model_2_score += 1
        elif g.current_game_state.winPlayer == 1:
            model_1_score += 1
    print(f"{Path(version_paths[i]).stem}: {model_1_score} | {Path(version_paths[i+1]).stem}: {model_2_score}")

