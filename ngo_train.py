from pathlib import Path
import os
import math
from time import time

from sympy import true
from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import NGO_4_MODELS_DIR
import matplotlib.pyplot as plt
import numpy as np

from gridGamesAi.ngo.gameState import NgoGameRunner, NgoGameState
from gridGamesAi.ngo.render import NgoRender, UserNgoAgent
from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game
from gridGamesAi.ngo.temporalDifferenceModel import Ngo_TD_Agent, Ngo_TD_Agent_v1b

model_dir = NGO_4_MODELS_DIR
LOAD_MOD_PATH = model_dir / "alpha_model_64000"
# LOAD_MOD_PATH = model_dir / "alpha_model_8000"
# LOAD_MOD_PATH = None

runner = NgoGameRunner(2, 4, true)

def update_save_path(training_calls):
    global SAVE_MOD_PATH
    SAVE_MOD_PATH = model_dir / f"beta_model_{training_calls}"

update_save_path(0)
os.makedirs(SAVE_MOD_PATH.parent, exist_ok=True)

mlAgent = Ngo_TD_Agent(None, runner, MinimaxAgent(None, max_depth=0))
if LOAD_MOD_PATH is not None:
    mlAgent.load(LOAD_MOD_PATH, verbose=True)
else:
    mlAgent.create_td_model(runner)
mlAgent.compile_td_model()
randAgent = RandomAgent()

# mlAgent.plot_training_total_moves()
# first_set_layer_weights = mlAgent.td_model.layers[1].get_weights()[0]
# print(first_set_layer_weights.shape)
# plt.imshow(first_set_layer_weights)
# plt.show()
# exit()
mlAgent.td_model.batch_size = 1

saveCallsIncrement = 8000

update_save_path(math.ceil((mlAgent.training_calls)/saveCallsIncrement)*saveCallsIncrement)
print(SAVE_MOD_PATH)

start = time()
while mlAgent.training_calls < 120000:
    gs = NgoGameState.init_with_n_random_moves(6, runner)
    mlAgent.train_td_from_game(gs)
    
    if mlAgent.training_calls % 40 == 0:
        mlAgent.save(SAVE_MOD_PATH, True)
        print("Time for 40 calls:", time() - start)
        start = time()

        if mlAgent.training_calls % saveCallsIncrement == 0:
            update_save_path(mlAgent.training_calls+saveCallsIncrement)
