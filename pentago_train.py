from pathlib import Path
import os
import math
from time import time
from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import PENTAGO_MODELS_DIR

from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.ngo.render import NgoRender, UserNgoAgent
from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game
from gridGamesAi.pentago.temporal_difference_model import Pentago_TD_Agent

MOD_PATH = None
LOAD_MOD_PATH = PENTAGO_MODELS_DIR / "alpha_model_32000"

def update_save_path(training_calls):
    global SAVE_MOD_PATH
    SAVE_MOD_PATH = PENTAGO_MODELS_DIR / f"alpha_model_{training_calls}"

update_save_path(0)
os.makedirs(SAVE_MOD_PATH.parent, exist_ok=True)

pentAgent = Pentago_TD_Agent(None, None, MinimaxAgent(None, max_depth=0))
if LOAD_MOD_PATH is not None:
    pentAgent.load(LOAD_MOD_PATH, verbose=True)
else:
    pentAgent.create_td_model()
pentAgent.compile_td_model()
randAgent = RandomAgent()

pentAgent.plot_training_total_moves()
pentAgent.td_model.batch_size = 20

update_save_path(math.ceil((pentAgent.training_calls)/4000)*4000)
print(SAVE_MOD_PATH)

start = time()
while pentAgent.training_calls < 60000:
    gs = PentagoGameState.init_with_n_random_moves(6)
    pentAgent.train_td_from_game(gs)
    
    if pentAgent.training_calls % 40 == 0:
        pentAgent.save(SAVE_MOD_PATH, True)
        print("Time for 40 calls:", time() - start)
        start = time()

        if pentAgent.training_calls % 4000 == 0:
            update_save_path(pentAgent.training_calls+4000)
