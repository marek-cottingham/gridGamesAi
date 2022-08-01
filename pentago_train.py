from pathlib import Path
import os
from gridGamesAi.paths import PENTAGO_MODELS_DIR

from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.pentago.render import PentagoRender, UserPentagoAgent
from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game
from gridGamesAi.temporal_difference_model import Pentago_TD_Agent

MOD_PATH = None
LOAD_MOD_PATH = PENTAGO_MODELS_DIR / "test_model_2000"
SAVE_MOD_PATH = PENTAGO_MODELS_DIR / "test_model_2000"
os.makedirs(SAVE_MOD_PATH.parent, exist_ok=True)

pentAgent = Pentago_TD_Agent(None, None)
if LOAD_MOD_PATH is not None:
    pentAgent.load(LOAD_MOD_PATH, verbose=True)
else:
    pentAgent.create_td_model()
pentAgent.compile_td_model()
randAgent = RandomAgent()

def init_with_n_random_moves(n):
    gs = PentagoGameState()
    for _ in range(n):
        gs: PentagoGameState = randAgent.move(gs)
        gs = gs.skipMove()
    return gs

while pentAgent.training_calls < 2000:
    gs = init_with_n_random_moves(5)
    pentAgent.train_td_from_game(gs)
    pentAgent.save(SAVE_MOD_PATH, True)