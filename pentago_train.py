from pathlib import Path
import os
from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import PENTAGO_MODELS_DIR

from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.pentago.render import PentagoRender, UserPentagoAgent
from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game
from gridGamesAi.temporal_difference_model import Pentago_TD_Agent

MOD_PATH = None
LOAD_MOD_PATH = PENTAGO_MODELS_DIR / "test_model_50000"
SAVE_MOD_PATH = PENTAGO_MODELS_DIR / "test_model_50000"
os.makedirs(SAVE_MOD_PATH.parent, exist_ok=True)

pentAgent = Pentago_TD_Agent(None, None, MinimaxAgent(None, max_depth=0))
if LOAD_MOD_PATH is not None:
    pentAgent.load(LOAD_MOD_PATH, verbose=True)
else:
    pentAgent.create_td_model()
pentAgent.compile_td_model()
randAgent = RandomAgent()

def init_with_n_random_moves(n):
    gs = PentagoGameState.fairVariant()
    for _ in range(n):
        gs: PentagoGameState = randAgent.move(gs)
        gs = gs.skipMove()
    return gs

pentAgent.plot_training_total_moves()

while pentAgent.training_calls < 50000:
    gs = init_with_n_random_moves(6)
    pentAgent.train_td_from_game(gs)
    if pentAgent.training_calls % 50 == 0:
        pentAgent.save(SAVE_MOD_PATH, True)