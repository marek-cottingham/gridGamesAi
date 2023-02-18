from datetime import datetime
from time import sleep

import matplotlib.pyplot as plt

from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game
from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import PENTAGO_MODELS_DIR, NGO_MODELS_DIR
from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.ngo.gameState import NgoGameState, NgoGameRunner
from gridGamesAi.ngo.temporalDifferenceModel import Ngo_TD_Agent
from gridGamesAi.ngo.render import NgoRender, UserNgoAgent
from gridGamesAi.pentago.scoringAgent import PentagoNaiveScoringAgent
from gridGamesAi.pentago.temporal_difference_model import Pentago_TD_Agent

path_0 = NGO_MODELS_DIR / "alpha_model_8000"
path_1 = NGO_MODELS_DIR / "alpha_model_64000"

def _run_game(wins_0, wins_1, scoreAgent_0, scoreAgent_1):
    g = Game(
        [MinimaxAgent(scoreAgent_0, 0), MinimaxAgent(scoreAgent_1, 0)],
        NgoGameState.init_with_n_random_placements(5, NgoGameRunner(2, 4, True)),
    )
    while not g.current_game_state.isEnd:
        g.moveWithCurrentPlayer()
    if g.current_game_state.winPlayer == 0:
        wins_0 += 1
    elif g.current_game_state.winPlayer == 1:
        wins_1 += 1
    return [wins_0, wins_1]

wins_0 = 0
wins_1 = 0
scoreAgent_0 = Ngo_TD_Agent(path_0)
scoreAgent_1 = Ngo_TD_Agent(path_1)
startTime = datetime.now()
for i in range(50):
    wins_0, wins_1 = _run_game(wins_0, wins_1, scoreAgent_0, scoreAgent_1)
    wins_1, wins_0 = _run_game(wins_1, wins_0, scoreAgent_1, scoreAgent_0)
    
    print(  f"  {wins_0}:{wins_1}|{path_0.stem}:{path_1.stem}         ", end='\r')
print(  f"  {wins_0}:{wins_1}|{path_0.stem}:{path_1.stem}           ")
print(f"Time taken: {datetime.now() - startTime}")