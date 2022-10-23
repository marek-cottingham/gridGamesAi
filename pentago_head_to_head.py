from datetime import datetime
from time import sleep

import matplotlib.pyplot as plt

from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game
from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import PENTAGO_MODELS_DIR
from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.ngo.render import PentagoRender, UserPentagoAgent
from gridGamesAi.pentago.scoringAgent import PentagoNaiveScoringAgent
from gridGamesAi.pentago.temporal_difference_model import Pentago_TD_Agent

path_0 = PENTAGO_MODELS_DIR / "alpha_model_12000"
path_1 = PENTAGO_MODELS_DIR / "alpha_model_20000"

def _run_game(wins_0, wins_1, pentAgent_0, pentAgent_1):
    g = Game(
        [MinimaxAgent(pentAgent_0, 0), MinimaxAgent(pentAgent_1, 0)],
        PentagoGameState.init_with_n_random_moves(5),
    )
    while not g.current_game_state.isEnd:
        g.moveWithCurrentPlayer()
    if g.current_game_state.winPlayer == 0:
        wins_0 += 1
    elif g.current_game_state.winPlayer == 1:
        wins_1 += 1
    return [wins_0, wins_1]

wins = [0,0]
pentAgent_0 = Pentago_TD_Agent(path_0)
pentAgent_1 = Pentago_TD_Agent(path_1)
startTime = datetime.now()
for i in range(50):
    wins = _run_game(wins[0], wins[1], pentAgent_0, pentAgent_1)
    wins = _run_game(wins[0], wins[1], pentAgent_1, pentAgent_0)
    
    print(  f"  {wins[0]}:{wins[1]}|{path_0.stem}:{path_1.stem}         ", end='\r')
print(  f"  {wins[0]}:{wins[1]}|{path_0.stem}:{path_1.stem}           ")
print(f"Time taken: {datetime.now() - startTime}")