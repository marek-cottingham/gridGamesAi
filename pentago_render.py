from time import sleep

import matplotlib.pyplot as plt

from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game
from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import PENTAGO_MODELS_DIR
from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.pentago.render import PentagoRender, UserPentagoAgent
from gridGamesAi.temporal_difference_model import Pentago_TD_Agent

MOD_PATH = PENTAGO_MODELS_DIR / "test_model_50000"

plt.ion()
renderer = PentagoRender()
pentAgent = Pentago_TD_Agent(MOD_PATH)
g = Game(
    [MinimaxAgent(pentAgent, 1), UserPentagoAgent(renderer)],
    PentagoGameState.fairVariant(),
    renderer.render
)

while not g.current_game_state.isEnd:
    g.moveWithCurrentPlayer()
    sleep(0.2)

plt.show(block=True)
