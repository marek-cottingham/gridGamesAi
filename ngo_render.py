from time import sleep

import matplotlib.pyplot as plt

from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game
from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import NGO_MODELS_DIR
from gridGamesAi.ngo.gameState import NgoGameRunner, NgoGameState
from gridGamesAi.ngo.render import NgoRender, UserNgoAgent
from gridGamesAi.ngo.scoringAgent import NgoNaiveScoringAgent

plt.ion()
ngoRunner = NgoGameRunner(2, 4, True)
renderer = NgoRender(ngoRunner.size_quadrant * 2)
ngoAgent = NgoNaiveScoringAgent()
g = Game(
    [MinimaxAgent(ngoAgent, 1), UserNgoAgent(renderer)],
    NgoGameState.fairVariant(ngoRunner),
    renderer.render
)

while not g.current_game_state.isEnd:
    g.moveWithCurrentPlayer()
    sleep(0.2)

plt.show(block=True)
