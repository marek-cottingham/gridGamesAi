from time import sleep
import matplotlib.pyplot as plt

from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.pentago.render import PentagoRender, UserPentagoAgent
from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game

plt.ion()
renderer = PentagoRender()
g = Game([RandomAgent(), UserPentagoAgent(renderer)], PentagoGameState(), renderer.render)

while not g.current_game_state.isEnd:
    g.moveWithCurrentPlayer()
    sleep(0.2)

plt.show(block=True)