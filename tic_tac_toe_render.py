from time import sleep
import matplotlib.pyplot as plt

from gridGamesAi.minimax import MinimaxAgent, PruningAgent
from gridGamesAi.render import GridRender, UserGridAgent
from gridGamesAi.tictactoe.gameState import TicTacToeGameState
from gridGamesAi.tictactoe.scoringAgent import TicTacToeManualScoringAgent
from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game

plt.ion()
renderer = GridRender()
g = Game([
    PruningAgent(TicTacToeManualScoringAgent()),
    UserGridAgent(renderer),
], TicTacToeGameState(), renderer.render)

while not g.current_game_state.isEnd:
    g.moveWithCurrentPlayer()

plt.show(block=True)