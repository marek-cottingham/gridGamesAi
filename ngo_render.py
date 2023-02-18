from time import sleep

import matplotlib.pyplot as plt

from gridGamesAi.agents import RandomAgent
from gridGamesAi.game import Game
from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import NGO_MODELS_DIR
from gridGamesAi.ngo.gameState import NgoGameRunner, NgoGameState
from gridGamesAi.ngo.render import NgoRender, UserNgoAgent
from gridGamesAi.ngo.scoringAgent import NgoNaiveScoringAgent
from gridGamesAi.ngo.modelManager import ModelManager

alpha_model_dir = NGO_MODELS_DIR / "alpha_4x4_with_rotation"
alpha_runner = NgoGameRunner(2, 4, True)

model_dir, ngoRunner = alpha_model_dir, alpha_runner
model = ModelManager(
    model_dir, ngoRunner
)

model.load_latest_model()
# try:
#     model.ml_agent.plot_training_total_moves()
# except (TypeError) as e:
#     print(e)

# model.ml_agent.td_model.summary()
# layer_1 = model.ml_agent.td_model.layers[1].get_weights()
# [print(i.shape) for i in layer_1]

# fig, ax = plt.subplots(1, 1)
# pos = ax.imshow(layer_1[0], cmap='PiYG')
# fig.colorbar(pos, ax=ax)
# plt.show()

plt.ion()

renderer = NgoRender(ngoRunner.size_quadrant * 2)
ngoAgent = model.ml_agent
g = Game(
    # [UserNgoAgent(renderer), MinimaxAgent(ngoAgent, 1)],
    [MinimaxAgent(ngoAgent, 1), UserNgoAgent(renderer)],
    # [MinimaxAgent(ngoAgent, 1), MinimaxAgent(ngoAgent, 1)],
    NgoGameState.fairVariant(ngoRunner),
    renderer.render
)

while not g.current_game_state.isEnd:
    g.moveWithCurrentPlayer()
    sleep(0.2)

plt.show(block=True)
