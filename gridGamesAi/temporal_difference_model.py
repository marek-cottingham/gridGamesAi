from functools import cache
from pathlib import Path
from typing import List
import tensorflow as tf
import numpy as np
from gridGamesAi.common import AbstractGameState, AbstractGridGameState, baseScoreStrategy
from gridGamesAi.game import Game
from gridGamesAi.minimax import PruningAgent
from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.pentago.scoringAgent import PentagoNaiveScoringAgent
from gridGamesAi.scoringAgents import AbstractScoringAgent, CachingScoringAgent

class Pentago_TD_Agent(CachingScoringAgent):
    def __init__(
        self,
        loadPath = None,
        createNewModel = False
    ) -> None:
        self.minimaxAgent = PruningAgent(scoringAgent=self, max_depth=6)
        self.td_model = None
        self.isCompiled = False
        self.training_games = 0

        if loadPath is not None:
            self.load_td_model(loadPath)
        if createNewModel:
            self.create_td_model()
        if self.td_model is not None:
            self.compile_td_model()

    def create_td_model(self):
        inputs = tf.keras.Input(shape=(74,))
        internal_1 = tf.keras.layers.Dense(100, activation='relu')
        internal_2 = tf.keras.layers.Dense(100, activation='relu')
        output = tf.keras.layers.Dense(1, activation='sigmoid')
        self.td_model = TD_model(
            inputs, 
            output(internal_2(internal_1(inputs))), 
        )

    def load_td_model(self, path: Path):
         self.td_model: TD_model = tf.keras.models.load_model(
            path,
            custom_objects={
                "pentagoTD_model": TD_model,
                "TD_model": TD_model,
            }
        )

    def save_td_model(self, path: Path, overwrite_ok=False):
        if (overwrite_ok or not path.exists()):
            self.td_model.save(path)

    def compile_td_model(self):
        if self.td_model is None:
            raise Exception("Must create or load a TD model before compiling")
        self.td_model.compile(
            loss=tf.keras.losses.MeanSquaredError(), 
            metrics=[],
            run_eagerly=True
        )
        self.isCompiled = True

    def requireComplied(self):
        if self.td_model is None:
            raise Exception("Must create or load a TD model")
        if not self.isCompiled:
            raise Exception("Must compile the TD model")

    @cache
    def score(self, gameState: PentagoGameState) -> float:
        return baseScoreStrategy(gameState, self._score)

    def _score(self, gameState: PentagoGameState):
        return self.td_model.__call__(gameState.asTensor()[None, :]).numpy()[0,0]

    def resetCache(self):
        self.score.cache_clear()

    def train_td_from_game(self, rootGameState: PentagoGameState):

        movesSequence: List[PentagoGameState] = [rootGameState]
        while not movesSequence[-1].isEnd:
            movesSequence.append(self.minimaxAgent.move(movesSequence[-1]))
        gameStateTensors = [
            gameState.asTensor() for gameState in movesSequence
        ]
        scores = np.array([self.score(gameState) for gameState in movesSequence])
        self.td_model.train_td_from_sequential_states(gameStateTensors, scores)
        self.resetCache()
class TD_model(tf.keras.Model):
    def __init__(self, *args, td_factor = 0.7, **kwargs) -> None:
        self.td_factor = td_factor
        self.training_calls = 0
        super().__init__(*args, **kwargs)

    def train_td_from_sequential_states(self, tensor_states: List[tf.Tensor], scores: np.ndarray):
        """Trains the tensor model from a series of tensor_states and their know score (using
        the latest verion of the model).
        Expects: 
            len(tensor_states) = len(scores)
        """
        deltas = scores[1:] - scores[:-1]
        gradients = []
        for tensor in tensor_states[:-1]:
            with tf.GradientTape() as tape:
                tfModelScore = self.__call__(tensor[None, :])
            gradients.append(
                tape.gradient(tfModelScore, self.trainable_variables)
            )
        delta_trainable = self.get_update_as_weighted_sum_gradients(deltas, gradients)
        self.optimizer.apply_gradients(zip(delta_trainable, self.trainable_variables))

        self.training_calls += 1


    def get_update_as_weighted_sum_gradients(
        self, deltas: np.ndarray, gradientsList: List[List[tf.Tensor]]
    ) -> List[tf.Tensor]:
        """Given the change in the score for a series of moves and the gradients of the scores of the
        positions prior to the moves, returns a list of tensor of updates to the model weights and 
        biases.
        Expect arguments of form:
            deltas[x]
            gradientsList[y][x]
            0 < x <= X
            0 < y <= Y
        """
        gradientsZip = zip(*gradientsList)
        weights = self.generate_temporal_difference_weights(deltas)
        dw = [self.weigh_gradients_by_temporal_difference(weights, g) for g in gradientsZip]
        return dw

    def generate_temporal_difference_weights(self, deltas: np.ndarray) -> np.ndarray:
        """Generates weights for the gradient at each time step"""
        powers = np.arange(deltas.size)
        per_step_weights = np.power(self.td_factor, powers)
        per_delta_weights = np.cumsum(per_step_weights)[::-1]
        return deltas * per_delta_weights

    def weigh_gradients_by_temporal_difference(
        self, weights: np.ndarray, gradients: List[tf.Tensor]
    ) -> tf.Tensor:
        """Applies a weight to each gradient, where the gradients are for position evaluation at
        subsequent time steps.
        
        Expects: len(deltas) == len(gradients)
        """
        dw_init = tf.zeros(gradients[0].shape)
        dw = tf.Variable(dw_init, trainable=False)

        M = len(weights)
        for t in range(M):
            dw.assign_add(weights[t] * gradients[t])
        return dw
