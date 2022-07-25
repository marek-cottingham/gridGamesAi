from typing import List
import tensorflow as tf
import numpy as np
from gridGamesAi.common import AbstractGameState, AbstractGridGameState, baseScoreStrategy
from gridGamesAi.game import Game
from gridGamesAi.minimax import PruningAgent
from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.pentago.scoringAgent import PentagoNaiveScoringAgent
from gridGamesAi.scoringAgents import AbstractScoringAgent

class Pentago_TD_model(tf.keras.Model):
    def __init__(self, 
        *args,
        bootstrapScoringAgent = PentagoNaiveScoringAgent(),
        **kwargs
    ) -> None:
        self.minimaxAgent = PruningAgent
        self.gameStateFromTensor = PentagoGameState.fromTensor
        self.bootstrapScoringAgent: AbstractScoringAgent = bootstrapScoringAgent
        self.maxTDsteps = 7

        super().__init__(*args, **kwargs)

    def score(self, gameState: AbstractGridGameState):
        return baseScoreStrategy(gameState, self._score)

    def _score(self, gameState: AbstractGridGameState):
        return self.__call__(gameState.asTensor()[None, :]).numpy()[0,0]

    def train_step(self, gameStateTensor: tf.Tensor):
        gameState = self.gameStateFromTensor(gameStateTensor)
        return self.train_td_from_game(gameState)


    def train_td_from_game(self, rootGameState: AbstractGameState):

        movesSequence: List[AbstractGameState] = [rootGameState]

        for i in range(self.maxTDsteps):
            movesSequence.append(self.minimaxAgent.move(movesSequence[-1]))
            if movesSequence[-1].isEnd:
                break
        
        gameStateTensors = [
            gameState.asTensor() for gameState in movesSequence[:-1]
        ]

        scores = np.array([self.score(gameState) for gameState in movesSequence])
        deltas = scores[1:] - scores[:-1]

    

class TD_model(tf.keras.Model):
    def __init__(self, *args, td_factor = 0.7, **kwargs) -> None:
        self.td_factor = td_factor
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
