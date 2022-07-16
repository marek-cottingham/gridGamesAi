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
        self.bootstrapMode = False
        self.maxTDsteps = 7

        super().__init__(*args, **kwargs)

    def score(self, gameState: AbstractGridGameState):
        return baseScoreStrategy(gameState, self._score)

    def _score(self, gameState: AbstractGridGameState):
        return self.__call__(gameState.asTensor()[None, :]).numpy()[0,0]

    def train_step(self, gameStateTensor: tf.Tensor):
        if self.bootstrapMode:
            return self.train_bootstrap_from_tensor(gameStateTensor)
        else:
            gameState = self.gameStateFromTensor(gameStateTensor)
            return self.train_td_from_game(gameState)

    def train_bootstrap_from_tensor(self, tensor: tf.Tensor):
        x = tensor
        y = tf.constant([
            self.bootstrapScoringAgent.score(self.gameStateFromTensor(tensor))
        ])
        with tf.GradientTape() as tape:
            y_pred = self(x[None, :], training=True)
            loss = self.compute_loss(x, y, y_pred)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, None)

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


        gradients = []

        for tensor in gameStateTensors:
            with tf.GradientTape() as tape:
                tfModelScore = self.__call__(tensor[None, :])
            gradients.append(
                tape.gradient(tfModelScore, self.trainable_variables)
            )

        delta_trainable = self.td_gradient_list(deltas, gradients)

        self.optimizer.apply_gradients(zip(delta_trainable, self.trainable_variables))

        self.compute_loss(None, tf.constant([scores[0]]), tf.constant([scores[-1]]))

        return self.compute_metrics(gameStateTensors[0], scores[-1], scores[0], None)

    

class TD_model(tf.keras.Model):
    def __init__(self, *args, td_factor = 0.7, **kwargs) -> None:
        self.td_factor = td_factor
        super().__init__(*args, **kwargs)

    def train_td_from_sequential_states(self, tensor_states: List[tf.Tensor], scores: np.ndarray):
        # Expect arguments where:
        # len(tensor_states) = len(scores)
        deltas = scores[1:] - scores[:-1]
        gradients = []
        for tensor in tensor_states[:-1]:
            with tf.GradientTape() as tape:
                tfModelScore = self.__call__(tensor[None, :])
            gradients.append(
                tape.gradient(tfModelScore, self.trainable_variables)
            )
        delta_trainable = self.td_gradient_list(deltas, gradients)
        self.optimizer.apply_gradients(zip(delta_trainable, self.trainable_variables))


    def td_gradient_list(self, deltas: np.ndarray, gradientsList: List[List[tf.Tensor]]):
        # Expect arguments of form:
        # deltas[x]
        # gradientsList[y][x]
        # 0 < x <= X
        # 0 < y <= Y
        gradientsZip = zip(*gradientsList)
        dw = [self.weight_gradients_by_temporal_difference(deltas, g) for g in gradientsZip]
        return dw

    def generate_temporal_difference_weights(self, deltas: np.ndarray):
        powers = np.arange(deltas.size)[::-1]
        per_step_weights = np.power(self.td_factor, powers)
        per_delta_weights = np.cumsum(per_step_weights)
        return deltas * per_delta_weights

    def weight_gradients_by_temporal_difference(self, deltas: List[int], gradients: List[tf.Tensor]):
        # len(deltas) == len(gradients)
        td_factor = 0.7
        dw_init = tf.zeros(gradients[0].shape)
        dw = tf.Variable(dw_init, trainable=False)
        trace = tf.Variable(0.0, trainable=False)

        M = len(deltas)
        for t in range(M):
            trace.assign(tf.constant(0.0))
            for j in range(t, M):
                trace.assign_add( td_factor**(j-t) * deltas[t] )
            dw.assign_add(trace * gradients[t])
        return dw
