from functools import cache
import json
from pathlib import Path
from typing import List, Tuple
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gridGamesAi.agents import AbstractAgent
from gridGamesAi.common import AbstractGameState, AbstractGridGameState, baseScoreStrategy
from gridGamesAi.game import Game
from gridGamesAi.minimax import MinimaxAgent, PruningAgent
from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.pentago.scoringAgent import PentagoNaiveScoringAgent
from gridGamesAi.scoringAgents import AbstractScoringAgent, CachingScoringAgent

CAN_GO_SELF_PLAY = False

try:
    from gridGamesAi.pentago.go_interface import goSelfPlay
    CAN_GO_SELF_PLAY = True
except Exception as e:
    print(e)

class Pentago_TD_Agent(CachingScoringAgent):
    def __init__(
        self,
        loadPath = None,
        createNewModel = False,
        minixmaxAgent = MinimaxAgent(None, max_depth=0),
    ) -> None:
        """ Initialize the agent. If loadPath is provided, td model 
        will be loaded from that path. If createNewModel is True and loadPath
        is None, a new model will be created. """
        
        self.td_model = None
        self.isCompiled = False

        self.minimaxAgent: MinimaxAgent = minixmaxAgent
        self.minimaxAgent.scoringAgent = self

        self.training_calls = 0
        self.trainingCall_totalMoves: List[Tuple[int, int]] = []

        if loadPath is not None:
            self.load(loadPath)
        if createNewModel and self.td_model is None:
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

    def load(self, path: Path, verbose=False):
        if verbose:
            print(f"Loading TD model from {path}")
        self._load_td_model(path)
        loadRecordSuccess = self._try_load_training_record(path)
        if verbose and loadRecordSuccess:
            print("    Found and loaded training calls record file")
        if verbose:
            print(f"    Previous model training calls: {self.training_calls}")

    def _try_load_training_record(self, path):
        try:
            with open(path.parent / (path.stem + ".json")) as f:
                obj = json.load(f)
                self.training_calls = obj["training_calls"]
                self.trainingCall_totalMoves = obj["trainingCall_totalMoves"]
                return True
        except FileNotFoundError:
            return False

    def _load_td_model(self, path):
        self.td_model: TD_model = tf.keras.models.load_model(
            path,
            custom_objects={
                "pentagoTD_model": TD_model,
                "TD_model": TD_model,
            }
        )

    def save(self, path: Path, overwrite_ok=False):
        self._save_td_model(path, overwrite_ok)
        self._save_training_record(path)

    def _save_training_record(self, path):
        json_path = path.parent / (path.stem + ".json")
        with open(json_path, "w") as f:
            json.dump({
                "training_calls": self.training_calls,
                "trainingCall_totalMoves": self.trainingCall_totalMoves
            },f)

    def _save_td_model(self, path, overwrite_ok):
        if (overwrite_ok or not path.exists()):
            self.td_model.save(path)
        elif path.exists():
            raise Exception(
                "Model path already exists, set overwrite_ok to True to overwrite")

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
        return self.td_model.__call__(
            gameState.flipCenterOfMassToUpperLeftBelowDiagonal().asTensor()[None, :]
        ).numpy()[0,0]

    def resetCache(self):
        self.score.cache_clear()

    def train_td_from_game(self, rootGameState: PentagoGameState, usePythonNative = False):
        if usePythonNative or not CAN_GO_SELF_PLAY:
            movesSequence = self._generate_self_play_moves_sequence(rootGameState)
        else:
            movesSequence = goSelfPlay(rootGameState, 0, self.score)
            print(f"Game {self.training_calls} ", end="\r")

        gameStateTensors = [
            gameState.flipCenterOfMassToUpperLeftBelowDiagonal().asTensor() 
            for gameState in movesSequence
        ]
        scores = np.array([self.score(gameState) for gameState in movesSequence])

        self.td_model.train_td_from_sequential_states(gameStateTensors, scores)
        self.resetCache()
        self._update_training_record(movesSequence)

    def _update_training_record(self, movesSequence: List[PentagoGameState]):
        self.training_calls += 1
        self.trainingCall_totalMoves.append(
            (
                self.training_calls,
                int(movesSequence[-1].turnTracker.total_moves), 
            )
        )

    def _generate_self_play_moves_sequence(self, rootGameState):
        movesSequence: List[PentagoGameState] = [rootGameState]
        while not movesSequence[-1].isEnd:
            print(
                f"Game {self.training_calls} ",
                f"- Total moves: {movesSequence[-1].turnTracker.total_moves}     ",
                end="\r")
            movesSequence.append(self.minimaxAgent.move(movesSequence[-1]))
        return movesSequence

    def plot_training_total_moves(self):
        x = [i[0] for i in self.trainingCall_totalMoves]
        y = [i[1] for i in self.trainingCall_totalMoves]
        df = pd.DataFrame({"training_run": x, "moves": y})
        df['moves_avg'] = df['moves'].rolling(window=100).mean()
        df['moves_10'] = df['moves'].rolling(window=100).quantile(0.1, 'linear')
        df['moves_90'] = df['moves'].rolling(window=100).quantile(0.9, 'linear')
        plt.plot(df['training_run'], df['moves_avg'])
        plt.fill_between(
            df['training_run'], 
            df["moves_10"],
            df["moves_90"],
            alpha=0.2
        )
        plt.xlabel("Training run")
        plt.ylabel(f"Moves (rolling average) - Shaded: 10% and 90% quantiles")
        plt.show()

class TD_model(tf.keras.Model):
    def __init__(self, *args, td_factor = 0.7, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.td_factor = td_factor
        self.batch_size = 20
        self.batch_pending = []

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
        if self.batch_size == 1:
            self.optimizer.apply_gradients(zip(delta_trainable, self.trainable_variables))
        else:
            self.batch_pending.append(delta_trainable)

        if len(self.batch_pending) >= self.batch_size:
            sum_delta_trainable = [tf.add_n(i) for i in zip(*self.batch_pending)]
            self.optimizer.apply_gradients(zip(sum_delta_trainable, self.trainable_variables))
            self.batch_pending = []


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
