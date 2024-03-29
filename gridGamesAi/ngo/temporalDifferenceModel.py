from functools import cache
import json
from pathlib import Path
from typing import List, Tuple
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gridGamesAi.common import baseScoreStrategy
from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.ngo.gameState import NgoGameRunner, NgoGameState
from gridGamesAi.scoringAgents import OnGameStateCachingScoringAgent

class Ngo_TD_Agent(OnGameStateCachingScoringAgent):
    def __init__(
        self,
        loadPath = None,
        newModelRunner = None,
        minixmaxAgent = MinimaxAgent(None, max_depth=0),
    ) -> None:
        """ Initialize the agent. If loadPath is provided, td model 
        will be loaded from that path. If newModelRunner is provided and loadPath
        is None, a new model will be created. """

        self.incrementSerial()
        
        self.td_model: TD_model = None
        self.isCompiled = False

        self.minimaxAgent: MinimaxAgent = minixmaxAgent
        self.minimaxAgent.scoringAgent = self

        self.training_calls = 0
        self.trainingCall_totalMoves: List[Tuple[int, int]] = []
        self.training_maxMoves = 0

        if loadPath is not None:
            self.load(loadPath)
        if newModelRunner is not None and self.td_model is None:
            self.create_td_model(newModelRunner)
        if self.td_model is not None:
            self.compile_td_model()

    def create_td_model(self, templateRunner: NgoGameRunner):
        """ Create a new TD model using the templateRunner to describe the shape of the input. """
        size = templateRunner.size_board ** 2 * 2 + 2
        inputs = tf.keras.Input(shape=(size,))
        internal_1 = tf.keras.layers.Dense(size, activation='relu')
        internal_2 = tf.keras.layers.Dense(size//2, activation='relu')
        output = tf.keras.layers.Dense(1, activation='tanh')
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
                if "training_maxMoves" in obj:
                    self.training_maxMoves = obj["training_maxMoves"]
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
                "trainingCall_totalMoves": self.trainingCall_totalMoves,
                "training_maxMoves": self.training_maxMoves,
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

    def _score(self, gameState: NgoGameState) -> float:
        return baseScoreStrategy(gameState, self.model_score)

    def model_score(self, gameState: NgoGameState):
        return self.td_model.__call__(
            gameState.asSingleTensor()[None, :]
        ).numpy()[0,0]

    def train_td_from_game(self, rootGameState: NgoGameState):
        movesSequence = self._generate_self_play_moves_sequence(rootGameState)

        gameStateTensors = [
            gameState.asSingleTensor() 
            for gameState in movesSequence
        ]
        scores = np.array([self.score(gameState) for gameState in movesSequence])

        self.td_model.train_td_from_sequential_states(gameStateTensors, scores)
        self.incrementSerial()
        self._update_training_record(movesSequence)

    def _update_training_record(self, movesSequence: List[NgoGameState]):
        moves = int(movesSequence[-1].turnTracker.total_moves)
        self.training_calls += 1
        self.trainingCall_totalMoves.append(
            (
                self.training_calls,
                moves
            )
        )
        self.training_maxMoves = max(self.training_maxMoves, moves)

    def _generate_self_play_moves_sequence(self, rootGameState):
        movesSequence: List[NgoGameState] = [rootGameState]
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

class Ngo_TD_Agent_v1b(Ngo_TD_Agent):
    def model_score(self, gameState: NgoGameState):
        return self.td_model.__call__(
            gameState.asSingleTensor()[None, :] * 2 - 1
        ).numpy()[0,0]


class TD_model(tf.keras.Model):
    def __init__(self, *args, td_factor = 0.7, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.td_factor = td_factor
        self.batch_size = 1
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
            return
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
