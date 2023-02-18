from pathlib import Path
import os
import math
from time import time
from dataclasses import dataclass
from typing import ClassVar

from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.agents import SemiRandomAgent
from gridGamesAi.paths import NGO_MODELS_DIR
import matplotlib.pyplot as plt
import numpy as np

from .gameState import NgoGameRunner, NgoGameState
from .temporalDifferenceModel import Ngo_TD_Agent
from .resolvedPositions import ResolvedPositions

def iterations_from_model_path(path: Path):
    return int(path.stem)

@dataclass
class ModelManager():
    base_path: Path = NGO_MODELS_DIR / "unnammed_model_1"
    game_runner: NgoGameRunner = NgoGameRunner(3, 5, True)
    random_moves_on_game_initialisation: int = 6
    new_save_file_after_training_calls: int = 8000
    save_after_traning_calls: int = 40
    max_training_calls: int = 120000
    ml_agent_class: ClassVar = Ngo_TD_Agent

    def __post_init__(self):
        self.ml_agent: Ngo_TD_Agent = None
        self.current_model_path = None
        self.randomise_model_moves_at_start = True
        self.training_model_moves_at_start = 0
        self.training_random_moves = 1
        self.base_starting_game_state = None

        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def update_save_path(self):
        self.current_model_path = self.base_path / str(
            (self.ml_agent.training_calls // self.new_save_file_after_training_calls + 1) *
            self.new_save_file_after_training_calls
        )

    def new_model(self):
        self.ml_agent = self.ml_agent_class(None, self.game_runner)

    def save_model(self):
        self.update_save_path()
        os.makedirs(self.current_model_path.parent, exist_ok=True)
        self.ml_agent.save(self.current_model_path, True)

    def load_model(self, verbose: bool = False):
        self.ml_agent = self.ml_agent_class(None, None)
        self.ml_agent.load(self.current_model_path, verbose)
        self.ml_agent.compile_td_model()

    def get_sorted_model_paths(self):
        all_model_paths = [path for path in self.base_path.iterdir() if path.is_dir()]
        all_model_paths.sort(key = iterations_from_model_path)
        return all_model_paths
    
    def load_latest_model(self):
        all_model_paths = self.get_sorted_model_paths()
        self.current_model_path = all_model_paths[-1]
        self.load_model()

    def train(self):
        start = time()
        while self.ml_agent.training_calls < 120000:
            if self.base_starting_game_state is None:
                self.base_starting_game_state = NgoGameState.init_with_n_agent_and_m_random_moves(
                    self.training_model_moves_at_start,
                    0,
                    SemiRandomAgent(MinimaxAgent(self.ml_agent, 0), 0.1),
                    self.game_runner
                )
            gs = self.base_starting_game_state.n_random_moves(self.training_random_moves)
            self.ml_agent.train_td_from_game(gs)
            
            if self.ml_agent.training_calls % self.save_after_traning_calls == 0:
                self.save_model()
                print("Time for 40 calls:", time() - start)
                start = time()

            if self.ml_agent.training_calls % (4) == 0:
                if self.randomise_model_moves_at_start:
                    self.training_model_moves_at_start = np.random.randint(
                        0, 
                        max(1, self.ml_agent.training_maxMoves - self.training_random_moves - 1)
                    )
                self.base_starting_game_state = None

    def rate_against_resolved_positions(self, resolved_positions: ResolvedPositions):
        sum_square_err = 0
        for position, expected_value in resolved_positions.positions:
            score = self.ml_agent.model_score(position)
            square_err = (score - expected_value) ** 2
            sum_square_err += square_err
        return sum_square_err / len(resolved_positions.positions)
            

