from pathlib import Path
import os
import math
from time import time
from dataclasses import dataclass
from typing import ClassVar

from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import NGO_MODELS_DIR
import matplotlib.pyplot as plt
import numpy as np

from .gameState import NgoGameRunner, NgoGameState
from .temporalDifferenceModel import Ngo_TD_Agent

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

    def load_model(self):
        self.ml_agent = self.ml_agent_class(None, None)
        self.ml_agent.load(self.current_model_path, True)
        self.ml_agent.compile_td_model()
    
    def load_latest_model(self):
        all_model_paths = [path for path in self.base_path.iterdir() if path.is_dir()]
        all_model_paths.sort(key = iterations_from_model_path)
        self.current_model_path = all_model_paths[-1]
        self.load_model()

    def train(self):
        start = time()
        while self.ml_agent.training_calls < 120000:
            gs = NgoGameState.init_with_n_random_moves(6, self.game_runner)
            self.ml_agent.train_td_from_game(gs)
            
            if self.ml_agent.training_calls % self.save_after_traning_calls == 0:
                self.save_model()
                print("Time for 40 calls:", time() - start)
                start = time()

    def rate_against_resolved_positions(self):
        pass
