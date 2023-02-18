from gridGamesAi.minimax import MinimaxAgent
from gridGamesAi.paths import NGO_MODELS_DIR
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from .gameState import NgoGameRunner, NgoGameState
from .temporalDifferenceModel import Ngo_TD_Agent

class ResolvedPositions:
    def __init__(self, game_runner: NgoGameRunner = None):
        if game_runner is None:
            game_runner = NgoGameRunner(3, 5, True)
        self.game_runner = game_runner
        self.positions: List[tuple[NgoGameState, float]] = []

    def add_winning_positions(self, n: int, player: int, extra_moves: int):
        for _ in range(n):
            gs = NgoGameState.init_as_winning_position(self.game_runner, player, extra_moves)
            if player == 0:
                self.positions.append((gs, 1.0))
            if player == 1:
                self.positions.append((gs, -1.0))