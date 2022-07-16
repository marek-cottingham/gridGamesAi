import numpy as np
from scipy.special import expit

from ..common import handle_wins_draws, handle_wins_draws_method
from ..scoringAgents import AbstractScoringAgent
from .gameState import PentagoGameState, _gridOccupancy

class PentagoNaiveScoringAgent(AbstractScoringAgent):

    @handle_wins_draws_method
    def score(self, gameState: PentagoGameState) -> float:
        expitScale = 0.05
        # squish the score to be between -1 and 1 using a sigmoid function
        return 2*(expit(self.unsquished_score(gameState)*expitScale) - 0.5)

    def unsquished_score(self, gameState: PentagoGameState) -> int:
        grid_0_occupancy = _gridOccupancy(gameState.grid_0)
        grid_1_occupancy = _gridOccupancy(gameState.grid_1)
        return np.sum(grid_0_occupancy**2) - np.sum(grid_1_occupancy**2)