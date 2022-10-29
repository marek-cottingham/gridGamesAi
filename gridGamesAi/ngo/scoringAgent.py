from scipy.special import expit

from ..common import  handle_wins_draws_method
from ..scoringAgents import  OnGameStateCachingScoringAgent
from .gameState import NgoGameState

class NgoNaiveScoringAgent(OnGameStateCachingScoringAgent):

    @handle_wins_draws_method
    def _score(self, gameState: NgoGameState) -> float:
        expitScale = 0.05
        # squish the score to be between -1 and 1 using a sigmoid function
        return 2*(expit(self.unsquished_score(gameState)*expitScale) - 0.5)

    def unsquished_score(self, gameState: NgoGameState) -> int:
        return gameState.gameRunner.naiveScore(gameState.grid).numpy()