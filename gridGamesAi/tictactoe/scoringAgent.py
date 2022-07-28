from ..common import handle_wins_draws, handle_wins_draws_method
from ..scoringAgents import AbstractScoringAgent
from .gameState import TicTacToeGameState

class TicTacToeManualScoringAgent(AbstractScoringAgent):

    @handle_wins_draws_method
    def score(self, gameState: TicTacToeGameState) -> float:
        return 0.2 * (
        gameState.grid_0[0][0] + gameState.grid_0[0][2] + 
        gameState.grid_0[2][0] + gameState.grid_0[2][2]
        - gameState.grid_1[1][1]
    )

    @property
    def hasCachingFacility(self) -> bool:
        return False