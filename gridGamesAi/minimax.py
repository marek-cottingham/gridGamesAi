import numpy as np

from .agents import AbstractAgent
from .common import AbstractGameState
from .scoringAgents import AbstractScoringAgent, sortNextMovesAscendingWithScoringAgent, sortNextMovesDescendingWithScoringAgent

class PruningAgent(AbstractAgent):
    """Implements minimax algorithm with alpha-beta pruning"""

    def __init__(self, scoringAgent: AbstractScoringAgent, max_depth = 6):
        self.scoringAgent = scoringAgent
        self.max_depth = max_depth

    def move(self, gameState: AbstractGameState):

        alpha = -1
        beta = 1
        bestMove = None

        isMaximisingPlayer = gameState.current_player == 0

        if isMaximisingPlayer:
            value = -1
            for nextState in sortNextMovesDescendingWithScoringAgent(gameState, self.scoringAgent):
                ab = self.alphabeta(nextState, self.max_depth, alpha, beta)
                if ab > value:
                    value = ab
                    bestMove = nextState
                alpha = max(alpha, value)

        if not isMaximisingPlayer:
            value = 1
            for nextState in sortNextMovesAscendingWithScoringAgent(gameState, self.scoringAgent):
                ab = self.alphabeta(nextState, self.max_depth, alpha, beta)
                if ab < value:
                    value = ab
                    bestMove = nextState
                beta = min(beta, value)

        return bestMove

    def alphabeta(self, 
        gameState: AbstractGameState, 
        depth: int, 
        alpha: float, 
        beta: float
    ):
        if gameState.isEnd or depth == 0:
            return self.scoringAgent.score(gameState)
        
        isMaximisingPlayer = gameState.current_player == 0

        if isMaximisingPlayer:
            value = -1
            for nextState in sortNextMovesDescendingWithScoringAgent(gameState, self.scoringAgent):
                value = max(value, self.alphabeta(nextState, depth-1, alpha, beta))
                if value >= beta:
                    break
                alpha = max(alpha, value)
            return value

        if not isMaximisingPlayer:
            value = 1
            for nextState in sortNextMovesAscendingWithScoringAgent(gameState, self.scoringAgent):
                value = min(value, self.alphabeta(nextState, depth-1, alpha, beta))
                if value <= alpha:
                    break
                beta = min(beta, value)
            return value



class MinimaxAgent(AbstractAgent):
    """Minimax scoring of a game's current state, using a depth limited
    minimax algorithm"""

    def __init__(self, scoringAgent: AbstractScoringAgent, max_depth: int = 4):
        self.scoringAgent = scoringAgent
        self.max_depth = max_depth

    def move(self, currentGameState: AbstractGameState):
        """Selects the best next move and updates the game,
        based on the minimax algorithm"""

        scores = [
            self.minimax(state,self.max_depth) for state in currentGameState.next_moves
        ]
        nextMove = currentGameState.next_moves[np.argmax(scores)] 
        return nextMove

    def minimax(self, gameState: AbstractGameState, depth: int) -> float:
        """Scores gameState from the perspective of the last player to move 
        (higher is better move) by performing a minimax search to the specified 
        depth."""

        exitCondition = gameState.isEnd or depth == 0
        if exitCondition:
            if gameState.last_player_to_move == 0:
                return self.scoringAgent.score(gameState)
            else:
                return -self.scoringAgent.score(gameState)

        scores = [self.minimax(state, depth-1) for state in gameState.next_moves]
        bestScoreNextMove = np.max(scores)

        if gameState.last_player_to_move != gameState.current_player:
            return -bestScoreNextMove
        else:
            return bestScoreNextMove

def next_moves_were_evaluated(gameState: AbstractGameState):
    """Determines if the next moves and their scores have been computed 
    for a given game state."""
    if 'next_moves' in gameState.__dict__ is None:
        return False
    for nextState in gameState.next_moves:
        if nextState._score is None:
            return False
    return True
    
# class TimeBoundMinimaxAgent(AbstractAgent):
#     """Minimax scoring of a game's current state, which attempts to prioritise
#     searching the most informative branches while limiting the time taken to
#     perform the search."""

#     def __init__(self, maxSeconds = 2, distance_penalty_const = 0.2):
#         """Initialises TimeBoundMinimaxAgent.

#         :param maxSeconds:
#             Time before algorithm should move to the final stage
#         :param distance_penalty_const:
#             How strongly moves many steps ahead should be penalised when chossing
#             the next game state to expand. 
#         """
#         self.maxSeconds = maxSeconds
#         self.distance_penalty_const = distance_penalty_const

#     def update(self, game: Game):
#         """Selects the best next move and updates the game."""

#         startTime = time()

#         # List of game states who's next moves have yet to be computed and/or scored
#         frontier = [game.currentGameState]
#         frontier = self.expand_to_scored_game_states(frontier)

#         while time() < startTime + self.maxSeconds and len(frontier) > 0:
#             frontier = self.expand_best_unscored_game_state(
#                 game.currentGameState, frontier
#             )
       
#         # Count the number of states on the frontier for each unique value
#         # of gameState.total_moves
#         total_moves_counts = np.unique(
#             [gameState.total_moves for gameState in frontier],
#             return_counts = True
#         )
#         # print(f"Frontier total_moves value counts: {total_moves_counts}")

#         scores = [
#             self.no_new_eval_minimax(state) 
#             for state in game.currentGameState.next_moves
#         ]
#         nextMove = game.currentGameState.next_moves()[np.argmax(scores)] 
#         super()._update(game, nextMove)

#     def expand_best_unscored_game_state(self, 
#         rootState: GameState, frontier: List[GameState]
#     ):
#         """Expands the frontier by finding the next moves for a game state and
#         scoring them."""
#         frontier_scores = []
#         for gameState in frontier:

#             # As expand_suitability_score depends on rootState.total_moves
#             # include this in the lookup key for caching this operation
#             caching_key = f"minimax_next_score_{rootState.total_moves}"

#             operation = partial( self.expand_suitability_score, rootState)
#             minixmax_next_score = gameState.cachedOperation(caching_key, operation)
#             frontier_scores.append(minixmax_next_score)

#         next_expand_index = np.argmax(frontier_scores)
#         next_expand = frontier[next_expand_index]
#         frontier.remove(next_expand)
#         frontier += next_expand.next_moves()
#         for gameState in next_expand.next_moves():
#             gameState.score()
#         return frontier

#     def expand_to_scored_game_states(self, frontier: List[GameState]):
#         """Expands the frontier to game states where next_moves and the score
#         of these moves have already been evaluated and cached."""
#         last_update_len = 1
#         while last_update_len > 0:
#             update_add = []
#             update_rm = []
#             for gameState in frontier:
#                 if next_moves_were_evaluated(gameState):
#                     update_rm.append(gameState)
#                     update_add += gameState.next_moves()
#             for gameState in update_rm:
#                 frontier.remove(gameState)
#             frontier += update_add
#             last_update_len = len(update_rm)
#         return frontier

#     def expand_suitability_score(self, rootGameState: GameState, gameState: GameState):
#         """Determine how suitable a gamestate is to be expanded on the next
#         minimax operation, where 0 is the worst and 2 is the best possible score."""

#         # Cannot expand a game which has ended
#         if gameState.win() or gameState.draw():
#             return 0

#         # Penalise expanding many moves ahead
#         distance_penalty = np.exp(
#             (rootGameState.total_moves - gameState.total_moves) * self.distance_penalty_const
#         )

#         if gameState.last_player_to_move == 0:
#             return (gameState.score()+1) * distance_penalty
#         else:
#             return (-gameState.score()+1) * distance_penalty

#     def no_new_eval_minimax(self, gameState: GameState):
#         """Minimax search which doesn't perform any new evaluations of
#         gameState.next_moves or gameState.score."""

#         exitCondition = (
#             gameState.win() or gameState.draw()
#             or not next_moves_were_evaluated(gameState)
#         )
#         if exitCondition:
#             if gameState.last_player_to_move == 0:
#                 return gameState.score()
#             else:
#                 return -gameState.score()

#         scores = [self.no_new_eval_minimax(state) for state in gameState.next_moves()]
#         bestScoreNextMove = np.max(scores)

#         if gameState.last_player_to_move != gameState.current_player:
#             return -bestScoreNextMove
#         else:
#             return bestScoreNextMove
