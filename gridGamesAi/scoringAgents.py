from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import List
import copy

from .common import AbstractGameState, SavedScoreInterface
class AbstractScoringAgent(ABC):

    @abstractmethod
    def score(self,gameState: AbstractGameState) -> float: pass

class CachingScoringAgent(AbstractScoringAgent):
    def hasCachingFacility(self) -> bool: return True

    @abstractmethod
    def resetCache(self): pass

class OnGameStateCachingScoringAgent(AbstractScoringAgent):
    nextSerialNumber = 0

    def __init__(self):
        self.serialNumber = OnGameStateCachingScoringAgent.nextSerialNumber
        OnGameStateCachingScoringAgent.nextSerialNumber += 1

    def score(self, gameState: SavedScoreInterface) -> float:
        if self.serialNumber in gameState.savedScores:
            return gameState.savedScores[self.serialNumber]
        else:
            score = self._score(gameState)
            gameState.savedScores[self.serialNumber] = score
            return score

    @abstractmethod
    def _score(self, gameState: AbstractGameState) -> float: pass

def sortNextMovesAscendingWithScoringAgent(
    gameState: AbstractGameState, agent: AbstractScoringAgent
) -> List[AbstractGameState]:
    next_moves = copy.copy(gameState.next_moves)
    next_moves.sort(key = lambda x: agent.score(x))
    return next_moves

def sortNextMovesDescendingWithScoringAgent(
    gameState: AbstractGameState, agent: AbstractScoringAgent
) -> List[AbstractGameState]:
    return sortNextMovesAscendingWithScoringAgent(gameState, agent)[::-1]