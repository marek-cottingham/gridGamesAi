from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import List
import copy

from .common import AbstractGameState

class AbstractScoringAgent(ABC):

    @abstractmethod
    def score(self,gameState: AbstractGameState) -> float: pass

    @abstractproperty
    def hasCachingFacility(self) -> bool: pass

class CachingScoringAgent(AbstractScoringAgent):
    def hasCachingFacility(self) -> bool: return True

    @abstractmethod
    def resetCache(self): pass

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