from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .common import AbstractGameState

class AbstractAgent(ABC):
    @abstractmethod
    def move(self, gameState: AbstractGameState) -> AbstractGameState:
        """Return the game state after making the next move"""

class RandomAgent(AbstractAgent):
    """An agent which randomly chosses to play a valid move in a game from the 
    set of available valid moves"""
    def __init__(self):
        self.rng = np.random.default_rng()

    def move(self, gameState: AbstractGameState) -> AbstractGameState:
        return self.rng.choice(
            gameState.next_moves
        )

class SemiRandomAgent(AbstractAgent):
    def __init__(self, subAgent: AbstractAgent, randomChance: float = 0.1):
        self.rng = np.random.default_rng()
        self.subAgent = subAgent
        self.randomChance = randomChance

        self.randomAgent = RandomAgent()

    def move(self, gameState: AbstractGameState):
        if self.rng.uniform() < self.randomChance:
            return self.randomAgent.move(gameState)
        else:
            return self.subAgent.move(gameState)