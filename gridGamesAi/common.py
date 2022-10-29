from __future__ import annotations

import functools
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, List, Tuple

import numpy as np


class AbstractGameState(ABC):
    @abstractproperty
    def isWin(self) -> bool: pass

    @abstractproperty
    def isDraw(self) -> bool: pass

    @abstractproperty
    def winPlayer(self) -> int: pass

    @abstractproperty
    def isEnd(self) -> bool: pass

    @abstractproperty
    def next_moves(self) -> List[AbstractGameState]: pass

    @abstractproperty
    def current_player(self) -> int: pass

    @abstractproperty
    def turn_step(self) -> int: pass

    @abstractproperty
    def last_player_to_move(self) -> int: pass

class SavedScoreInterface(ABC):
    @abstractproperty
    def savedScores(self) -> dict[int, float]: pass

class AbstractGridGameState(AbstractGameState):
    @abstractproperty
    def grid_0(self) -> np.ndarray: pass

    @abstractproperty
    def grid_1(self) -> np.ndarray: pass

    @abstractmethod
    def place(self, index: Tuple[int,int]) -> AbstractGridGameState: pass

class AbstractRotationGridGameState(AbstractGridGameState):
    @abstractmethod
    def rotate(self, rot: str) -> AbstractRotationGridGameState: pass

def baseScoreStrategy(gameState: AbstractGameState, notEndScoreStrategy: Callable) -> float:
    """Score is 1 if player 0 wins, -1 if player 1 wins and 0 in case of a draw.
    
    nonexitScoreStrategy: a strategy which will score the cases which are not wins or draws
    """
    if gameState.isWin:
        if gameState.winPlayer == 0:
            return 1.0
        if gameState.winPlayer == 1:
            return -1.0

    if gameState.isDraw:
        return 0.0
    
    return notEndScoreStrategy(gameState)

def handle_wins_draws(func: Callable) -> Callable:
    """Decorates a scoring function to handle wins and draws correctly"""
    @functools.wraps(func)
    def wrapper_handle_wins_draws(gameState: AbstractGameState) -> float:
        return baseScoreStrategy(gameState, func)
    return wrapper_handle_wins_draws

def handle_wins_draws_method(func: Callable) -> Callable:
    """Decorates a scoring method to handle wins and draws correctly"""
    @functools.wraps(func)
    def wrapper_handle_wins_draws(self, gameState: AbstractGameState) -> float:
        return baseScoreStrategy(gameState, functools.partial(func, self))
    return wrapper_handle_wins_draws
