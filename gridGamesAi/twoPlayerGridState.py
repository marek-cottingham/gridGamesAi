from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from functools import cache, cached_property

import numpy as np


def _placeOnGrid(grid: np.ndarray, index: Tuple[int, int]):
    new_grid = grid.copy()
    new_grid[index] = 1
    return new_grid

@cache
def get_COM_weights(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    x, y = shape
    weightsX = np.arange(x) - x/2 + 0.5
    weightsX = np.tile(weightsX, (y,1))
    weightsY = np.arange(y) - y/2 + 0.5
    weightsY = np.transpose(np.tile(weightsY, (x,1)))
    return weightsX, weightsY

@dataclass
class TwoPlayerGridState:
    """Represents a playing grid for two players, where only one
    player can place a piece on each gridspace.
    
    Expects that grid_0.shape == grid_1.shape """

    grid_0: np.ndarray
    grid_1: np.ndarray

    @cached_property
    def combined_grid(self) -> np.ndarray:
        return self.grid_0 + self.grid_1

    @property
    def grid_is_occupied(self) -> np.ndarray:
        return self.combined_grid == 1

    def placeOn(self, player, index: Tuple[int, int]) -> TwoPlayerGridState:
        if player == 0:
            return TwoPlayerGridState(
                _placeOnGrid(self.grid_0, index),
                self.grid_1
            )
        if player == 1:
            return TwoPlayerGridState(
                self.grid_0,
                _placeOnGrid(self.grid_1, index)
            )

    def nextValidPlacements(self, player) -> List[TwoPlayerGridState]:
        next_grid_states = []
        for index, is_occupied in np.ndenumerate(self.grid_is_occupied):
            if not is_occupied:
                next_state = self.placeOn(player, index)
                next_grid_states.append(next_state)
        return next_grid_states

    @cached_property
    def comX(self) -> int:
        weightsX, weightsY = get_COM_weights(self.combined_grid.shape)
        return np.sum(weightsX * self.combined_grid)
    
    @cached_property
    def comY(self) -> int:
        weightsX, weightsY = get_COM_weights(self.combined_grid.shape)
        return np.sum(weightsY * self.combined_grid)

    @property
    def comXY(self) -> int:
        "Centre of mass along the [X, -Y] axis"
        return self.comX - self.comY

    def flipCenterOfMassToUpperLeftBelowDiagonal(self) -> TwoPlayerGridState:
        """Flips the grid state to a symetric grid state where the centre of mass
        the placed pieces is located in the upper left quadrant and below the main
        diagonal. """

        comXY = self.comXY
        new_grid_0 = self.grid_0
        new_grid_1 = self.grid_1

        if self.comX > 0:
            new_grid_0 = np.fliplr(new_grid_0)
            new_grid_1 = np.fliplr(new_grid_1)
            comXY = -comXY
        if self.comY > 0:
            new_grid_0 = np.flipud(new_grid_0)
            new_grid_1 = np.flipud(new_grid_1)
            comXY = -comXY
        if comXY > 0:
            new_grid_0 = np.transpose(new_grid_0)
            new_grid_1 = np.transpose(new_grid_1)

        return TwoPlayerGridState(new_grid_0, new_grid_1)

    def assert_unoccupied(self, index):
        if self.grid_is_occupied[index]:
            raise ValueError(f"Position is already occupied: {index}")