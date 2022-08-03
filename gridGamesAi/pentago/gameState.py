from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, cache
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import hashlib

from ..turnTracker import TurnTracker
from ..twoPlayerGridState import TwoPlayerGridState
from .rotations import rotations, rotationsKeys
from ..common import AbstractGridGameState

def _gridOccupancy(grid: np.ndarray) -> List[int]:
    """Evaluates the number of pieces a player has on each row, column or diagonal
    of 5. Used in evaluating win states or in scoring a position using a naive algorithm.

    param: grid
        shape (6,6) ndarray representing pieces placed by the player
    """
    flipped_grid: np.ndarray = np.fliplr(grid)
    occupacy = np.concatenate((
        #horizontal lines occupancy
        np.sum(grid[0:5], axis=0), np.sum(grid[1:], axis=0),
        #vertical lines occupancy
        np.sum(grid[:, 0:5], axis=1), np.sum(grid[:, 1:], axis=1),
        [
            #diagonals
            np.sum(grid.diagonal(-1)), np.sum(grid.diagonal(1)),
            np.sum(grid.diagonal()[0:5]), np.sum(grid.diagonal()[1:]),
            #anti-diagonals
            np.sum(flipped_grid.diagonal(-1)), np.sum(flipped_grid.diagonal(1)),
            np.sum(flipped_grid.diagonal()[0:5]), np.sum(flipped_grid.diagonal()[1:]),
        ]
    ))
    return occupacy

def _gridInWinState(grid: np.ndarray) -> bool:
    """Does the grid contain a winning line of 5 pieces?

    param: grid
        shape (6,6) ndarray representing pieces placed by the player
    """
    return np.max(_gridOccupancy(grid)) == 5

def _rotatePentagoGrid(gridState: TwoPlayerGridState, rotationKey: str):
    new_grid_0 = gridState.grid_0.flatten()[rotations[rotationKey]].reshape((6,6))
    new_grid_1 = gridState.grid_1.flatten()[rotations[rotationKey]].reshape((6,6))
    return TwoPlayerGridState(new_grid_0, new_grid_1)

@dataclass
class PentagoGameState(AbstractGridGameState):
    turnTracker: TurnTracker = TurnTracker(2,2)
    gridState: TwoPlayerGridState = TwoPlayerGridState(
        np.zeros((6,6)), np.zeros((6,6))
    )

    @property
    def current_player(self) -> int: return self.turnTracker.current_player
    
    @property
    def turn_step(self) -> int: return self.turnTracker.current_turn_step
    
    @property
    def last_player_to_move(self) -> int: return self.turnTracker.last_player_to_move

    @property
    def grid_0(self) -> np.array: return self.gridState.grid_0

    @property
    def grid_1(self) -> np.array: return self.gridState.grid_1

    def _getNextGridStates(self):
        if self.turnTracker.current_turn_step == 0:
            return self.gridState.nextValidPlacements(self.current_player)
        if self.turnTracker.current_turn_step == 1:
            return self._getNextRotateGridStates()

    def _getNextRotateGridStates(self):
        nextGridStates = []
        for key in rotationsKeys:
            nextGrid = _rotatePentagoGrid(self.gridState, key)
            nextGridStates.append(nextGrid)
        return nextGridStates

    def rotate(self, rotate_key: str) -> PentagoGameState:
        return PentagoGameState(
            self.turnTracker.getIncremented(),
            _rotatePentagoGrid(self.gridState, rotate_key)
        )

    def place(self, index: Tuple[int, int]) -> PentagoGameState:
        self.gridState.assert_unoccupied(index)
        return PentagoGameState(
            self.turnTracker.getIncremented(),
            self.gridState.placeOn(self.current_player, index)
        )

    def skipRotation(self) -> PentagoGameState:
        if self.turn_step != 1:
            raise ValueError("Can only skip rotation on second part of turn")
        return self.skipMove()

    def skipMove(self) -> PentagoGameState:
        return PentagoGameState(
            self.turnTracker.getIncremented(),
            self.gridState
        )

    @cached_property
    def next_moves(self) -> List[PentagoGameState]:
        next_grid_states = self._getNextGridStates()
        next_turn_tracker = self.turnTracker.getIncremented()
        return [PentagoGameState(next_turn_tracker, state) for state in next_grid_states]

    @cached_property
    def _winInfo(self) -> Tuple[bool, int | None]:
        win_player_0 = _gridInWinState(self.gridState.grid_0)
        win_player_1 = _gridInWinState(self.gridState.grid_1)

        if win_player_0 and win_player_1:
            # If a player rotates a segment so both players achieve 5 in a row simultaneously,
            # the player who just moved *losses*
            return True, self.turnTracker.other_to_last_player_to_move 
        if win_player_0:
            return True, 0
        if win_player_1:
            return True, 1
        return False, None

    @property
    def isWin(self) -> bool:
        return self._winInfo[0]

    @property
    def winPlayer(self) -> int | None:
        return self._winInfo[1]

    @cached_property
    def isDraw(self) -> bool:
        return (
            self.turn_step == 0
            and np.sum(self.gridState.grid_0) + np.sum(self.gridState.grid_1) >= 36 
            and not self.isWin
        )

    @property
    def isEnd(self) -> bool:
        return self.isDraw or self.isWin

    def asNumpy(self) -> np.ndarray:
        return np.concatenate([
            np.array([self.current_player, self.turn_step]),
            self.gridState.grid_0.flatten(),
            self.gridState.grid_1.flatten(),
        ])

    def asTensor(self) -> tf.Tensor:
        return tf.constant(
            self.asNumpy()
        )

    def __hash__(self) -> int:
        arr = self.asNumpy().view()
        hash = hashlib.sha1(arr).hexdigest()
        return int(hash,16)

    def __eq__(self, other: object) -> bool:
        return np.all(self.asNumpy() == other.asNumpy())

    def flipCenterOfMassToUpperLeftBelowDiagonal(self) -> PentagoGameState:
        return PentagoGameState(
            self.turnTracker,
            self.gridState.flipCenterOfMassToUpperLeftBelowDiagonal()
        )

    

    @classmethod
    def fromTensor(self, tensor: tf.Tensor) -> PentagoGameState:
        array: np.ndarray = tensor.numpy()
        return self.fromNumpy(array)

    @classmethod
    def fromNumpy(self, array: np.ndarray) -> PentagoGameState:
        current_player = array[0]
        turn_step = array[1]
        grid_0 = array[2:38].reshape((6,6))
        grid_1 = array[38:].reshape((6,6))
        total_moves = (np.sum(grid_0) + np.sum(grid_1)) * 2 - turn_step
        if turn_step == 0:
            last_player_to_move = (current_player + 1) % 2
        else:
            last_player_to_move = current_player
        if total_moves == 0:
            last_player_to_move = None
        return PentagoGameState(
            TurnTracker(2,2,current_player,turn_step,total_moves,last_player_to_move),
            TwoPlayerGridState(grid_0, grid_1)
        )

    @classmethod
    def fairVariant(self) -> PentagoGameState:
        """Under ideal play, pentago is a first player win. In order to make the
        game fairer (draw under ideal play), we can force the first player to play a
        specific move. This is a more interesting variant of the game.

        See: https://perfect-pentago.net/
        """
        return PentagoGameState().place((0,0)).skipRotation()