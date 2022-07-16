from __future__ import annotations
from dataclasses import dataclass

from functools import cached_property
from typing import List, Tuple
import numpy as np

from ..turnTracker import TurnTracker
from ..twoPlayerGridState import TwoPlayerGridState
from ..common import AbstractGridGameState

def _gridContainsWin(grid: np.ndarray):
    """Evaluates whether a player grid contains a winning set of pieces, ie.
    3 on a row, column or diagonal

    param: grid
        shape (3,3) ndarray representing pieces placed by the player
    """
    max_horizontal_line_length = np.max(np.sum(grid, axis=0))
    max_vertical_line_length = np.max(np.sum(grid, axis=1))
    diag_1_length = grid[0,0] + grid[1,1] + grid[2,2]
    diag_2_length = grid[0,2] + grid[1,1] + grid[2,0]
    return (
        max_horizontal_line_length == 3 or 
        max_vertical_line_length == 3 or 
        diag_1_length == 3 or
        diag_2_length == 3
    )

@dataclass
class TicTacToeGameState(AbstractGridGameState):
    turnTracker: TurnTracker = TurnTracker(2,1)
    gridState: TwoPlayerGridState = TwoPlayerGridState(
        np.zeros((3,3)), np.zeros((3,3))
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

    @cached_property
    def _win_player_0(self) -> bool:
        return _gridContainsWin(self.grid_0)

    @cached_property
    def _win_player_1(self) -> bool:
        return _gridContainsWin(self.grid_1)

    @property
    def isWin(self) -> bool:
        return self._win_player_0 or self._win_player_1

    @property
    def winPlayer(self) -> bool:
        if self._win_player_0:
            return 0
        if self._win_player_1:
            return 1
        return None

    @property
    def isDraw(self) -> bool:
        return np.sum(self.gridState.combined_grid) == 9 and not self.isWin

    @property
    def isEnd(self) -> bool:
        return self.isDraw or self.isWin

    @property
    def next_moves(self) -> List[TicTacToeGameState]:
        next_grid_states = self.gridState.nextValidPlacements(self.current_player)
        next_turn_tracker = self.turnTracker.getIncremented()
        return [TicTacToeGameState(next_turn_tracker, state) for state in next_grid_states]

    def placeMove(self, index: Tuple[int, int]) -> TicTacToeGameState:
        self.gridState.assert_unoccupied(index)
        return TicTacToeGameState(
            self.turnTracker.getIncremented(),
            self.gridState.placeOn(self.current_player, index)
        )