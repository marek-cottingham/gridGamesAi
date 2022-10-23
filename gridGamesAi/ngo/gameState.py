from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import hashlib
import itertools

from ..turnTracker import TurnTracker
from ..common import AbstractGridGameState
from ..agents import RandomAgent

randAgent = RandomAgent()

class NgoGameRunner():
    keyTranslation = {
        'tl_c':  '00cw',
        'tl_ac': '00cc',
        'bl_c':  '10cw',
        'bl_ac': '10cc',
        'tr_c':  '01cw',
        'tr_ac': '01cc',
        'br_c':  '11cw',
        'br_ac': '11cc',
    }

    def __init__(self, size_quadrant: int, win_line_length: int, rotation_enabled: bool):
        self.size_quadrant = size_quadrant
        self.size_board = 2* self.size_quadrant
        self.win_line_length = win_line_length
        self.rotation_enabled = rotation_enabled
        self.rotations = self.generateRotationTensors()
        self.win = self.generateWinEvaluationTensor()

    def generateRotationTensors(self):
        identityMatrix = tf.eye(self.size_board,self.size_board,dtype=tf.int32)
        no_rotation_tensor = tf.transpose( tf.tensordot(identityMatrix, identityMatrix, 0), [0,2,1,3])
        quadrantIdentityMatrix = tf.eye(self.size_quadrant,self.size_quadrant,dtype=tf.int32)
        rot_sub = tf.transpose( tf.tensordot(quadrantIdentityMatrix, quadrantIdentityMatrix ,0), [0,2,1,3])
        rot_cw = tf.reverse(tf.transpose(rot_sub, [1,0,2,3]), [0])
        rot_cc = tf.reverse(tf.transpose(rot_sub, [1,0,2,3]), [1])
        rotations = {"none": no_rotation_tensor}
        s = [slice(0,self.size_quadrant),slice(self.size_quadrant,2*self.size_quadrant)]
        i = [0,1]
        for perm in itertools.product(i, repeat=2):
            sectorName = f"{perm[0]}{perm[1]}"
            sliceArr = [s[perm[0]],s[perm[1]],s[perm[0]],s[perm[1]]]
            x = tf.Variable(no_rotation_tensor)
            x = x.__getitem__(sliceArr).assign(rot_cw)
            rotations[f"{sectorName}cw"] = x
            x = tf.Variable(no_rotation_tensor)
            x = x.__getitem__(sliceArr).assign(rot_cc)
            rotations[f"{sectorName}cc"] = x
        return rotations

    def generateWinEvaluationTensor(self):
        win_mapping_tensors = []
        d = self.size_board
        win_n = self.win_line_length
        for i, j in itertools.product(range(d),range(d-win_n+1)):
            spaces_map = tf.Variable(tf.zeros((d,d),dtype=tf.int32))
            spaces_map = spaces_map[i, j:win_n+j].assign(tf.ones((win_n),dtype=tf.int32))
            win_mapping_tensors.append(spaces_map)
            spaces_map = tf.Variable(tf.zeros((d,d),dtype=tf.int32))
            spaces_map = spaces_map[j:win_n+j, i].assign(tf.ones((win_n),dtype=tf.int32))
            win_mapping_tensors.append(spaces_map)
        for i, j in itertools.product(range(d-win_n+1),repeat=2):
            spaces_map = tf.Variable(tf.zeros((d,d),dtype=tf.int32))
            spaces_map = spaces_map[i:win_n+i, j:win_n+j].assign(tf.eye(win_n,dtype=tf.int32))
            win_mapping_tensors.append(spaces_map)
            spaces_map = tf.Variable(tf.zeros((d,d),dtype=tf.int32))
            spaces_map = spaces_map[i:win_n+i, j:win_n+j].assign(tf.reverse(tf.eye(win_n,dtype=tf.int32),[0]))
            win_mapping_tensors.append(spaces_map)
        win = tf.stack(win_mapping_tensors, axis=2)
        return win

    def rotate(self, grid: tf.Tensor, rotationKey: str) -> tf.Tensor:
        try:
            rotationKey = self.keyTranslation[rotationKey]
        except KeyError:
            pass
        rot = self.rotations[rotationKey]
        return tf.Variable( tf.tensordot(grid, rot, (2)) )

    def hasWinningLine(self, grid: tf.Tensor):
        return tf.reduce_max( tf.tensordot(grid, self.win, 2), [1] )

    def initialiseTurnTracker(self):
        if self.rotation_enabled:
            return TurnTracker(2, 2)
        else:
            return TurnTracker(2, 1)

    def initialiseGrid(self):
        return tf.Variable(tf.zeros([2,self.size_board,self.size_board]))

    def place(self, grid: tf.Tensor, player: int, index: Tuple[int,int]):
        newGrid = tf.Variable(grid)
        tf.tensor_scatter_nd_add(newGrid, [[player, index[0], index[1]]], [1])
        if tf.reduce_max(newGrid) == 2:
            raise Exception("Invalid placement")
        return newGrid

    def nextValidPlacement(self):
        # look into using tf.equal here: https://stackoverflow.com/questions/39219414/in-tensorflow-how-can-i-get-nonzero-values-and-their-indices-from-a-tensor-with
        return NotImplemented

class NgoGameState(AbstractGridGameState):
    def __init__(self, turnTracker: TurnTracker, grid: tf.Tensor, gameRunner: NgoGameRunner):
        self.gameRunner = gameRunner

        if turnTracker is None:
            turnTracker = gameRunner.initialiseTurnTracker()
        self.turnTracker = turnTracker

        if grid is None:
            grid = gameRunner.initialiseGrid()
        self.grid = grid

    @property
    def current_player(self) -> int: return self.turnTracker.current_player
    
    @property
    def turn_step(self) -> int: return self.turnTracker.current_turn_step
    
    @property
    def last_player_to_move(self) -> int: return self.turnTracker.last_player_to_move

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

    def rotate(self, rotate_key: str) -> NgoGameState:
        return NgoGameState(
            self.turnTracker.getIncremented(),
            self.gameRunner.rotate(self.grid, rotate_key),
            self.gameRunner
        )

    def place(self, index: Tuple[int, int]) -> NgoGameState:
        return NgoGameState(
            self.turnTracker.getIncremented(),
            self.gameRunner.place(self.grid, self.current_player, index),
            self.gameRunner
        )

    def skipRotation(self) -> NgoGameState:
        if self.turn_step != 1:
            raise ValueError("Can only skip rotation on second part of turn")
        return self.skipMove()

    def skipMove(self) -> NgoGameState:
        return NgoGameState(
            self.turnTracker.getIncremented(),
            self.grid,
            self.gameRunner
        )

    @cached_property
    def next_moves(self) -> List[NgoGameState]:
        next_grid_states = self._getNextGridStates()
        next_turn_tracker = self.turnTracker.getIncremented()
        return [NgoGameState(next_turn_tracker, state) for state in next_grid_states]

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

    def __hash__(self) -> int:
        arr = self.asNumpy().view()
        hash = hashlib.sha1(arr).hexdigest()
        return int(hash,16)

    def __eq__(self, other: object) -> bool:
        return np.all(self.asNumpy() == other.asNumpy())

    def flipCenterOfMassToUpperLeftBelowDiagonal(self) -> NgoGameState:
        return NgoGameState(
            self.turnTracker,
            self.gridState.flipCenterOfMassToUpperLeftBelowDiagonal()
        )

    

    @classmethod
    def fairVariant(self) -> NgoGameState:
        """Under ideal play, pentago is a first player win. In order to make the
        game fairer (draw under ideal play), we can force the first player to play a
        specific move. This is a more interesting variant of the game.

        See: https://perfect-pentago.net/
        """
        return NgoGameState.place((0,0)).skipRotation()

    @classmethod
    def init_with_n_random_moves(self, n):
        gs = NgoGameState.fairVariant()
        for _ in range(n):
            gs: NgoGameState = randAgent.move(gs)
            gs = gs.skipMove()
        return gs