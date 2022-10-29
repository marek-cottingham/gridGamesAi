from __future__ import annotations

from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

from ..common import AbstractRotationGridGameState

from ..render import GridRender, UserGridAgent

def _patchArrowArc(x, y, theta1, theta2):
    return patches.Arc(
        (x,y), 0.6, 0.6,
        zorder = 50,
        theta1 = theta1,
        theta2 = theta2,
        linewidth = 2
    )

def clockwiseRotateArrow(x,y):
    theta1 = 0
    theta2 = 270
    return [
        # Patch for arrow tip
        patches.RegularPolygon(
            (x+0.3, y),
            3, 0.15,
            orientation = np.pi,
            color='k' 
        ),
        _patchArrowArc(x,y,theta1,theta2)
    ]

def antiClockwiseRotateArrow(x,y):
    theta1 = 180
    theta2 = 90
    return [
        # Patch for arrow tip
        patches.RegularPolygon(
            (x, y+0.3),
            3, 0.15,
            orientation = np.pi/2,
            color='k' 
        ),
        _patchArrowArc(x,y, theta1, theta2)
    ]

def rotateArrow(x,y,rotationKey):
    isClockwise = not 'a' in rotationKey
    if isClockwise:
        return clockwiseRotateArrow(x,y)
    if not isClockwise:
        return antiClockwiseRotateArrow(x,y)

class NgoRender(GridRender):

    # Rotate icons locations during turn_step = 1 for gridSize = 6
    # _____ tl_c  _ _ tr_ac ____
    # tl_ac _____ _ _ _____ tr_c
    # ...
    # ...
    # bl_c  _____ _ _ _____ br_ac
    # _____ bl_ac _ _ br_c  _____

    # Clicking on these locations should perform the corresponding rotation
    # operation, as given by the key lookup table
    
    def __init__(self, gridSize=6):
        super().__init__()

        first_index_ = 0
        second_index = 1
        second_last_index = gridSize - 2
        last_index_______ = gridSize - 1
        self.keyLookup = {
                (first_index_, second_index): 'tl_c',
                (second_index, first_index_): 'tl_ac',
                (second_last_index, first_index_): 'bl_c',
                (first_index_, second_last_index): 'tr_ac',
                (last_index_______, second_index): 'bl_ac',
                (second_index, last_index_______): 'tr_c',
                (last_index_______, second_last_index): 'br_c',
                (second_last_index, last_index_______): 'br_ac',
            }

         # Configure the plot
        self._generateRotateArrowPatches()
        self.ax.axhline((gridSize-1)/2, color='k')
        self.ax.axvline((gridSize-1)/2, color='k')

    def _generateRotateArrowPatches(self):
        self.rotateArrowsPatches = []
        for (x,y), rotationKey in self.keyLookup.items():
            self.rotateArrowsPatches += rotateArrow(x, y, rotationKey)
        for patch in self.rotateArrowsPatches:
            self.ax.add_patch(patch)
        self.setRotateArrowsVisibility(False)

    def setRotateArrowsVisibility(self, isVisible: bool):
        for patch in self.rotateArrowsPatches:
            patch.set_visible(isVisible)

class UserNgoAgent(UserGridAgent):

    def __init__(self, rend: NgoRender | None = None):
        if rend is None:
            rend = NgoRender()
        super().__init__(rend)
        self.rend: NgoRender
        self.inputGameState: AbstractRotationGridGameState

    def _move_setup(self, gameState):
        super()._move_setup(gameState)
        if self.inputGameState.turn_step == 1:
            self.rend.setRotateArrowsVisibility(True)

    def _move_cleanup(self):
        super()._move_cleanup()
        if self.inputGameState.turn_step == 1:
            self.rend.setRotateArrowsVisibility(False)

    def _try_move(self, x, y):
        if self.inputGameState.turn_step == 0:
            self._try_place_move(x, y)
        if self.inputGameState.turn_step == 1:
            self._try_rotate_move(x, y)

    def _try_rotate_move(self, x, y):
        try:
            rotate_key = self.rend.keyLookup[(x,y)]
            self.nextGameState = self.inputGameState.rotate(rotate_key)
        except KeyError:
            pass
