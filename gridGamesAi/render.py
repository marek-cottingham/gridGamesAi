from __future__ import annotations

import matplotlib
import matplotlib.backend_bases
import matplotlib.pyplot as plt
import numpy as np

from .agents import AbstractAgent
from .common import AbstractGridGameState


class GridRender():
    def __init__(self):

        if not matplotlib.is_interactive():
            raise Exception("Matplotlib must be in interactive mode")

        self.fig = plt.figure()
        self.ax: plt.Axes = plt.subplot(1,1,1)
        self.ax_image = plt.imshow(np.zeros((1,1)),vmin=-1, vmax=1, cmap=plt.get_cmap('bwr'))
        
        # Configure the plot
        self.ax.tick_params(which='both', length=0)
        self.ax.grid(True)

    def render(self, gameState: AbstractGridGameState):
        """Render the current grid game state of a game using matplotlib"""
        combined_grid = self.combineGrids(gameState)
        
        self.ax_image.set_data(combined_grid)
        self._render_axis_ticks(combined_grid)
        self.render_title(gameState)
        self.fig.canvas.flush_events()

    def _render_axis_ticks(self, combined_grid: np.ndarray):
        """Sets plot axis ticks so as to create gridlines around each space on the game board"""
        axis_ticks_x = np.arange(combined_grid.shape[0]+1) - 0.5
        axis_ticks_y = np.arange(combined_grid.shape[1]+1) - 0.5
        self.ax.set_xticks(axis_ticks_x)
        self.ax.set_yticks(axis_ticks_y)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        self.ax_image.set_extent((-0.5, combined_grid.shape[0]-0.5, -0.5, combined_grid.shape[0]-0.5))

    def render_title(self, gameState: AbstractGridGameState):
        if gameState.isEnd:
            self._render_ended_game_title(gameState)
        if not gameState.isEnd:
            self.ax.set_title(f"Player {gameState.current_player}'s turn to move.")
            
    def render_waiting_input_title(self, gameState: AbstractGridGameState):
        if gameState.isEnd:
            self._render_ended_game_title(gameState)
        if not gameState.isEnd:
            self.ax.set_title(f"Your turn to move (player {gameState.current_player}).")

    def _render_ended_game_title(self, gameState: AbstractGridGameState):
        if gameState.isWin:
            self.ax.set_title(f"Player {gameState.winPlayer} won!")
        if gameState.isDraw:
            self.ax.set_title("The game ended in a draw.")

        # self.ax.set_xlabel(f"Current position score: {game.currentGameState.score():.3f}")

    def combineGrids(self, gameState: AbstractGridGameState) -> np.ndarray:
        """Get the game grid a single array, where cells with value 1 represent player 0's pieces
        and cells with value -1 reperesent players 1's pieces"""

        return gameState.grid_0 - gameState.grid_1

def _nearest_integer(x: float):
    return int(np.rint(x))

class UserGridAgent(AbstractAgent):

    def __init__(self, rend: GridRender | None = None):
        if rend is None:
            rend = GridRender()
        self.rend: GridRender = rend

        self.inputEnabled: bool = False
        self.inputGameState: AbstractGridGameState | None = None
        self.nextGameState: AbstractGridGameState | None = None

        # Bind the mouse click event handler
        self.rend.fig.canvas.mpl_connect("button_press_event", lambda x: self.onclick(x))

    def move(self, gameState: AbstractGridGameState) -> AbstractGridGameState:
        self._move_setup(gameState)
        while (self.nextGameState is None):
            plt.waitforbuttonpress()
        self._move_cleanup()
        return self.nextGameState

    def _move_setup(self, gameState):
        self.inputEnabled = True
        self.inputGameState = gameState
        self.nextGameState = None
        self.rend.render(gameState)
        self.rend.render_waiting_input_title(gameState)

    def _move_cleanup(self):
        self.inputEnabled = False

    def onclick(self, event: matplotlib.backend_bases.MouseEvent):
        if self.inputEnabled and event.inaxes == self.rend.ax:
            x, y = self._get_index_from_click_event(event)
            self._try_move(x, y)

    def _try_move(self, x, y):
        self._try_place_move(x, y)

    def _try_place_move(self, x, y):
        try:
            self.nextGameState = self.inputGameState.place((x,y))
        except ValueError:
            self.rend.ax.set_title(f"Placement on position {x}, {y} is not a valid move")

    def _get_index_from_click_event(self, event):
        x_graph = _nearest_integer(event.xdata)
        y_graph = _nearest_integer(event.ydata)
        x, y = self._convertToGameCoords(x_graph, y_graph)
        return x,y

    def _convertToGameCoords(self, x_graph, y_graph):
        x = self.inputGameState.grid_0.shape[0] - y_graph - 1
        y = x_graph
        return x,y
        
