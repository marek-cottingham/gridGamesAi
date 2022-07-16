from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .agents import AbstractAgent
from .common import AbstractGameState


@dataclass
class Game:
    agents: List[AbstractAgent]
    _current_game_state: AbstractGameState
    onStateChange: Callable[[AbstractGameState], None] | None = None

    @property
    def current_game_state(self):
        return self._current_game_state

    @current_game_state.setter
    def current_game_state(self, new_val):
        self._current_game_state = new_val
        if self.onStateChange is not None:
            self.onStateChange(self._current_game_state)

    def moveWithAgent(self, agent: AbstractAgent):
        self.current_game_state = agent.move(self.current_game_state)

    def moveWithCurrentPlayer(self):
        self.moveWithAgent(self.agents[self.current_game_state.current_player])
