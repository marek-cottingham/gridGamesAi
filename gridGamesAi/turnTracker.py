from __future__ import annotations
from dataclasses import dataclass

import tensorflow as tf


@dataclass
class TurnTracker:
    number_players: int
    number_turn_steps: int
    current_player: int = 0
    current_turn_step: int = 0
    total_moves: int = 0
    last_player_to_move: int | None = None

    @property
    def _next_move_player(self):
        """Player who will make the move after the current move"""

        is_last_turn_step = self.current_turn_step == self.number_turn_steps - 1

        if is_last_turn_step:
            return (self.current_player + 1) % self.number_players
        else:
            return self.current_player

    @property
    def _next_move_turn_step(self):
        """Turn step after the current move"""
        return (self.current_turn_step + 1) % self.number_turn_steps
    
    def getIncremented(self):
        """Returns a TurnTracker which is incremented to the next move"""
        return TurnTracker(
            self.number_players,
            self.number_turn_steps,
            self._next_move_player,
            self._next_move_turn_step,
            self.total_moves + 1,
            self.current_player
        )
    
    @property
    def other_to_last_player_to_move(self):
        """Returns the player who was not the last player to move in a two player game"""
        if self.number_players != 2:
            raise Exception(f"Not available for game, self.number_players = {self.number_players} != 2")
        return (self.last_player_to_move + 1) % self.number_players

    def asTensor(self):
        return tf.constant([self.current_player, self.current_turn_step], dtype=tf.int32)