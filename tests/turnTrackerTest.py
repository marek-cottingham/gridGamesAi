import unittest

from gridGamesAi.turnTracker import TurnTracker

class TurnTrackerTestCase(unittest.TestCase):
    def test_increment_single_step_game(self):
        tt = TurnTracker(2, 1)
        tt2 = tt.getIncremented()
        self.assertEqual(tt.current_player, 0)
        self.assertEqual(tt2.current_player, 1)
        self.assertEqual(tt2.current_turn_step, 0)
        self.assertEqual(tt2.total_moves, 1)

    def test_increment_three_step_game(self):
        tt = TurnTracker(2, 3)
        tt2 = tt.getIncremented()
        tt3 = tt2.getIncremented()
        tt4 = tt3.getIncremented()
        self.assertEqual(tt3.current_player, 0)
        self.assertEqual(tt3.current_turn_step, 2)
        self.assertEqual(tt4.current_player, 1)
        self.assertEqual(tt4.current_turn_step, 0)