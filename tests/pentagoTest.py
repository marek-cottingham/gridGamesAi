
import unittest

import numpy as np
import numpy.testing as np_test
import scipy

from gridGamesAi.pentago.gameState import _gridOccupancy, PentagoGameState
from gridGamesAi.pentago.scoringAgent import PentagoNaiveScoringAgent

class pentagoGameStateTestCase(unittest.TestCase):
    def test_finding_grid_occupacy(self):
        grid = np.zeros((6,6))
        grid[0][0] = 1
        grid[1][1] = 1
        grid[1][2] = 1
        occupacy = _gridOccupancy(grid)
        expected_occupacy = [
            1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, #horzontal lines
            1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, #vertical lines
            0, 1, 2, 1, 0, 0, 0, 0 #diagonals
        ]
        np_test.assert_allclose(occupacy, expected_occupacy)

    def test_moves(self):
        gs = PentagoGameState()
        gs = gs.placeMove((2,2))
        self.assertEqual(gs.current_player, 0)
        self.assertEqual(gs.turn_step, 1)
        self.assertEqual(gs.grid_0[2][2], 1)

        gs = gs.rotateMove('tl_c')
        self.assertEqual(gs.current_player, 1)
        self.assertEqual(gs.grid_0[2][0], 1)
        self.assertEqual(gs.grid_0[2][2], 0)
        self.assertFalse(gs.isWin)
        self.assertFalse(gs.isDraw)

    def test_tensor_conversion(self):
        gs = PentagoGameState()
        gs = gs.placeMove((2,2))
        gs = gs.rotateMove('tl_c')

        tensor = gs.asTensor()
        self.assertEqual(tensor[0], 1)
        self.assertEqual(tensor[1], 0)

        gs = PentagoGameState.fromTensor(tensor)
        self.assertEqual(gs.current_player, 1)
        self.assertEqual(gs.grid_0[2][0], 1)
        self.assertEqual(gs.grid_0[2][2], 0)
        self.assertFalse(gs.isWin)
        self.assertFalse(gs.isDraw)

    def test_CenterMassFlip(self):
        gs = PentagoGameState()
        for index in [(2,2), (3,1), (3,2), (5,2)]:
            gs = gs.placeMove(index)
            # Skip rotation
            gs = gs.skipMove()
        
        flipedState = gs.flipCenterOfMassToUpperLeftBelowDiagonal()

        np_test.assert_allclose(
            flipedState.grid_0, 
            np.transpose(np.flipud(gs.grid_0))
        )
        np_test.assert_allclose(
            flipedState.grid_1, 
            np.transpose(np.flipud(gs.grid_1))
        )

    def test_ValidPlacementCheck(self):
        gs = PentagoGameState()
        gs = gs.placeMove((2,2))
        gs.gridState.assert_unoccupied((1,2))
        self.assertRaises(ValueError, gs.gridState.assert_unoccupied, (2,2))

class PentagoNaiveScoreTestCase(unittest.TestCase):
    def test_NaiveScore_1(self):
        gs = PentagoGameState()
        gs = gs.placeMove((2,2))
        score = PentagoNaiveScoringAgent().unsquished_score(gs)
        self.assertAlmostEqual(score, 7)

        gs = gs.skipMove()
        gs = gs.placeMove((0,0))
        score = PentagoNaiveScoringAgent().unsquished_score(gs)
        self.assertAlmostEqual(score, 4)