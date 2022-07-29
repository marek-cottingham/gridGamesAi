
from typing import List, Tuple
import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing as np_test

from gridGamesAi.pentago.gameState import _gridOccupancy, PentagoGameState
from gridGamesAi.pentago.scoringAgent import PentagoNaiveScoringAgent
from tests.patchInspector import PatchInspector

def prep_game( 
        placeIndexes: List[Tuple[int, int]], 
        endAtTurnStep0: bool = True
    ) -> PentagoGameState:
        gs = PentagoGameState()
        for index in placeIndexes[:-1]:
            gs = gs.place(index)
            gs = gs.skipRotation()
        gs = gs.place(placeIndexes[-1])
        if endAtTurnStep0:
            gs = gs.skipRotation()
        return gs

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
        gs = gs.place((2,2))
        self.assertEqual(gs.current_player, 0)
        self.assertEqual(gs.turn_step, 1)
        self.assertEqual(gs.grid_0[2][2], 1)

        gs = gs.rotate('tl_c')
        self.assertEqual(gs.current_player, 1)
        self.assertEqual(gs.grid_0[2][0], 1)
        self.assertEqual(gs.grid_0[2][2], 0)
        self.assertFalse(gs.isWin)
        self.assertFalse(gs.isDraw)

    def test_isValidKey(self):
        gs = prep_game([(2,2), (3,1), (3,2), (5,2)])
        gs2 = prep_game([(2,2), (3,1), (3,2), (5,2)])
        gs3 = prep_game([(2,2), (3,1), (3,3), (5,2)])
        gs4 = prep_game([(2,2), (3,1), (3,3), (5,2)], False)
        self.assertEqual(gs, gs2)
        self.assertEqual(hash(gs), hash(gs2))
        self.assertNotEqual(gs, gs3)
        self.assertNotEqual(hash(gs), hash(gs3))
        self.assertNotEqual(gs3, gs4)
        self.assertNotEqual(hash(gs3), hash(gs4))

        d = {gs: 1}
        self.assertEqual(d[gs], 1)
        self.assertEqual(d[gs2], 1)
        self.assertRaises(KeyError, d.__getitem__, gs3)
        
    def test_tensor_conversion(self):
        for gs in [
            prep_game([(2,2), (3,1), (3,2), (5,2)]),
            prep_game([(2,2), (3,1), (3,2), (5,2)], False),
            prep_game([(2,2), (3,1), (3,2), (5,2), (4,3), (4,4), (5,1), (5,3)]),
            PentagoGameState(),
        ]:
            gs: PentagoGameState = gs
            tensor = gs.asTensor()
            self.assertEqual(tensor[0], gs.current_player)
            self.assertEqual(tensor[1], gs.turn_step)

            gs2 = PentagoGameState.fromTensor(tensor)
            self.assertEqual(gs, gs2)
            self.assertEqual(gs.turnTracker.total_moves, gs2.turnTracker.total_moves)

    def test_CenterMassFlip(self):
        gs = prep_game([(2,2), (3,1), (3,2), (5,2)])
        
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
        gs = prep_game([(2,2)])
        gs.gridState.assert_unoccupied((1,2))
        self.assertRaises(ValueError, gs.gridState.assert_unoccupied, (2,2))

class PentagoNaiveScoreTestCase(unittest.TestCase):
    def patchInspector_gridOccupancy(self):
        return PatchInspector(
                patch('gridGamesAi.pentago.scoringAgent._gridOccupancy'),
                _gridOccupancy
        )

    def test_returnsCorrectScore(self):
        agent = PentagoNaiveScoringAgent()

        gs = PentagoGameState()
        gs = gs.place((2,2))
        score = agent.unsquished_score(gs)
        self.assertAlmostEqual(score, 7)

        gs = gs.skipRotation()
        gs = gs.place((0,0))
        score = agent.unsquished_score(gs)
        self.assertAlmostEqual(score, 4)

    def test_employsCaching(self):
        agent = PentagoNaiveScoringAgent()
        
        with self.patchInspector_gridOccupancy() as mock_gridOccupancy:
            
            score = agent.unsquished_score(prep_game([(2,2)]))
            score = agent.unsquished_score(prep_game([(2,2), (0,0)], False))
            score = agent.unsquished_score(prep_game([(2,2), (0,0)], False))

            # _gridOccupancy is called twice by unsquished_score.
            # If 6 calls are made, then caching is not functioning.
            mock_gridOccupancy.assertCalledTimes(4)

    def test_cacheCanBeReset(self):
        agent = PentagoNaiveScoringAgent()

        self.assertTrue(agent.hasCachingFacility)

        with self.patchInspector_gridOccupancy() as mock_gridOccupancy:

            score = agent.unsquished_score(prep_game([(2,2)]))
            agent.resetCache()
            score = agent.unsquished_score(prep_game([(2,2)]))

            mock_gridOccupancy.assertCalledTimes(4)
