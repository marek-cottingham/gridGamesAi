import unittest

from gridGamesAi.go_interface import goMinimaxMove
from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.pentago.scoringAgent import PentagoNaiveScoringAgent

class goInterfaceTestCase(unittest.TestCase):
    def test_c_interface(self):
        score = PentagoNaiveScoringAgent().score
        g = PentagoGameState.fairVariant()
        g1 = goMinimaxMove(g, 1, score)
        g2 = g.place((0, 1))
        if (g1 != g2):
            print("Not equal")
            print(g1)
            print(g2)
            self.assertTrue(False)