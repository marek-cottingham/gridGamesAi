from time import time
import unittest

from gridGamesAi.pentago.go_interface import goMinimaxMove, goSelfPlay
from gridGamesAi.paths import PENTAGO_MODELS_DIR
from gridGamesAi.pentago.gameState import PentagoGameState
from gridGamesAi.pentago.scoringAgent import PentagoNaiveScoringAgent
from gridGamesAi.pentago.temporal_difference_model import Pentago_TD_Agent

class goInterfaceTestCase(unittest.TestCase):
    def test_c_interface(self):
        score = PentagoNaiveScoringAgent().score
        g = PentagoGameState.fairVariant()
        g1 = goMinimaxMove(g, 0, score)
        g2 = g.place((2, 2))
        if (g1 != g2):
            print("Not equal")
            print(g1)
            print(g2)
            self.assertTrue(False)

    def test_selfPlay(self):
        scoreAgent = Pentago_TD_Agent(PENTAGO_MODELS_DIR / "test_model_2000")
        score = scoreAgent.score

        n = 10
        start = time()
        for i in range(n):
            g = PentagoGameState.fairVariant()
            moves = goSelfPlay(g, 0, score)
            g1 = moves[-1]
            if not g1.isEnd:
                print("Error")
                print(g1)
                print("Is draw:", g1.isDraw)
                print("Is win:", g1.isWin)
                raise AssertionError

        print("Time per game:", (time() - start)/n, "s")