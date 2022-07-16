import unittest
import tensorflow as tf

from gridGamesAi.temporal_difference_model import TD_model

class TDmodelTestCase(unittest.TestCase):
    def test_temporal_difference(self):
        t = TD_model()
        res = t.weight_gradients_by_temporal_difference([0.2, 0.1, -0.05],tf.constant([1.0,1.0,1.0]))
        self.assertAlmostEqual(
            res,
            ((1 + 0.7 + 0.7**2) * 0.2 + (1 + 0.7) * 0.1 + (1) * -0.05)
        )

    