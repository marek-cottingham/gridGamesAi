import unittest
import tensorflow as tf
import numpy as np

from gridGamesAi.pentago.temporal_difference_model import TD_model

class TDmodelTestCase(unittest.TestCase):
    def test_generate_td_weights(self):
        t = TD_model()
        deltas = np.array([0.2, 0.1, -0.05])
        result = t.generate_temporal_difference_weights(deltas)
        self.assertAlmostEqual(result[0], (1 + 0.7 + 0.7**2) * 0.2)
        self.assertAlmostEqual(result[1], (1 + 0.7) * 0.1)
        self.assertAlmostEqual(result[2], (1) * -0.05)

    def test_weigh_gradients_by_temporal_difference(self):
        t = TD_model()
        weights = t.generate_temporal_difference_weights(np.array([0.2, 0.1, -0.05]))
        result = t.weigh_gradients_by_temporal_difference(weights,[tf.constant(1.0)]*3)
        self.assertAlmostEqual(
            result.numpy(),
            ((1 + 0.7 + 0.7**2) * 0.2 + (1 + 0.7) * 0.1 + (1) * -0.05)
        )

    


    