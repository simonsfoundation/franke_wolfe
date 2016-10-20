
import unittest
from franke_wolfe import fw_optimize
import numpy as np
import numpy.testing

def objective_2d(a):
    [x, y] = a
    return (x - 1) ** 2 + (y - 1) ** 2

bounds_table = [
    #{"bounds": [(0, 1), (0, 1)], "expect": (1, 1)},
    {"bounds": [(0, 1), (0, 4)], "expect": (1, 2)},
]

class Test2d(unittest.TestCase):

    def test_bounds(self, verbose=True):
        for D in bounds_table:
            bounds = D["bounds"]
            expect = D["expect"]
            options = {}
            #if verbose:
            #    options={"disp": True}
            lm = 10
            runner = fw_optimize.FrankWolfe(objective_2d, bounds=bounds, options=options, iteration_limit=lm)
            runner.verbose = verbose
            result = runner.run()
            if verbose or 1:
                print(result)
            self.assertTrue(result.success)
            x = result.x
            self.assertEqual((2,), x.shape)
            numpy.testing.assert_almost_equal(expect, x)

