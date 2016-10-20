
import unittest
from franke_wolfe import fw_optimize
import numpy as np
import numpy.testing

def objective_1d(a):
    x = a[0]
    return (x - 1) ** 2

class Test1d(unittest.TestCase):

    def test_bounds_left(self, bounds=[(-1, 0.5)], expect=[0.5], verbose=False):
        options = {}
        if verbose:
            options={"disp": True}
        runner = fw_optimize.FrankWolfe(objective_1d, bounds=bounds, options=options, iteration_limit=10)
        result = runner.run()
        if verbose:
            print(result)
        self.assertTrue(result.success)
        x = result.x
        self.assertEqual((1,), x.shape)
        numpy.testing.assert_almost_equal(expect, x)

    def test_open_left(self):
        return self.test_bounds_left(bounds=[(None, 0.77)], expect=[0.77])

    def test_right(self):
        return self.test_bounds_left(bounds=[(1.9, 22)], expect=[1.9])

    def test_open_right(self):
        return self.test_bounds_left(bounds=[(1.5, None)], expect=[1.5])

    def test_center(self):
        return self.test_bounds_left(bounds=[(-10, 22)], expect=[1.0])

    def xtest_open_right_center(self):
        return self.test_bounds_left(bounds=[(-1.5, None)], expect=[1.5])

    def test_eq1(self, x0=1, verbose=False):
        A_eq = [[1.0]]
        b_eq = [x0]
        expect = [x0]
        options = {}
        if verbose:
            options={"disp": True}
        runner = fw_optimize.FrankWolfe(objective_1d, A_eq=A_eq, b_eq=b_eq, options=options, iteration_limit=10)
        result = runner.run()
        if verbose:
            print(result)
        self.assertTrue(result.success)
        x = result.x
        self.assertEqual((1,), x.shape)
        numpy.testing.assert_almost_equal(expect, x)
