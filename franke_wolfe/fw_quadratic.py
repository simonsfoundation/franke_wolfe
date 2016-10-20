"""

"""

# next: convergence estimate using border point rather than vertex.

import numpy as np
import fw_optimize

class FWQuadratic(fw_optimize.FrankWolfe):

    def __init__(self,
                C,
                d,
                A_ub=None, b_ub=None,
                A_eq=None, b_eq=None,
                bounds=None,
                tol=1.0E-12,
                gradient_delta=1.0E-10,
                iteration_limit=1000,
                **other_linprog_options
                ):
        """
        Run container to solve the following quadratic programming problem:

        minimize: x.T * C * x + d.T * x

        subject to:  A_ub * x <= b_ub 
        and A_eq * x == b_eq

        The parameters A_ub, b_ub, A_eq, b_eq,
        bounds and tol are as defined for scipy.optimize.linprog.

        Parameters
        ----------
        C: n by n real valued array.
        d: n vector of reals.
        A_ub : array_like
            2-D array which, when matrix-multiplied by x, gives the values of the
            upper-bound inequality constraints at x
            (passed to scipy.optimize.linprog). Optional.
        b_ub : array_like of None
            1-D array of values representing the upper-bound of each inequality
            constraint (row) in A_ub. Passed to scipy.optimize.linprog.
            Optional.
        A_eq : array_like
            2-D array which, when matrix-multiplied by x, gives the values of the
            equality constraints at x.  Passed to scipy.optimize.linprog. Optional.
        b_eq : array_like
            1-D array of values representing the RHS of each equality constraint
            (row) in A_eq.  Passed to scipy.optimize.linprog. Optional.
        bounds : array_like
            The bounds for each independent variable in the solution, which can take
            one of three forms::
            None : The default bounds, all variables are non-negative.
            (lb, ub) : If a 2-element sequence is provided, the same
                    lower bound (lb) and upper bound (ub) will be applied
                    to all variables.
            [(lb_0, ub_0), (lb_1, ub_1), ...] : If an n x 2 sequence is provided,
                    each variable x_i will be bounded by lb[i] and ub[i].
            Infinite bounds are specified using -np.inf (negative)
            or np.inf (positive). Passed to scipy.optimize.linprog. Optional.
        tol : float
            The tolerance which determines when a solution is "close enough".
            Passed to scipy.optimize.linprog. Optional.
        gradient_delta: float
            The offset used for estimating the gradient using finite differences
            if the gradient is not specified. Optional.
        iteration_limit: int
            Stop and return last estimate after this many iterations.
        Other_keyword_arguments: dict
            Any other keyword arguments are passed to scipy.optimize.linprog.

        Based on http://www.math.chalmers.se/Math/Grundutb/CTH/tma946/0203/fw_eng.pdf
        """
        fw_optimize.FrankWolfe.__init__(self,
                    None,
                    None,
                    A_ub, b_ub,
                    A_eq, b_eq,
                    bounds,
                    tol,
                    gradient_delta,
                    iteration_limit,
                    **other_linprog_options)
        C = np.array(C)
        # Make C symmetrical, for convenience.
        C = 0.5 * (C + C.T)
        self.C = C
        self.d = np.array(d)
        dim = self.dimensions
        assert self.C.shape == (dim, dim), "bad quadratic array dimensions." + repr((dim, self.C.shape))
        assert self.d.shape == (dim,), "bad quadratic vector dimensions" + repr((dim, self.d.shape))

    def objective(self, x):
        self.f_count += 1
        xT = x.T
        C = self.C
        dT = self.d.T
        return xT.dot(C).dot(x) + dT.dot(x)

    def gradient(self, x):
        self.g_count += 1
        C = self.C
        d = self.d
        return 2.0 * (C.dot(x)) + d

def squared_distance_C_d(xc, yc):
    "Squared distance to xc yc quadratic parameters. For testing."
    C = np.array([[1, 0], [0, 1]])
    d = d = -2 * np.array([xc, yc])
    return (C, d)

def srt_squared_distance_C_d(xc, yc, yscale, rotate):
    "Squared distance to xc yc quadratic parameters rotated and scaled. For testing."
    not finished
