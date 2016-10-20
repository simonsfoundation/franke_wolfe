
"""
Vector-trace mixin heuristic for Franke Wolfe optimization.

At each iteration with estimate x_0 objective y = f(x) and gradient v = g(x)
the standard algorithm finds a vertex v within the feasible polygon
which minimizes g * v and optimizes f along along the line segment from x_0 to v.

This mixin instead finds the intersection point p of the line x + l * vars
with the feasible polygon where g * p is minimum (ie, positive) and optimizes
f along the line segment from x_0 to p.

This heuristic may work well when the search space is expected to have
a "large" volume.  If the search space is expected to have a "small" volume
this heuristic may just do extra work.
"""

import numpy as np
from scipy.optimize import linprog
from . import fw_optimize

class VectorMixin(object):

    def optimize_linear_approximation(self, gradient_vector, estimate=None, epsilon=1e-12):
        M = np.max(np.abs(gradient_vector))
        if estimate is None or M < epsilon:
            # default to find vertex if gradient is near zero or estimate point is missing.
            if self.verbose:
                print("fallback to standard linear approximation.")
            return super(VectorMixin, self).optimize_linear_approximation(gradient_vector)
        # otherwise add equality constraints restricting feasible points to line
        # at the estimate point in the direction of the gradient using an additional dimension (lambda).
        dimension = self.dimensions
        assert gradient_vector.shape == (dimension,)
        assert estimate.shape == (dimension,)
        ident = np.identity(dimension)
        vt = gradient_vector.reshape((dimension, 1))
        # Positive section of line should be direction of decrease (negative gradient).
        #  line_point + lambda * gradient = estimate
        # or rewritten:
        #  line_point = estimate - lambda * gradient_vector
        # as lambda increases line_point moves in the direction of decrease.
        addl_A_eq = np.hstack([ident, vt])
        A_eq = self.A_eq
        b_eq = self.b_eq
        if A_eq is None:
            assert b_eq is None
            A_eq_augmented = addl_A_eq
            b_eq_augmented = estimate
        else:
            assert b_eq is not None
            (n_equalities, c) = A_eq.shape
            assert c == dimension
            assert b_eq.shape == (n_equalities,)
            A_eq_with_zeros = np.hstack([A_eq, np.zeros((n_equalities,1))])
            A_eq_augmented = np.vstack([A_eq_augmented, addl_A_eq])
            b_eq_augmented = np.hstack([b_eq, estimate])
        # Add additional dimension to inequality constraints too.
        A_ub = A_ub_with_zeros = self.A_ub
        b_ub = self.b_ub
        if A_ub is not None:
            assert b_ub is not None
            (n_inequalities, c) = A_ub.shape
            assert c == dimension
            A_ub_with_zeros = np.hstack([A_ub, np.zeros((n_inequalities, 1))])
        bounds = self.bounds
        if bounds:
            bounds = list(bounds) + [(None, None)]
        augmented_gradient = np.hstack([gradient_vector, [0.0]])
        if self.verbose:
            print("vector optimizing: " + repr(augmented_gradient))
            print("bounds " + repr(bounds))
        result = linprog(
            augmented_gradient,
            A_ub=A_ub_with_zeros,
            b_ub=b_ub,
            A_eq=A_eq_augmented,
            b_eq=b_eq_augmented,
            bounds=bounds,
            **self.other_linprog_options)
        augmented_vertex = result.x
        # eliminate additional dimension from optimal point.
        vertex = augmented_vertex[:-1]
        if np.allclose(vertex, estimate):
            # default to find vertex when restriction to line segment makes no progress.
            if self.verbose:
                print("no progress in vector optimization: fallback to standard.")
            return super(VectorMixin, self).optimize_linear_approximation(gradient_vector)
        # for consistency
        result.x = vertex
        result.augmented_x = augmented_vertex
        return (result, vertex)

class FrankeWolfeVectorized(VectorMixin, fw_optimize.FrankWolfe):

    pass
