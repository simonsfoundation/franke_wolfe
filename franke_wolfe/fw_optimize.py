
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import approx_fprime
from scipy.optimize import minimize_scalar
from scipy.optimize.optimize import OptimizeResult

class FrankWolfe(object):

    verbose = False

    blend = False  # use cumulative estimate.
    blend_ratio = 1/1.66  # golden ratio?
    vertex_ratio = 1 - 1/1.66   # average vertices if not None
    scalar_fallback = True

    def __init__(self,
                f,
                g=None,
                A_ub=None, b_ub=None,
                A_eq=None, b_eq=None,
                bounds=None,
                tol=1.0E-12,
                gradient_delta=1.0E-10,
                iteration_limit=1000,
                **other_linprog_options
                ):
        """
        Run container to solve the following problem:

        minimize: f(x)

        subject to:  A_ub * x <= b_ub 
        and A_eq * x == b_eq

        It is assumed that f is a continuously differentiable function
        and that g (if given) is the gradient of f.  If f is not convex the method
        will find a local minimum and if f is not well behaved the method
        may not converge. The parameters A_ub, b_ub, A_eq, b_eq,
        bounds and tol are as defined for scipy.optimize.linprog.

        Parameters
        ----------
        f: callable
            The function to minimize where f(x) is a scalar when x is a vector
            of length n.
        g: callable
            The gradient of f where g(x) is a vector of length n
            if x is a vector of length n.  The gradient will be
            approximated using finite differences if this argument is None.
            Optional.
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
        self.f = f
        if g is not None:
            self.g = g
        else:
            self.g = self.gradient_estimate
        # Validate and determine dimensionality.
        dimensions = None
        if A_ub is not None or b_ub is not None:
            assert A_ub is not None and b_ub is not None, "upper bounds not specified correctly"
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            (r, c) = A_ub.shape
            (r2,) = b_ub.shape
            assert r == r2, "inconsistent upper bound dimensions " + repr((r,c,r2))
            if dimensions is None:
                dimensions = c
            assert dimensions == c, "inconsistent upper bound dimension " + repr((dimensions, c))
        if A_eq is not None or b_eq is not None:
            assert A_eq is not None and b_eq is not None, "upper bounds not specified correctly"
            A_eq = np.array(A_eq)
            b_eq = np.array(b_eq)
            (r, c) = A_eq.shape
            (r2,) = b_eq.shape
            assert r == r2, "inconsistent equality dimensions " + repr((r,c,r2))
            if dimensions is None:
                dimensions = c
            assert dimensions == c, "inconsistent equality dimension " + repr((dimensions, c))
        if bounds:
            nbounds = len(bounds)
            if dimensions is None:
                dimensions = nbounds
            assert dimensions == nbounds, "bad bounds length " + repr((dimensions, nbounds))
        assert dimensions is not None, "Some constraints must be specified"
        self.dimensions = dimensions
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.bounds = bounds
        self.tol = tol
        self.gradient_delta = gradient_delta
        self.iteration_limit = iteration_limit
        self.other_linprog_options = other_linprog_options
        self.f_count = 0
        self.g_count = 0
        self.callback = None

    def objective(self, x):
        "evaluate objective function at x"
        self.f_count += 1
        result = self.f(x)
        #if self.verbose:
        #    print "   obj %s -> %s" % (x, result)
        return result

    def gradient(self, x):
        "evaluate gradient at x"
        self.g_count += 1
        return self.g(x)

    def optimize_objective_on_Line_segment0(self, start_point, end_point, iteration, blended_point, previous_end):
        "line segment interpolation fallback."
        if self.blend:
            start_point = blended_point
        verbose = self.verbose
        if verbose:
            print("   line segment " + repr((start_point, end_point)))
        f = self.objective
        def interpolate(fraction):
            return start_point * (1.0 - fraction) + end_point * fraction
        def scalar_f(fraction):
            return f(interpolate(fraction))
        res = minimize_scalar(scalar_f, bounds=(0.0, 1.0), method="bounded")
        if res.success:
            optimal_point = interpolate(res.x)
            if verbose:
                print("    interpolated by fallback " + repr((res.x, optimal_point)))
            # XXXX Each step must improve by at least half a tolerance -- right?
            difference = f(start_point) - f(optimal_point)
            if verbose:
                print("    improvement " + repr(difference))
            if difference < self.tol * 0.5:
                progress = False
                if difference < 0:
                    optimal_point = start_point
            else:
                progress = True
        else:
            optimal_point = None
        return (optimal_point, res, progress)

    def optimize_objective_on_Line_segment(self, start_point, end_point, iteration, blended_point, previous_end,
            ratio=0.5, limit=1000):
        f = self.objective
        if self.blend and not np.allclose(blended_point, end_point):
            start_point = blended_point
        vertex_ratio = self.vertex_ratio
        if vertex_ratio is not None and previous_end is not None:
            # mix the current and previous end points
            mixed_end = end_point + vertex_ratio * (previous_end - end_point)
            if not np.allclose(mixed_end, start_point) and f(mixed_end) < f(end_point):
                end_point = mixed_end
        left = start_point
        right = end_point
        f_left = f(left)
        f_right = f(right)
        count = 0
        done = False
        convex_ok = True
        while convex_ok and not done:
            count += 1
            middle = left + ratio * (right - left)
            f_middle = f(middle)
            if f_middle <= f_right:
                right = middle
                f_right = f_middle
            elif f_middle > f_left:
                convex_ok = False
            done = np.allclose(left, right) or f_right < f_left
            if count > limit:
                break
        if not done and self.scalar_fallback:
            return self.optimize_objective_on_Line_segment0(start_point, end_point, iteration, blended_point, previous_end)
        res = OptimizeResult()
        res.success = done
        res.start = start_point
        res.end = end_point
        res.x = right
        if not done:
            if not convex_ok:
                res.message = "objective is not convex on line segment"
            else:
                res.message = "limit reached seeking improvement on line segment"
        else:
            res.message = "improvement found"
        return (right, res, done)

    def optimize_linear_approximation(self, gradient_vector, estimate=None):
        verbose = self.verbose
        if verbose:
            print("optimizing " + repr(gradient_vector))
        other_linprog_options = self.other_linprog_options
        bounds = self.bounds
        result = linprog(
            gradient_vector,
            A_ub=self.A_ub, b_ub=self.b_ub,
            A_eq=self.A_eq, b_eq=self.b_eq,
            bounds = bounds,
            **other_linprog_options)
        vertex = result.x
        if verbose:
            print("optimized at " + repr(vertex))
        return (result, vertex)

    def check_convergence(self, gradient, estimate, vertex, result):
        difference = abs(gradient.dot(estimate) - gradient.dot(vertex))
        tol = self.tol
        result.convergence_value = difference
        result.tolerance = tol
        return (difference < tol)

    def next_estimate(self, estimate, iteration, blended_estimate, last_vertex):
        gradient = self.gradient(estimate)
        (result, vertex) = self.optimize_linear_approximation(gradient, estimate)
        result.scalar_result = None
        if result.success:
            converged = self.check_convergence(gradient, estimate, vertex, result)
            (next_estimate, scalar_result, progress) = self.optimize_objective_on_Line_segment(
                estimate, vertex, iteration, blended_estimate, last_vertex)
            result.scalar_result = scalar_result
            if not scalar_result.success:
                if converged:
                    next_estimate = estimate
                else:
                    result.success = False
                    result.message = "Could not optimize objective on line segment given by linear approximation."
            else:
                if not progress:
                    converged = True
        else:
            converged = False
            next_estimate = None
        result.runner = self
        result.estimate = estimate
        result.next_estimate = next_estimate
        result.iteration = iteration
        result.blend = blended_estimate
        result.vertex = vertex
        if self.callback:
            self.callback(result, converged, estimate, gradient, vertex, next_estimate, blended_estimate)
        return (result, converged, next_estimate, vertex)

    def choose_feasible_point(self):
        "repeat linear approximation until start vertex with min objective found."
        estimate = gradient = np.zeros((self.dimensions,))
        best_vertex = best_value = None
        done = False
        while not done:
            (result, estimate) = self.optimize_linear_approximation(gradient)
            if not result.success or (best_vertex is not None and np.allclose(best_vertex, estimate)):
                done = True  # could not optimize or repeat vertex.
            else:
                value = self.objective(estimate)
                if best_value is None or value < best_value:
                    best_vertex = estimate
                    best_value = value
                    gradient = self.gradient(estimate)
                else:
                    done = True  # found best starting vertex.
        return (result, best_vertex)

    def run(self):
        """
        Run the main loop to find the estimated solution.

        Returns
        -------
        A scipy.optimize.OptimizeResult consisting of the following fields::
            x : ndarray
                The independent variable vector which optimizes the problem.
            success : bool
                Returns True if the algorithm succeeded in finding an optimal
                solution.
            status : int
                An integer representing the exit status of the optimization::
                0: Optimization terminated successfully
                1: Iteration limit reached
                2: Problem appears to be infeasible
                3: Problem appears to be unbounded
            nit : int
                The number of iterations performed.
            message : str
                A string descriptor of the exit status of the optimization.
        """
        verbose = self.verbose
        (result, estimate) = self.choose_feasible_point()
        blended_estimate = estimate
        if verbose:
            print("feasible point " + repr(estimate))
        converged = False
        iteration_limit = self.iteration_limit
        blend_ratio = self.blend_ratio
        vertex = None
        for iteration in range(iteration_limit):
            #estimates.append(estimate)
            (result, converged, estimate, vertex) = self.next_estimate(estimate, iteration, blended_estimate, vertex)
            if verbose:
                print("next estimate " + repr(estimate))
            if converged:
                break
            if not result.success:
                break
            blended_estimate = blended_estimate + blend_ratio * (estimate - blended_estimate)
        result.f_count = self.f_count
        result.g_count = self.g_count
        if result.success:
            result.x = estimate
            result.fun = self.objective(estimate)
            result.nit = iteration
            #result.estimates = estimates
            if not converged:
                result.status = 1  # Iteration limit reached.
                result.success = False
                result.message = "Did not converge after " + str(iteration_limit) + " iterations."
            else:
                result.message = "Successful."
        return result

    def gradient_estimate(self, x):
        "finite difference gradient estimate for self.f at x."
        return approx_fprime(x, self.objective, self.gradient_delta)
    