
import numpy as np
import time
from jp_svg_canvas import cartesian_svg
from franke_wolfe import fw_optimize

def line_illustration(a, b, c, x, y):
    m = min(x, y, -1)
    M = max(x, y, 1)
    ((x1, y1), (x2, y2)) = line_extrema(m, m, M, M, a, b, c)
    m = min(m, x1, y1, x2, y2)
    M = max(M, x1, y1, x2, y2)
    s = M - m
    D = cartesian_svg.sdoodle(m, m, M, M, html_width=300, html_height=300)
    N = 30
    mesh = np.linspace(m, M, N)
    (xs, ys) = np.meshgrid(mesh, mesh)
    distances = left_distance2(a,b,c,xs,ys)
    distances2 = right_distance2(a,b,c,xs,ys)
    dm = np.min(distances)
    dM = np.max(distances)
    dm2 = np.min(distances2)
    dM2 = np.max(distances2)
    dd = s * 1.0/(N - 1)
    for i in range(N):
        for j in range(N):
            x = xs[i,j]
            y = ys[i,j]
            d = distances[i,j]
            if d > 0:
                clr = color(d, dm, dM)
                D.rect(None, x, y, dd, dd, clr)
            d2 = distances2[i,j]
            if d2 > 0:
                clr2 = color2(d2, dm2, dM2)
                D.rect(None, x, y, dd, dd, clr2)
    ((x1, y1), (x2, y2)) = line_extrema(m, m, M, M, a, b, c)
    D.line(None, x1, y1, x2, y2)
    D.circle(None, x, y, s * 0.03, "pink")
    d = np.sqrt(line_point_distance2(a,b,c,x,y))
    D.text(None, x, y, " " + repr(d))
    D.axes()
    D.embed()

def color(v, m, M):
    d = M - m
    i = int(255 * (v - m) * 1.0 / d)
    assert i >= 0 and i <= 256, repr((i, v, m, M))
    ii = max(255 - i, 0)
    return "#%02x%02xff" % (ii, ii)

def color2(v, m, M):
    d = M - m
    i = int(255 * (v - m) * 1.0 / d)
    assert i >= 0 and i <= 256, repr((i, v, m, M))
    ii = max(255 - i, 0)
    return "#ff%02x%02x" % (ii, ii)

def line_point_distance2(a,b,c,x,y):
    return (a*x + b*y + c)**2/float(a**2 + b**2)

def line_offset(a, b, c, x, y):
    return a * x + b * y + c

def left_distance2(a, b, c, x, y):
    offset = line_offset(a, b, c, x, y)
    distance = line_point_distance2(a,b,c,x,y)
    return np.where(offset > 0, distance, 0.0)

def right_distance2(a, b, c, x, y):
    offset = line_offset(a, b, c, x, y)
    distance = line_point_distance2(a,b,c,x,y)
    return np.where(offset < 0, distance, 0.0)

def line_point_distance2_gradient_abc(a, b, c, x, y):
    a2b2inv = 1.0/(a**2 + b**2)
    axbyc = a * x + b * y + c
    da = - (2 * a * (a2b2inv ** 2)) * (axbyc ** 2) + (2 * x * a2b2inv * axbyc)
    db = - (2 * b * (a2b2inv ** 2)) * (axbyc ** 2) + (2 * y * a2b2inv * axbyc)
    dc = a2b2inv * 2 * axbyc
    return (da, db, dc)

def right_gradient(a, b, c, x, y):
    offset = line_offset(a, b, c, x, y)
    g = line_point_distance2_gradient_abc(a, b, c, x, y)
    g2 = [np.where(offset < 0, x, 0.0) for x in g]
    return np.array([np.sum(x) for x in g2])

def left_gradient(a, b, c, x, y):
    offset = line_offset(a, b, c, x, y)
    g = line_point_distance2_gradient_abc(a, b, c, x, y)
    g2 = [np.where(offset >= 0, x, 0.0) for x in g]
    return np.array([np.sum(x) for x in g2])

def line_extrema(minx, miny, maxx, maxy, a, b, c):
    if a != 0:
        def p(x,y):
            return (-(b*y + c)*1.0/a, y)
    else:
        assert b!=0
        def p(x,y):
            return (x, -(a*x + c)*1.0/b)
    return (p(minx, miny), p(maxx, maxy))

class RedBlueObjective(object):

    def __init__(self, red_points, green_points, limit=10):
        red_points = np.array(red_points)
        green_points = np.array(green_points)
        self.red_points = red_points
        self.green_points = green_points
        all_points = np.vstack([red_points, green_points])
        self.mins = all_points.min(axis=0)
        self.maxes = all_points.max(axis=0)
        self.limit = limit
        self.drawing = None
        self.point_radius = None
        self.callback_args = []
        self.last_abc = None

    def run(self, limit=10):
        extrema = np.hstack([self.mins, - self.mins, self.maxes, - self.maxes])
        MM = 2 * extrema.max()
        ii = MM
        bounds = [[-ii, ii], [-ii, ii], [-MM, MM]]
        print 'bounds', bounds
        limit = self.limit
        runner = fw_optimize.FrankWolfe(self.objective, self.gradient, bounds=bounds, iteration_limit=limit)
        self.runner = runner
        runner.callback = self.callback
        runner.run()

    def gradient(self, abc):
        (a,b,c) = abc
        r = self.red_points
        g = self.green_points
        result = left_gradient(a, b, c, r[:,0], r[:,1]) + right_gradient(a, b, c, g[:,0], g[:,1])
        return result

    def objective(self, abc):
        #print "objective for", abc
        (a,b,c) = abc
        r = self.red_points
        g = self.green_points
        r_errors = left_distance2(a, b, c, r[:,0], r[:,1])
        g_errors = right_distance2(a, b, c, g[:,0], g[:,1])
        result = r_errors.sum() + g_errors.sum()
        #print "result is", result
        return result

    def illustrate(self, abc):
        last_abc = self.last_abc
        (a,b,c) = abc
        D = self.get_drawing()
        m = self.mins.min()
        M = self.maxes.max()
        if last_abc is not None:
            for percent in range(100):
                (aa, bb, cc) = last_abc + 0.01 * percent * (abc - last_abc)
                ((x1, y1), (x2, y2)) = line_extrema(m, m, M, M, aa, bb, cc)
                D.line("interpolated", x1, y1, x2, y2, "magenta")
                time.sleep(0.01)
                D.delete("interpolated")
        self.last_abc = abc
        D.empty()
        D.axes()
        for (offset, color) in [(0.1, "red"), (-0.1, "green"), (0, "black")]:
            ((x1, y1), (x2, y2)) = line_extrema(m, m, M, M, a, b, c + offset)
            D.line(None, x1, y1, x2, y2, color)
        radius = self.point_radius
        for (x, y) in self.red_points:
            #print "circle", x, y, radius, "red"
            D.circle(None, x, y, radius, "red")
        for (x, y) in self.green_points:
            D.circle(None, x, y, radius, "green")

    def get_drawing(self):
        drawing = self.drawing
        if drawing is None:
            (mx, my) = self.mins
            (sx, sy) = self.maxes - self.mins
            self.point_radius = 1.0
            if sx > 0:
                self.point_radius = sx * 0.03
            margin = max(sx * 0.1, sy * 0.1, 1.0)
            margin2 = 2 * margin
            drawing = cartesian_svg.doodle(mx-margin, my-margin, sx+margin2, sy+margin2, html_width=500)
            self.drawing = drawing
            drawing.show()
        return drawing

    def callback(self, *args):
        self.callback_args.append(args)
        (result, converged, estimate, gradient, vertex, next_estimate, blended) = args
        abc = estimate
        print "abc", abc
        self.illustrate(abc)
        time.sleep(2)

def mvn_test():
    from numpy import random
    mean = (1,1)
    cov = np.diag([3,4])
    red_points = random.multivariate_normal(mean, cov, 20) - 1
    mean = (-1,-2)
    cov = np.diag([4,3])
    green_points = random.multivariate_normal(mean, cov, 20) 
    RB = RedBlueObjective(red_points, green_points)
    RB.run()
    return RB
