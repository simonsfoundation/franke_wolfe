
from jp_svg_canvas import cartesian_svg
from IPython.display import display
import numpy as np

def illustrate_step(step, runner):
    xs = list(np.linspace(-1, 1, 100))
    result = step[0]
    scalar_result = result.scalar_result
    estimate = scalar_result.start
    next_estimate = scalar_result.x
    vertex = scalar_result.end
    def f(lmda):
        return runner.objective(estimate + lmda * (vertex - estimate))
    ys = [f(x) for x in xs]
    mx = np.min(xs)
    Mx = np.max(xs)
    my = np.min(ys)
    My = np.max(ys)
    sx = Mx - mx
    sy = My - my
    margin = 0.1 * max(sx, sy)
    margin2 = margin * 2
    D = cartesian_svg.sdoodle(mx - margin, my-margin, sx+margin2, sy+margin2, html_width=300, html_height=300)
    D.sequence(None, xs, ys)
    yn = f(next_estimate)
    D.sequence(None, [-1, 1], [yn, yn], "red")
    (x0, y0) = next_estimate
    D.axes(0, (My - my) * 0.5)
    D.embed()

class Illustration(object):

    def __init__(self, runner):
        self.runner = runner
        self.callback_args = []
        runner.callback = self.capture
        self.maxes = self.mins = None
        self.D = None
        runner.run()

    def illustrate_steps(self, start=0, end=None):
        runner = self.runner
        if end is not None:
            steps = self.callback_args[start:end]
        else:
            steps = self.callback_args[start:]
        print ("illustrating %s steps."  % len(steps))
        for step in steps:
            illustrate_step(step, runner)

    def capture(self, *args):
        self.callback_args.append(args)
        (result, converged, estimate, gradient, vertex, next_estimate, blended) = args
        for point in (estimate, vertex, next_estimate, blended):
            self.location(point)

    def location(self, point):
        if self.maxes is None:
            self.maxes = np.array(point)
        else:
            self.maxes = np.maximum(point, self.maxes)
        if self.mins is None:
            self.mins = np.array(point)
        else:
            self.mins = np.minimum(point, self.mins)

    def get_canvas(self, grid=25):
        if self.D is not None:
            return self.D
        (mx, my) = self.mins
        (Mx, My) = self.maxes
        sx = Mx - mx
        sy = My - my
        margin = max(sx, sy) * 0.1
        margin2 = 2*margin
        D = cartesian_svg.sdoodle(mx - margin, my-margin, sx+margin2, sy+margin2, html_width=500)
        D.rect(None, mx, my, sx, sy, "cornsilk")
        xs = np.linspace(mx, Mx, grid)
        ys = np.linspace(my, My, grid)
        #print mx, Mx, xs
        #print my, My, ys
        xg, yg = np.meshgrid(xs, ys, sparse=True)
        xyg = np.array([xg, yg])
        values = self.runner.objective(xyg)
        values = np.log(values + (1 - values.min()))
        #print values
        M = values.max()
        m = values.min()
        scaled = 255 * (values - m) * (1.0/(M-m))
        #print scaled
        dx = sx * 1.0/(grid-1)
        dy = sy * 1.0/(grid-1)
        for i in range(grid):
            for j in range(grid):
                s = scaled[j, i]
                (x, y) = xs[i], ys[j]
                color = "#ffff%02x" % int(s)
                #print (x, y), s, color
                D.rect(None, x - dx*0.5, y-dy*0.5, dx, dy, color)
                
        D.axes()
        self.D = D
        return D

    def draw(self):
        D = self.get_canvas()
        for (n, args) in enumerate(self.callback_args):
            self.last_args = args
            (result, converged, estimate, gradient, vertex, next_estimate, blended) = args
            (ex, ey) = estimate
            (vx, vy) = vertex
            (nx, ny) = next_estimate
            (ox, oy) = estimate + gradient * 100
            (bx, by) = blended
            D.line(None, ex, ey, vx, vy, "pink")
            D.line(None, bx, by, vx, vy, "#ffddaa")
            D.line(None, ex, ey, nx, ny, "#5555ff", 3)
            D.line(None, ex, ey, ox, oy, "#55ff55")
            D.text(None, ex, ey, repr(n))
            D.text(None, vx, vy, repr(n), "red")
        return D
