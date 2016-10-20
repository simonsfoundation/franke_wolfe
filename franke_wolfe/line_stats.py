
import numpy as np
from numpy.linalg import norm

class LineStats(object):

    "Line on the xy plane."

    def __init__(self, abc, epsilon=1e-10):
        "a * x + b * y + c == 0"
        abc = numpy.array(abc, dtype=np.float)
        self.abc = abc
        (a, b, c) = abc
        absa = abs(a)
        assert max(absa, abs(b)) > epsilon, "parameters too small."
        # find a point on the line
        if absa > epsilon:
            y = 0
            x = - c / a
        else:
            x = 0
            y = - c / b
        self.point = np.array([x, y], dtype=np.float)
        # find the line normal, gradient of a * x + b * y + c
        self.normal = np.array([a, b], dtype=np.float)
        # find a directional unit vector along the line, normal to gradient
        self.vector = np.array([-b, a], dtype=np.float)

    def project(self, points):
        "Return projection of points to nearest point on line with signed offsets."
        points = numpy.array(points, dtype=np.float)
        (npoints, two) = points.shape
        assert two == 2, "Only 2d points allowed."
        point = self.point.reshape((1,2))
        translated_points = points - point
        normal = self.normal
        offsets = translated_points.dot(normal)
        normal_components = offsets.reshape((npoints, 1)) * normal.reshape((1, 2))
        translated_projection = translated_points - normal_components
        projection = translated_projection + point
        return (projection, offsets)
