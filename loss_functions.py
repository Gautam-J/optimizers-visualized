import numpy as np


def sphere_function(w):
    x, y = w[0], w[1]
    return x**2 + y**2


def hyperbolic_paraboloid(w):
    x, y = w[0], w[1]
    return x**2 - y**2


def rosenbrock(w):
    x, y = w[0], w[1]
    return (100 * ((y - x**2)**2)) + ((1 - x)**2)


def matyas(w):
    x, y = w[0], w[1]
    return 0.26 * (x**2 + y**2) - (0.48 * x * y)


def himmelblau(w):
    x, y = w[0], w[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def mc_cormick(w):
    x, y = w[0], w[1]
    return np.sin(x + y) + (x - y)**2 - (1.5 * x) + (2.5 * y) + 1


def styblinski_tang(w):
    x, y = w[0], w[1]
    return ((x**4 - 16 * (x**2) + 5 * x) + (y**4 - 16 * (y**2) + 5 * y)) / 2
