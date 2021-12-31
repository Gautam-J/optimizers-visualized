def matyas(w):
    x, y = w[0], w[1]
    return 0.26 * (x**2 + y**2) - (0.48 * x * y)


def himmelblau(w):
    x, y = w[0], w[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
