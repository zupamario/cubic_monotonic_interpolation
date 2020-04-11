import math

import numpy as np
import matplotlib.pyplot as plt


"""
Algorithm was taken from the javascript implementation at:
https://en.wikipedia.org/wiki/Monotone_cubic_interpolation
"""


def create_interpolant(points):
    copy_points = np.copy(points)
    new_indices = np.argsort(points[:, 0])
    sorted_points = copy_points[new_indices]
    xs = sorted_points[:, 0]
    ys = sorted_points[:, 1]

    # Get consecutive differences and slopes
    dys = []
    dxs = []
    ms = []
    for i in range(len(points) - 1):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        m = dy / dx
        dxs.append(dx)
        dys.append(dy)
        ms.append(m)

    # Get degree - 1 coefficients
    c1s = [ms[0]]
    for i in range(len(dxs) - 1):
        m = ms[i]
        m_next = ms[i + 1]
        if m * m_next <= 0.0:
            c1s.append(0)
        else:
            dx_ = dxs[i]
            dx_next = dxs[i + 1]
            common = dx_ + dx_next
            c1s.append(3.0 * common / ((common + dx_next) / m + (common + dx_) / m_next))
    c1s.append(ms[-1])

    c2s = []
    c3s = []
    for i in range(len(c1s) - 1):
        c1 = c1s[i]
        m_ = ms[i]
        inv_dx = 1.0 / dxs[i]
        common_ = c1 + c1s[i + 1] - m_ - m_
        c2s.append((m_ - c1 - common_) * inv_dx)
        c3s.append(common_ * inv_dx * inv_dx)

    def func(x):
        # The rightmost point in the dataset should give an exact result var
        i = len(xs) - 1
        if x == xs[i]:
            return ys[i]

        # Search for the interval x is in, returning the corresponding y if x is one of the original xs
        low = 0
        mid = len(c3s) - 1
        high = len(c3s) - 1
        while low <= high:
            mid = math.floor(0.5 * (low + high))
            xHere = xs[mid]
            if xHere < x:
                low = mid + 1
            elif xHere > x:
                high = mid - 1
            else:
                return ys[mid]

        i = max(0, high)

        # Interpolate
        diff = x - xs[i]
        diffSq = diff * diff
        return ys[i] + c1s[i] * diff + c2s[i] * diffSq + c3s[i] * diff * diffSq

    return func


if __name__ == '__main__':
    points = np.column_stack(([0.0, 1.0, 2.0, 2.2, 4.0], [4.0, 1.0, 12.0, 9.0, 16.0]))
    plt.plot(points[:, 0], points[:, 1], 'o')

    f = create_interpolant(points)
    coords_to_evaluate = np.linspace(points[0, 0], points[-1, 0], 100)
    results = [f(x) for x in coords_to_evaluate]
    plt.plot(coords_to_evaluate, results, 'o', markersize=1)

    plt.show()
