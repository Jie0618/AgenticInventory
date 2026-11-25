"""Concurrent family evaluation must not race SciPy's shared Sobol tables."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
from scipy.stats import qmc

from optimization.solvers.pipeline import _sobol


def test_sobol_serializes_lazy_initialization(monkeypatch):
    active = 0
    maximum = 0
    guard = threading.Lock()

    class Sobol:
        def __init__(self, dimensions, **kwargs):
            nonlocal active, maximum
            self.dimensions = dimensions
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.002)
            with guard:
                active -= 1

        def random_base2(self, exponent):
            return np.full((2 ** exponent, self.dimensions), .25)

    monkeypatch.setattr(qmc, "Sobol", Sobol)
    with ThreadPoolExecutor(max_workers=8) as pool:
        points = list(pool.map(lambda d: _sobol(3, d, 123), range(1, 17)))
    assert maximum == 1
    for dimensions, value in enumerate(points, 1):
        assert value.shape == (3, dimensions)
        assert np.all(value == .25)
