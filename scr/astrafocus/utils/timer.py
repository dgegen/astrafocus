import time
from functools import wraps

import numpy as np


class Catchtime:
    """
    Example
    with Catchtime() as t:
        f(initial)
    ttx[i] = t.time

    Catchtime.mean(f, N_sample=10, x=x)
    """
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.perf_counter() - self.time
        if self.verbose:
            self.readout = f"Time: {self.time:.10f} seconds"
            print(self.readout)

    @classmethod
    def mean(cls, f, N_sample=10, verbose=False, **kwargs):
        ttx = np.zeros(N_sample)
        for i in range(N_sample):
            with cls(verbose) as t:
                f(**kwargs)
            ttx[i] = t.time
        print(ttx.mean())
        return ttx


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:9.4f} seconds to execute.")
        return result

    return wrapper
