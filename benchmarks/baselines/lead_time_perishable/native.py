"""Load a versioned local C++ numerical kernel; never downloads dependencies."""
import ctypes
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import threading
import weakref

import numpy as np

_function = None
_lock = threading.Lock()


def reduced_batch(roots, periods, shelf, capacity, demands, probabilities, want_waste, want_inventory=True):
    global _function
    if _function is None:
        with _lock:
            if _function is None:
                source = Path(__file__).with_name('projection_native.cpp')
                signature = f"{sys.platform}_{source.stat().st_mtime_ns}_{source.stat().st_size}"
                cache = Path(tempfile.gettempdir()) / f'operational_sr_native_{os.getuid()}'
                cache.mkdir(mode=0o700, exist_ok=True)
                target = cache / f'ltp_projection_{signature}.so'
                if not target.exists():
                    # A private temporary build and atomic replace permit many
                    # isolated workers to start concurrently without partial .so.
                    with tempfile.TemporaryDirectory(dir=cache) as build:
                        temporary = Path(build) / 'projection.so'
                        try:
                            subprocess.run(['c++', '-O3', '-std=c++11', '-shared', '-fPIC',
                                            str(source), '-o', str(temporary)], check=True,
                                           capture_output=True, text=True, timeout=60)
                        except subprocess.CalledProcessError as exc:
                            raise RuntimeError(f'Native projection build failed: {exc.stderr}') from exc
                        os.replace(temporary, target)
                library = ctypes.CDLL(str(target))
                function = library.ltp_reduced_batch
                integer_array = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
                float_array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
                function.argtypes = [integer_array] + [ctypes.c_int]*5 + [integer_array,
                                      float_array, ctypes.c_int, ctypes.c_int, ctypes.c_int, float_array]
                function.restype = ctypes.c_int
                # Function retains the CDLL handle for its complete lifetime.
                function._library = library
                _function = function
    roots = np.ascontiguousarray(roots, dtype=np.int32)
    output = np.empty((len(roots), 2), dtype=np.float64)
    status = _function(roots, len(roots), roots.shape[1], shelf, periods, capacity,
                       np.ascontiguousarray(demands, dtype=np.int32),
                       np.ascontiguousarray(probabilities, dtype=np.float64),
                       len(demands), int(want_waste), int(want_inventory), output)
    if status:
        raise ValueError(f'Native reduced projection rejected dimensions: {status}')
    return output


def inventory_forward(roots, periods, shelf, capacity, demands, probabilities):
    if _function is None:
        # Initialize the shared versioned library without propagating any roots.
        reduced_batch(np.empty((0, 1), dtype=np.int32), 0, 1, capacity,
                      demands, probabilities, False)
    function = _function._library.ltp_inventory_forward
    integer_array = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    float_array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    function.argtypes = [integer_array] + [ctypes.c_int]*5 + [integer_array,
                        float_array, ctypes.c_int, float_array]
    function.restype = ctypes.c_int
    roots = np.ascontiguousarray(roots, dtype=np.int32)
    output = np.zeros((len(roots), 2), dtype=np.float64)
    status = function(roots, len(roots), roots.shape[1], shelf, periods, capacity,
        np.ascontiguousarray(demands, dtype=np.int32),
        np.ascontiguousarray(probabilities, dtype=np.float64), len(demands), output)
    if status:
        raise ValueError(f'Native cached projection rejected dimensions: {status}')
    return output


class LongProjector:
    """Own the complete native DAG and its bounded, PMF-specific caches."""

    def __init__(self, shelf, lead, capacity, quantity, expiry, cached_inventory,
                 layer_limit, demands, probabilities):
        reduced_batch(np.empty((0, 1), dtype=np.int32), 0, 1, capacity,
                      demands, probabilities, False)
        library = _function._library
        integers = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        floats = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        create = library.ltp_long_create
        create.argtypes = [ctypes.c_int]*7 + [integers, floats, ctypes.c_int]
        create.restype = ctypes.c_void_p
        destroy = library.ltp_long_destroy
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = None
        self.function = library.ltp_long_batch
        self.function.argtypes = [ctypes.c_void_p, integers, ctypes.c_int, floats,
            np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')]
        self.function.restype = ctypes.c_int
        self.context = create(shelf, lead, capacity, {'inventory':0, 'waste':1, 'both':2}[quantity],
            int(expiry), int(cached_inventory), layer_limit,
            np.ascontiguousarray(demands, dtype=np.int32),
            np.ascontiguousarray(probabilities, dtype=np.float64), len(demands))
        if not self.context:
            raise ValueError('Invalid native long projection dimensions')
        self._finalize = weakref.finalize(self, destroy, self.context)

    def batch(self, rows):
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        output = np.empty((len(rows), 2), dtype=np.float64)
        stats = np.zeros(4, dtype=np.uint64)
        if self.function(self.context, rows, len(rows), output, stats):
            raise ValueError('Native long projection rejected batch')
        return output, dict(zip(('native_expanded_states', 'native_cache_hits',
                                'native_demand_branches', 'native_reduced_roots'), map(int, stats)))
