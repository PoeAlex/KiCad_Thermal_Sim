"""Optional loader for the native ThermalSim CPU core."""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


def _as_contiguous_f64(arr):
    """Return ``arr`` as a C-contiguous float64 array without copying when possible."""
    if (
        isinstance(arr, np.ndarray)
        and arr.dtype == np.float64
        and arr.flags.c_contiguous
    ):
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _library_names(platform=None):
    """Return candidate shared-library filenames for the given platform."""
    plat = sys.platform if platform is None else platform
    if plat.startswith("win") or plat in ("cygwin", "msys"):
        return ("thermalsim_core.dll",)
    if plat == "darwin":
        return ("libthermalsim_core.dylib", "thermalsim_core.dylib")
    return ("libthermalsim_core.so", "thermalsim_core.so")


class NativeCore:
    """Thin stable C-ABI wrapper around the native ``thermalsim_core`` shared library."""

    def __init__(self, path):
        self.path = str(path)
        self.library = ctypes.CDLL(self.path)
        self.library.thermalsim_core_version.argtypes = []
        self.library.thermalsim_core_version.restype = ctypes.c_int
        self.library.thermalsim_apply_structured.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
            _DOUBLE_POINTER,
        ]
        self.library.thermalsim_apply_structured.restype = ctypes.c_int
        self._apply_fn = self.library.thermalsim_apply_structured
        self.version = int(self.library.thermalsim_core_version())
        if self.version < 1:
            raise RuntimeError("Unsupported ThermalSim native-core version.")

    @staticmethod
    def _pointer(array):
        if array.size == 0:
            return None
        return array.ctypes.data_as(_DOUBLE_POINTER)

    def apply_structured(self, layer_count, rows, cols, gx, gy, gz, boundary, vector):
        """Apply the structured operator and return a new result array."""
        gx_array = _as_contiguous_f64(gx)
        gy_array = _as_contiguous_f64(gy)
        gz_array = _as_contiguous_f64(gz)
        boundary_array = _as_contiguous_f64(boundary)
        vector_array = _as_contiguous_f64(vector)
        result = np.empty_like(vector_array)
        status = self._apply_fn(
            int(layer_count),
            int(rows),
            int(cols),
            self._pointer(gx_array),
            self._pointer(gy_array),
            self._pointer(gz_array),
            self._pointer(boundary_array),
            self._pointer(vector_array),
            self._pointer(result),
        )
        if status != 0:
            raise RuntimeError(f"Native structured operator failed ({status}).")
        return result


def _candidate_paths():
    override = os.environ.get("THERMALSIM_NATIVE_CORE")
    if override:
        yield Path(override)
    root = Path(__file__).resolve().parent
    names = _library_names()
    for name in names:
        yield root / name
    for name in names:
        yield root / "native" / "bin" / name


def _load_native_core():
    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            return NativeCore(path)
        except (OSError, RuntimeError, AttributeError):
            continue
    return None


NATIVE_CORE = _load_native_core()
HAS_NATIVE_CORE = NATIVE_CORE is not None


def apply_structured_native(layer_count, rows, cols, gx, gy, gz, boundary, vector):
    """Return a native operator result or ``None`` when unavailable."""
    if NATIVE_CORE is None:
        return None
    try:
        return NATIVE_CORE.apply_structured(
            layer_count, rows, cols, gx, gy, gz, boundary, vector
        )
    except (OSError, RuntimeError, ValueError):
        return None
