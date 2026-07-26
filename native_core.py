"""Optional loader for the Windows x64 ThermalSim CPU core."""

import ctypes
import os
from pathlib import Path

import numpy as np


class NativeCore:
    """Thin stable C-ABI wrapper around ``thermalsim_core.dll``."""

    def __init__(self, path):
        self.path = str(path)
        self.library = ctypes.CDLL(self.path)
        double_pointer = ctypes.POINTER(ctypes.c_double)
        self.library.thermalsim_core_version.argtypes = []
        self.library.thermalsim_core_version.restype = ctypes.c_int
        self.library.thermalsim_apply_structured.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
        ]
        self.library.thermalsim_apply_structured.restype = ctypes.c_int
        self.version = int(self.library.thermalsim_core_version())
        if self.version < 1:
            raise RuntimeError("Unsupported ThermalSim native-core version.")

    @staticmethod
    def _pointer(array):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def apply_structured(self, layer_count, rows, cols, gx, gy, gz, boundary, vector):
        """Apply the structured operator and return a new result array."""
        arrays = [
            np.ascontiguousarray(item, dtype=np.float64)
            for item in (gx, gy, gz, boundary, vector)
        ]
        gx_array, gy_array, gz_array, boundary_array, vector_array = arrays
        result = np.empty_like(vector_array)
        status = self.library.thermalsim_apply_structured(
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
    yield root / "thermalsim_core.dll"
    yield root / "native" / "bin" / "thermalsim_core.dll"


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
