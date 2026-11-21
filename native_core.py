"""Optional loader for the native ThermalSim CPU core."""

import ctypes
import os
import sys
import threading
from pathlib import Path
from typing import Optional

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

        self.has_mg = False
        self._damped_jacobi_fn = None
        self._residual_restrict_fn = None
        self._prolong_add_fn = None
        self._tls = threading.local()

        if self.version >= 2:
            try:
                damped_jacobi = self.library.thermalsim_damped_jacobi
                residual_restrict = self.library.thermalsim_residual_restrict_2x2
                prolong_add = self.library.thermalsim_prolong_add_2x2
            except AttributeError:
                pass
            else:
                damped_jacobi.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                damped_jacobi.restype = ctypes.c_int

                residual_restrict.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                ]
                residual_restrict.restype = ctypes.c_int

                prolong_add.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                ]
                prolong_add.restype = ctypes.c_int

                self._damped_jacobi_fn = damped_jacobi
                self._residual_restrict_fn = residual_restrict
                self._prolong_add_fn = prolong_add
                self.has_mg = True

    @staticmethod
    def _pointer(array):
        if array.size == 0:
            return None
        return array.ctypes.data_as(_DOUBLE_POINTER)

    def _get_scratch(self, size):
        buf = getattr(self._tls, "jacobi_scratch", None)
        if buf is None or buf.size < size:
            buf = np.empty(size, dtype=np.float64)
            self._tls.jacobi_scratch = buf
        return buf if buf.size == size else buf[:size]

    @staticmethod
    def _validate_grid_arrays(layer_count, rows, cols, gx, gy, gz, *node_arrays):
        if layer_count <= 0 or rows <= 0 or cols <= 0:
            raise RuntimeError("Invalid structured grid dimensions.")
        node_count = layer_count * rows * cols
        if cols > 1 and gx.size != layer_count * rows * (cols - 1):
            raise RuntimeError("Invalid gx array size.")
        if rows > 1 and gy.size != layer_count * (rows - 1) * cols:
            raise RuntimeError("Invalid gy array size.")
        if layer_count > 1 and gz.size != (layer_count - 1) * rows * cols:
            raise RuntimeError("Invalid gz array size.")
        for arr in node_arrays:
            if arr.size != node_count:
                raise RuntimeError("Invalid nodal array size.")
        return node_count

    def apply_structured(self, layer_count, rows, cols, gx, gy, gz, boundary, vector):
        """Apply the structured operator and return a new result array."""
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        gx_array = _as_contiguous_f64(gx)
        gy_array = _as_contiguous_f64(gy)
        gz_array = _as_contiguous_f64(gz)
        boundary_array = _as_contiguous_f64(boundary)
        vector_array = _as_contiguous_f64(vector)
        self._validate_grid_arrays(
            lc, r, c, gx_array, gy_array, gz_array, boundary_array, vector_array
        )
        result = np.empty_like(vector_array)
        status = self._apply_fn(
            lc,
            r,
            c,
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

    def damped_jacobi(
        self,
        layer_count,
        rows,
        cols,
        gx,
        gy,
        gz,
        diag_mass,
        omega_inv_diag,
        rhs,
        value,
        iterations,
        zero_init=False,
    ):
        """Run damped Jacobi smoothing and return the updated contiguous float64 array."""
        if not self.has_mg:
            raise RuntimeError("Native multigrid kernels unavailable.")
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        n_iters = int(iterations)
        z_init = 1 if zero_init else 0
        if n_iters < 0:
            raise ValueError("iterations must be non-negative.")

        gx_array = _as_contiguous_f64(gx)
        gy_array = _as_contiguous_f64(gy)
        gz_array = _as_contiguous_f64(gz)
        diag_mass_array = _as_contiguous_f64(diag_mass)
        omega_inv_array = _as_contiguous_f64(omega_inv_diag)
        rhs_array = _as_contiguous_f64(rhs)

        if value is None:
            if not z_init:
                raise ValueError("value cannot be None when zero_init is False.")
            value_array = np.empty_like(rhs_array)
        else:
            value_array = _as_contiguous_f64(value)

        node_count = self._validate_grid_arrays(
            lc,
            r,
            c,
            gx_array,
            gy_array,
            gz_array,
            diag_mass_array,
            omega_inv_array,
            rhs_array,
            value_array,
        )
        scratch = self._get_scratch(node_count)
        status = self._damped_jacobi_fn(
            lc,
            r,
            c,
            self._pointer(gx_array),
            self._pointer(gy_array),
            self._pointer(gz_array),
            self._pointer(diag_mass_array),
            self._pointer(omega_inv_array),
            self._pointer(rhs_array),
            self._pointer(value_array),
            self._pointer(scratch),
            n_iters,
            z_init,
        )
        if status != 0:
            raise RuntimeError(f"Native damped Jacobi failed ({status}).")
        return value_array

    def residual_restrict_2x2(
        self,
        layer_count,
        rows,
        cols,
        gx,
        gy,
        gz,
        diag_mass,
        rhs,
        value,
    ):
        """Compute ``r = rhs - A(value)`` and 2x2-restrict into a flat float64 array."""
        if not self.has_mg:
            raise RuntimeError("Native multigrid kernels unavailable.")
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        gx_array = _as_contiguous_f64(gx)
        gy_array = _as_contiguous_f64(gy)
        gz_array = _as_contiguous_f64(gz)
        diag_mass_array = _as_contiguous_f64(diag_mass)
        rhs_array = _as_contiguous_f64(rhs)
        value_array = _as_contiguous_f64(value)
        self._validate_grid_arrays(
            lc,
            r,
            c,
            gx_array,
            gy_array,
            gz_array,
            diag_mass_array,
            rhs_array,
            value_array,
        )
        coarse_size = lc * ((r + 1) // 2) * ((c + 1) // 2)
        coarse_rhs = np.empty(coarse_size, dtype=np.float64)
        status = self._residual_restrict_fn(
            lc,
            r,
            c,
            self._pointer(gx_array),
            self._pointer(gy_array),
            self._pointer(gz_array),
            self._pointer(diag_mass_array),
            self._pointer(rhs_array),
            self._pointer(value_array),
            self._pointer(coarse_rhs),
        )
        if status != 0:
            raise RuntimeError(f"Native residual restriction failed ({status}).")
        return coarse_rhs

    def prolong_add_2x2(self, layer_count, rows, cols, coarse_correction, value):
        """In-place add piecewise-constant 2x2 prolongation into ``value``."""
        if not self.has_mg:
            raise RuntimeError("Native multigrid kernels unavailable.")
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        if lc <= 0 or r <= 0 or c <= 0:
            raise RuntimeError("Invalid structured grid dimensions.")
        if not isinstance(value, np.ndarray) or not value.flags.writeable:
            raise ValueError("value must be a writable numpy.ndarray.")

        coarse_array = _as_contiguous_f64(coarse_correction)
        expected_coarse = lc * ((r + 1) // 2) * ((c + 1) // 2)
        expected_fine = lc * r * c
        if coarse_array.size != expected_coarse:
            raise RuntimeError("Invalid coarse_correction array size.")
        if value.size != expected_fine:
            raise RuntimeError("Invalid fine value array size.")

        direct = value.dtype == np.float64 and value.flags.c_contiguous
        target = value if direct else np.ascontiguousarray(value, dtype=np.float64)
        status = self._prolong_add_fn(
            lc,
            r,
            c,
            self._pointer(coarse_array),
            self._pointer(target),
        )
        if status != 0:
            raise RuntimeError(f"Native prolongation failed ({status}).")
        if not direct:
            value[...] = target.reshape(value.shape)


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
HAS_NATIVE_MG = bool(NATIVE_CORE is not None and NATIVE_CORE.has_mg)


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


def damped_jacobi_native(
    layer_count,
    rows,
    cols,
    gx,
    gy,
    gz,
    diag_mass,
    omega_inv_diag,
    rhs,
    value,
    iterations,
    zero_init=False,
) -> Optional[np.ndarray]:
    """Run native damped Jacobi smoothing or return ``None`` when unavailable."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_mg:
        return None
    try:
        return NATIVE_CORE.damped_jacobi(
            layer_count,
            rows,
            cols,
            gx,
            gy,
            gz,
            diag_mass,
            omega_inv_diag,
            rhs,
            value,
            iterations,
            zero_init=zero_init,
        )
    except (OSError, RuntimeError, ValueError):
        return None


def residual_restrict_2x2_native(
    layer_count,
    rows,
    cols,
    gx,
    gy,
    gz,
    diag_mass,
    rhs,
    value,
) -> Optional[np.ndarray]:
    """Compute native fused residual + 2x2 restriction or return ``None``."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_mg:
        return None
    try:
        return NATIVE_CORE.residual_restrict_2x2(
            layer_count, rows, cols, gx, gy, gz, diag_mass, rhs, value
        )
    except (OSError, RuntimeError, ValueError):
        return None


def prolong_add_2x2_native(
    layer_count,
    rows,
    cols,
    coarse_correction,
    value,
) -> bool:
    """In-place add native 2x2 prolongation into ``value``; return success flag."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_mg:
        return False
    try:
        NATIVE_CORE.prolong_add_2x2(
            layer_count, rows, cols, coarse_correction, value
        )
        return True
    except (OSError, RuntimeError, ValueError):
        return False
