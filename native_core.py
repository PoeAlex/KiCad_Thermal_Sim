"""Optional loader for the native ThermalSim CPU core."""

import ctypes
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
_FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)
_UINT8_POINTER = ctypes.POINTER(ctypes.c_uint8)


def _as_contiguous_f64(arr):
    """Return ``arr`` as a C-contiguous float64 array without copying when possible."""
    if (
        isinstance(arr, np.ndarray)
        and arr.dtype == np.float64
        and arr.flags.c_contiguous
    ):
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_contiguous_f32(arr):
    """Return ``arr`` as a C-contiguous float32 array without copying when possible."""
    if (
        isinstance(arr, np.ndarray)
        and arr.dtype == np.float32
        and arr.flags.c_contiguous
    ):
        return arr
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_contiguous_u8(arr):
    """Return ``arr`` as a C-contiguous uint8 array without copying when possible."""
    if isinstance(arr, np.ndarray) and arr.flags.c_contiguous:
        if arr.dtype == np.uint8:
            return arr
        if arr.dtype == np.bool_:
            return arr.view(np.uint8)
    return np.ascontiguousarray(arr, dtype=np.uint8)


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
        self.has_compact = False
        self._damped_jacobi_fn = None
        self._residual_restrict_fn = None
        self._prolong_add_fn = None
        self._apply_compact_fn = None
        self._damped_jacobi_compact_f32_fn = None
        self._residual_restrict_compact_f32_fn = None
        self._coarsen_compact_to_f32_fn = None
        self._damped_jacobi_f32_fn = None
        self._residual_restrict_f32_fn = None
        self._prolong_add_f32_fn = None
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

        if self.version >= 3:
            try:
                apply_compact = self.library.thermalsim_apply_compact
                damped_jacobi_compact_f32 = self.library.thermalsim_damped_jacobi_compact_f32
                residual_restrict_compact_f32 = (
                    self.library.thermalsim_residual_restrict_2x2_compact_f32
                )
                coarsen_compact_to_f32 = self.library.thermalsim_coarsen_compact_to_f32
                damped_jacobi_f32 = self.library.thermalsim_damped_jacobi_f32
                residual_restrict_f32 = self.library.thermalsim_residual_restrict_2x2_f32
                prolong_add_f32 = self.library.thermalsim_prolong_add_2x2_f32
            except AttributeError:
                pass
            else:
                apply_compact.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    _UINT8_POINTER,
                    _FLOAT_POINTER,
                    _UINT8_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                ]
                apply_compact.restype = ctypes.c_int

                damped_jacobi_compact_f32.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    _UINT8_POINTER,
                    _FLOAT_POINTER,
                    _UINT8_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                damped_jacobi_compact_f32.restype = ctypes.c_int

                residual_restrict_compact_f32.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    _UINT8_POINTER,
                    _FLOAT_POINTER,
                    _UINT8_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                ]
                residual_restrict_compact_f32.restype = ctypes.c_int

                coarsen_compact_to_f32.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    _UINT8_POINTER,
                    _FLOAT_POINTER,
                    _UINT8_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    _DOUBLE_POINTER,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    ctypes.c_double,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                ]
                coarsen_compact_to_f32.restype = ctypes.c_int

                damped_jacobi_f32.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                damped_jacobi_f32.restype = ctypes.c_int

                residual_restrict_f32.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                ]
                residual_restrict_f32.restype = ctypes.c_int

                prolong_add_f32.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    _FLOAT_POINTER,
                    _FLOAT_POINTER,
                ]
                prolong_add_f32.restype = ctypes.c_int

                self._apply_compact_fn = apply_compact
                self._damped_jacobi_compact_f32_fn = damped_jacobi_compact_f32
                self._residual_restrict_compact_f32_fn = residual_restrict_compact_f32
                self._coarsen_compact_to_f32_fn = coarsen_compact_to_f32
                self._damped_jacobi_f32_fn = damped_jacobi_f32
                self._residual_restrict_f32_fn = residual_restrict_f32
                self._prolong_add_f32_fn = prolong_add_f32
                self.has_compact = True

    @staticmethod
    def _pointer(array):
        if array is None or array.size == 0:
            return None
        return array.ctypes.data_as(_DOUBLE_POINTER)

    @staticmethod
    def _f32_pointer(array):
        if array is None or array.size == 0:
            return None
        return array.ctypes.data_as(_FLOAT_POINTER)

    @staticmethod
    def _u8_pointer(array):
        if array is None or array.size == 0:
            return None
        return array.ctypes.data_as(_UINT8_POINTER)

    def _get_scratch(self, size):
        buf = getattr(self._tls, "jacobi_scratch", None)
        if buf is None or buf.size < size:
            buf = np.empty(size, dtype=np.float64)
            self._tls.jacobi_scratch = buf
        return buf if buf.size == size else buf[:size]

    def _get_scratch_f32(self, size):
        buf = getattr(self._tls, "jacobi_scratch_f32", None)
        if buf is None or buf.size < size:
            buf = np.empty(size, dtype=np.float32)
            self._tls.jacobi_scratch_f32 = buf
        return buf if buf.size == size else buf[:size]

    def clear_scratch(self):
        """Release any thread-local scratch buffers held on the current thread."""
        for attr in ("jacobi_scratch", "jacobi_scratch_f32"):
            if hasattr(self._tls, attr):
                delattr(self._tls, attr)

    @staticmethod
    def _validate_output_buffer(out, expected_size, dtype, shape=None):
        if out is None:
            return np.empty(expected_size if shape is None else shape, dtype=dtype)
        if (
            not isinstance(out, np.ndarray)
            or not out.flags.writeable
            or not out.flags.c_contiguous
            or out.dtype != dtype
        ):
            raise ValueError(f"out must be a writable C-contiguous {dtype} numpy.ndarray.")
        if out.size != expected_size:
            raise RuntimeError("Invalid output buffer size.")
        return out

    @staticmethod
    def _validate_scratch_buffer(scratch, min_size, dtype):
        if (
            not isinstance(scratch, np.ndarray)
            or not scratch.flags.writeable
            or not scratch.flags.c_contiguous
            or scratch.dtype != dtype
        ):
            raise ValueError(f"scratch must be a writable C-contiguous {dtype} numpy.ndarray.")
        if scratch.size < min_size:
            raise RuntimeError("Invalid scratch buffer size.")
        return scratch if scratch.size == min_size else scratch.reshape(-1)[:min_size]

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

    @staticmethod
    def _validate_compact_arrays(
        layer_count,
        rows,
        cols,
        copper_mask,
        via_enh,
        hs_mask,
        gx_lut,
        gy_lut,
        gz_scale,
        cap_lut,
        *node_arrays,
    ):
        if layer_count <= 0 or rows <= 0 or cols <= 0:
            raise RuntimeError("Invalid structured grid dimensions.")
        plane = rows * cols
        node_count = layer_count * plane
        if copper_mask.size != node_count:
            raise RuntimeError("Invalid copper_mask array size.")
        if hs_mask.size != plane:
            raise RuntimeError("Invalid hs_mask array size.")
        if gx_lut.size != layer_count * 3:
            raise RuntimeError("Invalid gx_lut array size.")
        if gy_lut.size != layer_count * 3:
            raise RuntimeError("Invalid gy_lut array size.")
        if cap_lut.size != layer_count * 2:
            raise RuntimeError("Invalid cap_lut array size.")
        if layer_count > 1:
            if via_enh is None or via_enh.size != plane:
                raise RuntimeError("Invalid via_enh array size.")
            if gz_scale is None or gz_scale.size != layer_count - 1:
                raise RuntimeError("Invalid gz_scale array size.")
        for arr in node_arrays:
            if arr.size != node_count:
                raise RuntimeError("Invalid nodal array size.")
        return node_count

    def apply_structured(self, layer_count, rows, cols, gx, gy, gz, boundary, vector, out=None):
        """Apply the structured operator and return a contiguous float64 result array."""
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        gx_array = _as_contiguous_f64(gx)
        gy_array = _as_contiguous_f64(gy)
        gz_array = _as_contiguous_f64(gz)
        boundary_array = _as_contiguous_f64(boundary)
        vector_array = _as_contiguous_f64(vector)
        node_count = self._validate_grid_arrays(
            lc, r, c, gx_array, gy_array, gz_array, boundary_array, vector_array
        )
        result = self._validate_output_buffer(
            out, node_count, np.float64, shape=vector_array.shape
        )
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
        scratch=None,
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
        scratch_buf = (
            self._get_scratch(node_count)
            if scratch is None
            else self._validate_scratch_buffer(scratch, node_count, np.float64)
        )
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
            self._pointer(scratch_buf),
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
        out=None,
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
        coarse_rhs = self._validate_output_buffer(out, coarse_size, np.float64)
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

    def apply_compact(
        self,
        layer_count,
        rows,
        cols,
        copper_mask,
        via_enh,
        hs_mask,
        gx_lut,
        gy_lut,
        gz_scale,
        cap_lut,
        pad_cap_cell,
        top_diag,
        bot_air_diag,
        bot_pad_diag,
        scale,
        vector,
        out=None,
    ):
        """Apply the compact Level-0 float64 operator and return the result array."""
        if not self.has_compact:
            raise RuntimeError("Native compact kernels unavailable.")
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        cm_array = _as_contiguous_u8(copper_mask)
        ve_array = None if via_enh is None else _as_contiguous_f32(via_enh)
        hs_array = _as_contiguous_u8(hs_mask)
        gx_lut_array = _as_contiguous_f64(gx_lut)
        gy_lut_array = _as_contiguous_f64(gy_lut)
        gz_scale_array = None if gz_scale is None else _as_contiguous_f64(gz_scale)
        cap_lut_array = _as_contiguous_f64(cap_lut)
        vector_array = _as_contiguous_f64(vector)

        node_count = self._validate_compact_arrays(
            lc,
            r,
            c,
            cm_array,
            ve_array,
            hs_array,
            gx_lut_array,
            gy_lut_array,
            gz_scale_array,
            cap_lut_array,
            vector_array,
        )
        result = self._validate_output_buffer(
            out, node_count, np.float64, shape=vector_array.shape
        )
        status = self._apply_compact_fn(
            lc,
            r,
            c,
            self._u8_pointer(cm_array),
            self._f32_pointer(ve_array),
            self._u8_pointer(hs_array),
            self._pointer(gx_lut_array),
            self._pointer(gy_lut_array),
            self._pointer(gz_scale_array),
            self._pointer(cap_lut_array),
            float(pad_cap_cell),
            float(top_diag),
            float(bot_air_diag),
            float(bot_pad_diag),
            float(scale),
            self._pointer(vector_array),
            self._pointer(result),
        )
        if status != 0:
            raise RuntimeError(f"Native compact operator failed ({status}).")
        return result

    def damped_jacobi_compact_f32(
        self,
        layer_count,
        rows,
        cols,
        copper_mask,
        via_enh,
        hs_mask,
        gx_lut,
        gy_lut,
        gz_scale,
        cap_lut,
        pad_cap_cell,
        top_diag,
        bot_air_diag,
        bot_pad_diag,
        scale,
        omega,
        rhs,
        value,
        iterations,
        zero_init=False,
        scratch=None,
    ):
        """Run compact Level-0 float32 damped Jacobi smoothing."""
        if not self.has_compact:
            raise RuntimeError("Native compact kernels unavailable.")
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        n_iters = int(iterations)
        z_init = 1 if zero_init else 0
        if n_iters < 0:
            raise ValueError("iterations must be non-negative.")

        cm_array = _as_contiguous_u8(copper_mask)
        ve_array = None if via_enh is None else _as_contiguous_f32(via_enh)
        hs_array = _as_contiguous_u8(hs_mask)
        gx_lut_array = _as_contiguous_f64(gx_lut)
        gy_lut_array = _as_contiguous_f64(gy_lut)
        gz_scale_array = None if gz_scale is None else _as_contiguous_f64(gz_scale)
        cap_lut_array = _as_contiguous_f64(cap_lut)
        rhs_array = _as_contiguous_f32(rhs)

        if value is None:
            if not z_init:
                raise ValueError("value cannot be None when zero_init is False.")
            value_array = np.empty_like(rhs_array)
        else:
            value_array = _as_contiguous_f32(value)
            if not value_array.flags.writeable:
                raise ValueError("value must be writable.")

        node_count = self._validate_compact_arrays(
            lc,
            r,
            c,
            cm_array,
            ve_array,
            hs_array,
            gx_lut_array,
            gy_lut_array,
            gz_scale_array,
            cap_lut_array,
            rhs_array,
            value_array,
        )
        scratch_buf = (
            self._get_scratch_f32(node_count)
            if scratch is None
            else self._validate_scratch_buffer(scratch, node_count, np.float32)
        )
        status = self._damped_jacobi_compact_f32_fn(
            lc,
            r,
            c,
            self._u8_pointer(cm_array),
            self._f32_pointer(ve_array),
            self._u8_pointer(hs_array),
            self._pointer(gx_lut_array),
            self._pointer(gy_lut_array),
            self._pointer(gz_scale_array),
            self._pointer(cap_lut_array),
            float(pad_cap_cell),
            float(top_diag),
            float(bot_air_diag),
            float(bot_pad_diag),
            float(scale),
            float(omega),
            self._f32_pointer(rhs_array),
            self._f32_pointer(value_array),
            self._f32_pointer(scratch_buf),
            n_iters,
            z_init,
        )
        if status != 0:
            raise RuntimeError(f"Native compact damped Jacobi failed ({status}).")
        return value_array

    def residual_restrict_2x2_compact_f32(
        self,
        layer_count,
        rows,
        cols,
        copper_mask,
        via_enh,
        hs_mask,
        gx_lut,
        gy_lut,
        gz_scale,
        cap_lut,
        pad_cap_cell,
        top_diag,
        bot_air_diag,
        bot_pad_diag,
        scale,
        rhs,
        value,
        out=None,
    ):
        """Compute compact Level-0 float32 residual and 2x2 restriction."""
        if not self.has_compact:
            raise RuntimeError("Native compact kernels unavailable.")
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        cm_array = _as_contiguous_u8(copper_mask)
        ve_array = None if via_enh is None else _as_contiguous_f32(via_enh)
        hs_array = _as_contiguous_u8(hs_mask)
        gx_lut_array = _as_contiguous_f64(gx_lut)
        gy_lut_array = _as_contiguous_f64(gy_lut)
        gz_scale_array = None if gz_scale is None else _as_contiguous_f64(gz_scale)
        cap_lut_array = _as_contiguous_f64(cap_lut)
        rhs_array = _as_contiguous_f32(rhs)
        value_array = _as_contiguous_f32(value)

        self._validate_compact_arrays(
            lc,
            r,
            c,
            cm_array,
            ve_array,
            hs_array,
            gx_lut_array,
            gy_lut_array,
            gz_scale_array,
            cap_lut_array,
            rhs_array,
            value_array,
        )
        coarse_size = lc * ((r + 1) // 2) * ((c + 1) // 2)
        coarse_rhs = self._validate_output_buffer(out, coarse_size, np.float32)
        status = self._residual_restrict_compact_f32_fn(
            lc,
            r,
            c,
            self._u8_pointer(cm_array),
            self._f32_pointer(ve_array),
            self._u8_pointer(hs_array),
            self._pointer(gx_lut_array),
            self._pointer(gy_lut_array),
            self._pointer(gz_scale_array),
            self._pointer(cap_lut_array),
            float(pad_cap_cell),
            float(top_diag),
            float(bot_air_diag),
            float(bot_pad_diag),
            float(scale),
            self._f32_pointer(rhs_array),
            self._f32_pointer(value_array),
            self._f32_pointer(coarse_rhs),
        )
        if status != 0:
            raise RuntimeError(f"Native compact residual restriction failed ({status}).")
        return coarse_rhs

    def coarsen_compact_to_f32(
        self,
        layer_count,
        rows,
        cols,
        copper_mask,
        via_enh,
        hs_mask,
        gx_lut,
        gy_lut,
        gz_scale,
        cap_lut,
        pad_cap_cell,
        top_diag,
        bot_air_diag,
        bot_pad_diag,
    ):
        """Coarsen compact Level-0 physics directly into Level-1 float32 arrays."""
        if not self.has_compact:
            raise RuntimeError("Native compact kernels unavailable.")
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        cm_array = _as_contiguous_u8(copper_mask)
        ve_array = None if via_enh is None else _as_contiguous_f32(via_enh)
        hs_array = _as_contiguous_u8(hs_mask)
        gx_lut_array = _as_contiguous_f64(gx_lut)
        gy_lut_array = _as_contiguous_f64(gy_lut)
        gz_scale_array = None if gz_scale is None else _as_contiguous_f64(gz_scale)
        cap_lut_array = _as_contiguous_f64(cap_lut)

        self._validate_compact_arrays(
            lc,
            r,
            c,
            cm_array,
            ve_array,
            hs_array,
            gx_lut_array,
            gy_lut_array,
            gz_scale_array,
            cap_lut_array,
        )
        coarse_rows = (r + 1) // 2
        coarse_cols = (c + 1) // 2
        coarse_nodes = lc * coarse_rows * coarse_cols

        gx_c = np.empty((lc, coarse_rows, max(coarse_cols - 1, 0)), dtype=np.float32)
        gy_c = np.empty((lc, max(coarse_rows - 1, 0), coarse_cols), dtype=np.float32)
        gz_c = np.empty((max(lc - 1, 0), coarse_rows, coarse_cols), dtype=np.float32)
        boundary_c = np.empty(coarse_nodes, dtype=np.float32)
        stiffness_diag_c = np.empty(coarse_nodes, dtype=np.float32)
        cap_c = np.empty(coarse_nodes, dtype=np.float32)

        status = self._coarsen_compact_to_f32_fn(
            lc,
            r,
            c,
            self._u8_pointer(cm_array),
            self._f32_pointer(ve_array),
            self._u8_pointer(hs_array),
            self._pointer(gx_lut_array),
            self._pointer(gy_lut_array),
            self._pointer(gz_scale_array),
            self._pointer(cap_lut_array),
            float(pad_cap_cell),
            float(top_diag),
            float(bot_air_diag),
            float(bot_pad_diag),
            self._f32_pointer(gx_c),
            self._f32_pointer(gy_c),
            self._f32_pointer(gz_c),
            self._f32_pointer(boundary_c),
            self._f32_pointer(stiffness_diag_c),
            self._f32_pointer(cap_c),
        )
        if status != 0:
            raise RuntimeError(f"Native compact coarsening failed ({status}).")
        return gx_c, gy_c, gz_c, boundary_c, stiffness_diag_c, cap_c

    def damped_jacobi_f32(
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
        scratch=None,
    ):
        """Run float32 damped Jacobi smoothing and return the updated array."""
        if not self.has_compact:
            raise RuntimeError("Native compact kernels unavailable.")
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        n_iters = int(iterations)
        z_init = 1 if zero_init else 0
        if n_iters < 0:
            raise ValueError("iterations must be non-negative.")

        gx_array = _as_contiguous_f32(gx)
        gy_array = _as_contiguous_f32(gy)
        gz_array = _as_contiguous_f32(gz)
        diag_mass_array = _as_contiguous_f32(diag_mass)
        omega_inv_array = _as_contiguous_f32(omega_inv_diag)
        rhs_array = _as_contiguous_f32(rhs)

        if value is None:
            if not z_init:
                raise ValueError("value cannot be None when zero_init is False.")
            value_array = np.empty_like(rhs_array)
        else:
            value_array = _as_contiguous_f32(value)
            if not value_array.flags.writeable:
                raise ValueError("value must be writable.")

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
        scratch_buf = (
            self._get_scratch_f32(node_count)
            if scratch is None
            else self._validate_scratch_buffer(scratch, node_count, np.float32)
        )
        status = self._damped_jacobi_f32_fn(
            lc,
            r,
            c,
            self._f32_pointer(gx_array),
            self._f32_pointer(gy_array),
            self._f32_pointer(gz_array),
            self._f32_pointer(diag_mass_array),
            self._f32_pointer(omega_inv_array),
            self._f32_pointer(rhs_array),
            self._f32_pointer(value_array),
            self._f32_pointer(scratch_buf),
            n_iters,
            z_init,
        )
        if status != 0:
            raise RuntimeError(f"Native float32 damped Jacobi failed ({status}).")
        return value_array

    def residual_restrict_2x2_f32(
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
        out=None,
    ):
        """Compute float32 ``r = rhs - A(value)`` and 2x2-restrict into float32."""
        if not self.has_compact:
            raise RuntimeError("Native compact kernels unavailable.")
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        gx_array = _as_contiguous_f32(gx)
        gy_array = _as_contiguous_f32(gy)
        gz_array = _as_contiguous_f32(gz)
        diag_mass_array = _as_contiguous_f32(diag_mass)
        rhs_array = _as_contiguous_f32(rhs)
        value_array = _as_contiguous_f32(value)
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
        coarse_rhs = self._validate_output_buffer(out, coarse_size, np.float32)
        status = self._residual_restrict_f32_fn(
            lc,
            r,
            c,
            self._f32_pointer(gx_array),
            self._f32_pointer(gy_array),
            self._f32_pointer(gz_array),
            self._f32_pointer(diag_mass_array),
            self._f32_pointer(rhs_array),
            self._f32_pointer(value_array),
            self._f32_pointer(coarse_rhs),
        )
        if status != 0:
            raise RuntimeError(f"Native float32 residual restriction failed ({status}).")
        return coarse_rhs

    def prolong_add_2x2_f32(self, layer_count, rows, cols, coarse_correction, value):
        """In-place add float32 piecewise-constant 2x2 prolongation into ``value``."""
        if not self.has_compact:
            raise RuntimeError("Native compact kernels unavailable.")
        lc = int(layer_count)
        r = int(rows)
        c = int(cols)
        if lc <= 0 or r <= 0 or c <= 0:
            raise RuntimeError("Invalid structured grid dimensions.")
        if not isinstance(value, np.ndarray) or not value.flags.writeable:
            raise ValueError("value must be a writable numpy.ndarray.")

        coarse_array = _as_contiguous_f32(coarse_correction)
        expected_coarse = lc * ((r + 1) // 2) * ((c + 1) // 2)
        expected_fine = lc * r * c
        if coarse_array.size != expected_coarse:
            raise RuntimeError("Invalid coarse_correction array size.")
        if value.size != expected_fine:
            raise RuntimeError("Invalid fine value array size.")

        direct = value.dtype == np.float32 and value.flags.c_contiguous
        target = value if direct else np.ascontiguousarray(value, dtype=np.float32)
        status = self._prolong_add_f32_fn(
            lc,
            r,
            c,
            self._f32_pointer(coarse_array),
            self._f32_pointer(target),
        )
        if status != 0:
            raise RuntimeError(f"Native float32 prolongation failed ({status}).")
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
HAS_NATIVE_COMPACT = bool(NATIVE_CORE is not None and NATIVE_CORE.has_compact)


def clear_native_scratch() -> None:
    """Release any thread-local native scratch buffers held on the current thread."""
    if NATIVE_CORE is not None:
        NATIVE_CORE.clear_scratch()


def apply_structured_native(
    layer_count, rows, cols, gx, gy, gz, boundary, vector, out=None
):
    """Return a native operator result or ``None`` when unavailable."""
    if NATIVE_CORE is None:
        return None
    try:
        return NATIVE_CORE.apply_structured(
            layer_count, rows, cols, gx, gy, gz, boundary, vector, out=out
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
    scratch=None,
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
            scratch=scratch,
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
    out=None,
) -> Optional[np.ndarray]:
    """Compute native fused residual + 2x2 restriction or return ``None``."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_mg:
        return None
    try:
        return NATIVE_CORE.residual_restrict_2x2(
            layer_count, rows, cols, gx, gy, gz, diag_mass, rhs, value, out=out
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


def apply_compact_native(
    layer_count,
    rows,
    cols,
    copper_mask,
    via_enh,
    hs_mask,
    gx_lut,
    gy_lut,
    gz_scale,
    cap_lut,
    pad_cap_cell,
    top_diag,
    bot_air_diag,
    bot_pad_diag,
    scale,
    vector,
    out=None,
) -> Optional[np.ndarray]:
    """Apply the compact Level-0 float64 operator or return ``None`` when unavailable."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_compact:
        return None
    try:
        return NATIVE_CORE.apply_compact(
            layer_count,
            rows,
            cols,
            copper_mask,
            via_enh,
            hs_mask,
            gx_lut,
            gy_lut,
            gz_scale,
            cap_lut,
            pad_cap_cell,
            top_diag,
            bot_air_diag,
            bot_pad_diag,
            scale,
            vector,
            out=out,
        )
    except (OSError, RuntimeError, ValueError):
        return None


def damped_jacobi_compact_f32_native(
    layer_count,
    rows,
    cols,
    copper_mask,
    via_enh,
    hs_mask,
    gx_lut,
    gy_lut,
    gz_scale,
    cap_lut,
    pad_cap_cell,
    top_diag,
    bot_air_diag,
    bot_pad_diag,
    scale,
    omega,
    rhs,
    value,
    iterations,
    zero_init=False,
    scratch=None,
) -> Optional[np.ndarray]:
    """Run compact Level-0 float32 damped Jacobi or return ``None`` when unavailable."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_compact:
        return None
    try:
        return NATIVE_CORE.damped_jacobi_compact_f32(
            layer_count,
            rows,
            cols,
            copper_mask,
            via_enh,
            hs_mask,
            gx_lut,
            gy_lut,
            gz_scale,
            cap_lut,
            pad_cap_cell,
            top_diag,
            bot_air_diag,
            bot_pad_diag,
            scale,
            omega,
            rhs,
            value,
            iterations,
            zero_init=zero_init,
            scratch=scratch,
        )
    except (OSError, RuntimeError, ValueError):
        return None


def residual_restrict_2x2_compact_f32_native(
    layer_count,
    rows,
    cols,
    copper_mask,
    via_enh,
    hs_mask,
    gx_lut,
    gy_lut,
    gz_scale,
    cap_lut,
    pad_cap_cell,
    top_diag,
    bot_air_diag,
    bot_pad_diag,
    scale,
    rhs,
    value,
    out=None,
) -> Optional[np.ndarray]:
    """Compute compact Level-0 float32 residual + 2x2 restriction or return ``None``."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_compact:
        return None
    try:
        return NATIVE_CORE.residual_restrict_2x2_compact_f32(
            layer_count,
            rows,
            cols,
            copper_mask,
            via_enh,
            hs_mask,
            gx_lut,
            gy_lut,
            gz_scale,
            cap_lut,
            pad_cap_cell,
            top_diag,
            bot_air_diag,
            bot_pad_diag,
            scale,
            rhs,
            value,
            out=out,
        )
    except (OSError, RuntimeError, ValueError):
        return None


def coarsen_compact_to_f32_native(
    layer_count,
    rows,
    cols,
    copper_mask,
    via_enh,
    hs_mask,
    gx_lut,
    gy_lut,
    gz_scale,
    cap_lut,
    pad_cap_cell,
    top_diag,
    bot_air_diag,
    bot_pad_diag,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Coarsen compact Level-0 physics into Level-1 float32 arrays or return ``None``."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_compact:
        return None
    try:
        return NATIVE_CORE.coarsen_compact_to_f32(
            layer_count,
            rows,
            cols,
            copper_mask,
            via_enh,
            hs_mask,
            gx_lut,
            gy_lut,
            gz_scale,
            cap_lut,
            pad_cap_cell,
            top_diag,
            bot_air_diag,
            bot_pad_diag,
        )
    except (OSError, RuntimeError, ValueError):
        return None


def damped_jacobi_f32_native(
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
    scratch=None,
) -> Optional[np.ndarray]:
    """Run coarse-level float32 damped Jacobi or return ``None`` when unavailable."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_compact:
        return None
    try:
        return NATIVE_CORE.damped_jacobi_f32(
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
            scratch=scratch,
        )
    except (OSError, RuntimeError, ValueError):
        return None


def residual_restrict_2x2_f32_native(
    layer_count,
    rows,
    cols,
    gx,
    gy,
    gz,
    diag_mass,
    rhs,
    value,
    out=None,
) -> Optional[np.ndarray]:
    """Compute float32 residual + 2x2 restriction or return ``None`` when unavailable."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_compact:
        return None
    try:
        return NATIVE_CORE.residual_restrict_2x2_f32(
            layer_count,
            rows,
            cols,
            gx,
            gy,
            gz,
            diag_mass,
            rhs,
            value,
            out=out,
        )
    except (OSError, RuntimeError, ValueError):
        return None


def prolong_add_2x2_f32_native(
    layer_count,
    rows,
    cols,
    coarse_correction,
    value,
) -> bool:
    """In-place add float32 2x2 prolongation into ``value``; return success flag."""
    if NATIVE_CORE is None or not NATIVE_CORE.has_compact:
        return False
    try:
        NATIVE_CORE.prolong_add_2x2_f32(
            layer_count, rows, cols, coarse_correction, value
        )
        return True
    except (OSError, RuntimeError, ValueError):
        return False
