"""
Thermal solver using finite volume method with BDF2 time integration.

This module implements the core numerical solver for transient thermal
simulation, including sparse matrix assembly and time-stepping.
"""

import contextlib
import ctypes
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .native_core import HAS_NATIVE_CORE, apply_structured_native

try:
    from .native_core import (
        HAS_NATIVE_MG,
        damped_jacobi_native,
        residual_restrict_2x2_native,
        prolong_add_2x2_native,
    )
except ImportError:
    HAS_NATIVE_MG = False
    damped_jacobi_native = None
    residual_restrict_2x2_native = None
    prolong_add_2x2_native = None

try:
    from .native_core import (
        HAS_NATIVE_COMPACT,
        apply_compact_native,
        damped_jacobi_compact_f32_native,
        residual_restrict_2x2_compact_f32_native,
        coarsen_compact_to_f32_native,
        damped_jacobi_f32_native,
        residual_restrict_2x2_f32_native,
        prolong_add_2x2_f32_native,
        clear_native_scratch,
    )
except ImportError:
    HAS_NATIVE_COMPACT = False
    apply_compact_native = None
    damped_jacobi_compact_f32_native = None
    residual_restrict_2x2_compact_f32_native = None
    coarsen_compact_to_f32_native = None
    damped_jacobi_f32_native = None
    residual_restrict_2x2_f32_native = None
    prolong_add_2x2_f32_native = None

    def clear_native_scratch():
        pass

try:
    from .native_core import (
        HAS_NATIVE_PCG,
        apply_compact_dot_native,
        pcg_update_xr_norm2_native,
        pcg_update_p_native,
        f64_to_f32_native,
        f32_to_f64_dot_native,
        dot_f64_native,
    )
except ImportError:
    HAS_NATIVE_PCG = False
    apply_compact_dot_native = None
    pcg_update_xr_norm2_native = None
    pcg_update_p_native = None
    f64_to_f32_native = None
    f32_to_f64_dot_native = None
    dot_f64_native = None


def _as_1d_contiguous_f64(arr) -> np.ndarray:
    """Return ``arr`` as a 1D C-contiguous float64 array without copying when possible."""
    if (
        isinstance(arr, np.ndarray)
        and arr.dtype == np.float64
        and arr.ndim == 1
        and arr.flags.c_contiguous
    ):
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64).reshape(-1)


def _dot_f64(a: np.ndarray, b: np.ndarray) -> float:
    """Compute float64 dot product using native OpenMP kernel when available."""
    if HAS_NATIVE_PCG and dot_f64_native is not None:
        val = dot_f64_native(a, b)
        if val is not None:
            return float(val)
    return float(np.dot(a, b))


def _norm_f64(vec: np.ndarray) -> float:
    """Compute float64 Euclidean norm using native OpenMP dot kernel when available."""
    norm2 = _dot_f64(vec, vec)
    return math.sqrt(norm2) if not (norm2 < 0.0) else 0.0


HEATSINK_MATERIAL_CP = {
    "aluminium": 897.0,
    "copper": 385.0,
    "steel": 490.0,
    "magnesium": 1020.0,
}


def resolve_heatsink_cp(settings: Dict) -> float:
    """Resolve specific heat capacity (J/(kg*K)) for the heatsink material."""
    mat = str(settings.get("heatsink_material", "aluminium") or "aluminium").strip().lower()
    if mat == "custom":
        val = settings.get("heatsink_cp_j_kgk", 897.0)
        return max(0.0, float(897.0 if val is None else val))
    return float(HEATSINK_MATERIAL_CP.get(mat, HEATSINK_MATERIAL_CP["aluminium"]))


def compute_heatsink_capacity_j_k(settings: Dict) -> float:
    """Compute lumped 0D heatsink thermal capacity (J/K) from mass and material."""
    mass_val = settings.get("heatsink_mass_g", 50.0)
    mass_g = max(0.0, float(50.0 if mass_val is None else mass_val))
    return mass_g * 1e-3 * resolve_heatsink_cp(settings)


@dataclass
class SparsePowerVector:
    """Sparse nodal heat source vector storing only active node indices and values."""

    indices: np.ndarray
    values: np.ndarray
    size: int

    def __post_init__(self):
        self.indices = np.ascontiguousarray(self.indices, dtype=np.int64).reshape(-1)
        self.values = np.ascontiguousarray(self.values, dtype=np.float64).reshape(-1)
        self.size = int(self.size)
        if self.indices.size != self.values.size:
            raise ValueError(
                f"SparsePowerVector indices ({self.indices.size}) and "
                f"values ({self.values.size}) lengths must match."
            )

    def total_power(self) -> float:
        """Return total injected heat power (W)."""
        return float(np.sum(self.values))

    def add_to(self, out: np.ndarray) -> np.ndarray:
        """Accumulate sparse heat sources in-place into ``out``."""
        if self.indices.size:
            np.add.at(out, self.indices, self.values)
        return out

    def to_dense(self) -> np.ndarray:
        """Materialize a dense 1D float64 power vector."""
        dense = np.zeros(self.size, dtype=np.float64)
        self.add_to(dense)
        return dense


@dataclass
class HeatsinkModel:
    """Lumped 0D dynamic heatsink coupled to bottom-layer thermal pad nodes."""

    pad_conductance: np.ndarray
    rth_kw: float
    capacity_j_k: float = 0.0
    initial_temp_c: Optional[float] = None
    mode: str = "heatsink"
    bottom_layer_only: bool = False

    def __post_init__(self):
        if self.pad_conductance is not None:
            self.pad_conductance = np.ascontiguousarray(
                self.pad_conductance, dtype=np.float64
            ).reshape(-1)

    @property
    def g_hs_amb(self) -> float:
        """Heatsink-to-ambient thermal conductance (W/K)."""
        return 1.0 / max(float(self.rth_kw), 1e-9)

    @property
    def tau_s(self) -> float:
        """Lumped heatsink thermal time constant R_th * C_hs (seconds)."""
        return max(0.0, float(self.capacity_j_k)) * max(float(self.rth_kw), 0.0)

    @property
    def is_active(self) -> bool:
        """Return True when at least one PCB node couples to the heatsink."""
        return (
            self.pad_conductance is not None
            and self.pad_conductance.size > 0
            and float(np.sum(self.pad_conductance)) > 0.0
        )


def build_heatsink_coupling_vector(
    layer_count: int,
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    H_map: np.ndarray,
    settings: Dict,
) -> np.ndarray:
    """Build the 1D pad conductance vector g (W/K) on bottom-layer heatsink nodes."""
    pad_thick_m = max(0.0001, float(settings.get("pad_th", 1.0)) * 1e-3)
    pad_k = float(settings.get("pad_k", 3.0))
    contact_factor = max(1e-6, float(settings.get("pad_contact_factor", 0.2)))
    h_sink = (pad_k / pad_thick_m) * contact_factor
    plane_cells = rows * cols
    g = np.zeros(layer_count * plane_cells, dtype=np.float64)
    g[(layer_count - 1) * plane_cells :] = (
        np.asarray(H_map, dtype=np.float64) * h_sink * (dx * dy)
    ).ravel()
    return g


def build_heatsink_model(
    layer_count: int,
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    H_map: np.ndarray,
    settings: Dict,
    amb: float,
) -> Optional[HeatsinkModel]:
    """Construct a HeatsinkModel when dynamic heatsink mode is active on H_map."""
    if not bool(settings.get("use_heatsink", True)):
        return None
    sink_mode = str(settings.get("pad_sink_mode", "fixed") or "fixed").strip().lower()
    if sink_mode != "heatsink":
        return None
    h_map_arr = np.asarray(H_map, dtype=np.float64)
    if not np.any(h_map_arr):
        return None
    pad_conductance = build_heatsink_coupling_vector(
        layer_count, rows, cols, dx, dy, h_map_arr, settings
    )
    rth_val = settings.get("heatsink_rth_kw", 5.0)
    rth_kw = max(1e-6, float(5.0 if rth_val is None else rth_val))
    return HeatsinkModel(
        pad_conductance=pad_conductance,
        rth_kw=rth_kw,
        capacity_j_k=compute_heatsink_capacity_j_k(settings),
        initial_temp_c=float(amb),
    )


@dataclass
class SolverConfig:
    """
    Configuration for the thermal solver.

    Attributes
    ----------
    sim_time : float
        Total simulation time in seconds.
    amb : float
        Ambient temperature in degrees Celsius.
    dt_base : float
        Base time step in seconds.
    steps_target : int
        Target number of time steps.
    use_multi_phase : bool
        Whether to use multi-phase time stepping.
    snapshots_enabled : bool
        Whether to capture time-series snapshots.
    snap_times : List[float]
        Times at which to capture snapshots.
    steady_state : bool
        When True (and heat source is constant), solve ``K * T = Q + b`` directly
        in one linear solve instead of transient time stepping.
    early_stop : bool
        When True (and heat source is constant with no pending snapshots), stop
        transient stepping early once steady-state equilibrium is reached.
    """
    sim_time: float
    amb: float
    dt_base: float
    steps_target: int
    use_multi_phase: bool = True
    snapshots_enabled: bool = False
    snap_times: List[float] = None
    time_stepping: str = "multi_phase"
    steady_state: bool = False
    early_stop: bool = True

    def __post_init__(self):
        if self.snap_times is None:
            self.snap_times = []
        mode = str(self.time_stepping or "multi_phase").strip().lower()
        if mode not in {"auto", "multi_phase", "two_phase", "uniform"}:
            mode = "multi_phase"
        self.time_stepping = mode


def _build_phase_plan(config: SolverConfig, node_count: int):
    """Return the validated time-step phase plan for a simulation."""
    mode = config.time_stepping
    auto_large = False
    if not config.use_multi_phase:
        mode = "uniform"
    elif mode == "auto":
        # Large direct-solver systems pay a high price for every distinct
        # matrix factorization (6 factorizations for 3-phase BE+BDF2 vs 1 for
        # derivative-start uniform BDF2). Switch to single-factorization
        # uniform BDF2 at 100k nodes.
        auto_large = node_count >= 100_000
        mode = "uniform" if auto_large else "multi_phase"

    if mode == "multi_phase":
        phase_defs = [
            {"name": "A", "frac": 0.08, "dt_scale": 0.5},
            {"name": "B", "frac": 0.35, "dt_scale": 1.0},
            {"name": "C", "frac": 0.57, "dt_scale": 2.0},
        ]
    elif mode == "two_phase":
        phase_defs = [
            {"name": "A", "frac": 0.15, "dt_scale": 0.5},
            {"name": "B", "frac": 0.85, "dt_scale": 1.5},
        ]
    else:
        phase_defs = [{"name": "uniform", "frac": 1.0, "dt_scale": 1.0}]
        if auto_large:
            legacy_defs = ((0.08, 0.5), (0.35, 1.0), (0.57, 2.0))
            phase_defs[0]["steps_override"] = sum(
                max(1, int(round(config.steps_target * fraction / scale)))
                for fraction, scale in legacy_defs
            )
            phase_defs[0]["derivative_start"] = True

    phase_times = [config.sim_time * p["frac"] for p in phase_defs]
    phase_times[-1] = config.sim_time - sum(phase_times[:-1])
    plan = []
    for phase_time, pdef in zip(phase_times, phase_defs):
        requested_dt = config.dt_base * pdef["dt_scale"]
        step_count = int(pdef.get("steps_override", max(1, int(round(phase_time / requested_dt)))))
        phase_dt = phase_time / step_count if phase_time > 0 else config.dt_base
        plan.append((pdef, phase_time, step_count, phase_dt))
    return mode, plan


@dataclass
class SolverResult:
    """
    Results from the thermal solver.

    Attributes
    ----------
    T : np.ndarray
        Final temperature distribution, shape (layers, rows, cols).
    aborted : bool
        True if simulation was cancelled by user.
    step_counter : int
        Number of time steps completed.
    total_solve_time : float
        Total time spent in linear solves (seconds).
    total_factor_time : float
        Total time spent in matrix factorization (seconds).
    factor_count : int
        Number of factorizations performed.
    snapshot_stats : List[Dict]
        Statistics for each captured snapshot.
    snapshot_files : List[tuple]
        List of (time, filename) for saved snapshots.
    phase_metrics : List[Dict]
        Performance metrics for each solver phase.
    balance_history : List[Dict]
        Energy balance history for convergence checking.
    k_norm_info : Dict
        Normalization and debug information.
    """
    T: np.ndarray
    aborted: bool
    step_counter: int
    total_solve_time: float
    total_factor_time: float
    factor_count: int
    snapshot_stats: List[Dict]
    snapshot_files: List[tuple]
    phase_metrics: List[Dict]
    balance_history: List[Dict]
    k_norm_info: Dict


@dataclass
class CachedLinearSolver:
    """Reusable factorization for an unchanged implicit BDF2 operator."""

    solve: Callable[[np.ndarray], np.ndarray]
    owner: Any
    backend: str

    def release(self):
        """Release the optional native factorization."""
        _release_linear_solver(self.owner)
        self.owner = None


def _restrict_2x2_into(array: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Conservatively sum 2x2 spatial blocks of ``(layers, rows, cols)`` into ``out``."""
    layers, rows, cols = array.shape
    coarse_rows = (rows + 1) // 2
    coarse_cols = (cols + 1) // 2
    even_rows = (rows // 2) * 2
    even_cols = (cols // 2) * 2

    if rows == even_rows:
        row_sum = array[:, 0::2, :] + array[:, 1::2, :]
    else:
        row_sum = np.empty((layers, coarse_rows, cols), dtype=array.dtype)
        if even_rows > 0:
            row_sum[:, :-1, :] = (
                array[:, 0:even_rows:2, :] + array[:, 1:even_rows:2, :]
            )
        row_sum[:, -1, :] = array[:, -1, :]

    if cols == even_cols:
        np.add(row_sum[:, :, 0::2], row_sum[:, :, 1::2], out=out)
        return out
    if even_cols > 0:
        np.add(
            row_sum[:, :, 0:even_cols:2],
            row_sum[:, :, 1:even_cols:2],
            out=out[:, :, :-1],
        )
    out[:, :, -1] = row_sum[:, :, -1]
    return out


def _restrict_2x2(array: np.ndarray, dtype=None) -> np.ndarray:
    """Conservatively sum 2x2 spatial blocks on a ``(layers, rows, cols)`` array."""
    layers, rows, cols = array.shape
    out_dtype = array.dtype if dtype is None else dtype
    out = np.empty(
        (layers, (rows + 1) // 2, (cols + 1) // 2),
        dtype=out_dtype,
    )
    return _restrict_2x2_into(array, out)


def _coarsen_structured_operator(
    operator: "StructuredThermalOperator",
    dtype=np.float64,
) -> "StructuredThermalOperator":
    """Return the exact Galerkin 2x2 spatial coarsening ``P^T K P``."""
    layers = operator.layer_count
    rows = operator.rows
    cols = operator.cols
    coarse_rows = (rows + 1) // 2
    coarse_cols = (cols + 1) // 2
    even_rows = (rows // 2) * 2
    even_cols = (cols // 2) * 2

    if coarse_cols > 1:
        gx_odd = operator.gx[:, :, 1::2]
        if rows == even_rows:
            gx_c = gx_odd[:, 0::2, :] + gx_odd[:, 1::2, :]
        else:
            gx_c = np.empty((layers, coarse_rows, coarse_cols - 1), dtype=dtype)
            if even_rows > 0:
                gx_c[:, :-1, :] = (
                    gx_odd[:, 0:even_rows:2, :] + gx_odd[:, 1:even_rows:2, :]
                )
            gx_c[:, -1, :] = gx_odd[:, -1, :]
    else:
        gx_c = np.empty((layers, coarse_rows, 0), dtype=dtype)

    if coarse_rows > 1:
        gy_odd = operator.gy[:, 1::2, :]
        if cols == even_cols:
            gy_c = gy_odd[:, :, 0::2] + gy_odd[:, :, 1::2]
        else:
            gy_c = np.empty((layers, coarse_rows - 1, coarse_cols), dtype=dtype)
            if even_cols > 0:
                gy_c[:, :, :-1] = (
                    gy_odd[:, :, 0:even_cols:2] + gy_odd[:, :, 1:even_cols:2]
                )
            gy_c[:, :, -1] = gy_odd[:, :, -1]
    else:
        gy_c = np.empty((layers, 0, coarse_cols), dtype=dtype)

    if layers > 1 and operator.gz.size:
        gz_c = _restrict_2x2(operator.gz, dtype=dtype)
    else:
        gz_c = np.empty((0, coarse_rows, coarse_cols), dtype=dtype)

    boundary_c = _restrict_2x2(
        operator.boundary_diag.reshape(layers, rows, cols),
        dtype=dtype,
    ).reshape(-1)
    diagonal_view = np.array(
        boundary_c.reshape(layers, coarse_rows, coarse_cols),
        dtype=dtype,
        copy=True,
    )
    if coarse_cols > 1:
        diagonal_view[:, :, :-1] += gx_c
        diagonal_view[:, :, 1:] += gx_c
    if coarse_rows > 1:
        diagonal_view[:, :-1, :] += gy_c
        diagonal_view[:, 1:, :] += gy_c
    if layers > 1 and gz_c.size:
        diagonal_view[:-1, :, :] += gz_c
        diagonal_view[1:, :, :] += gz_c

    return StructuredThermalOperator(
        layer_count=layers,
        rows=coarse_rows,
        cols=coarse_cols,
        gx=np.ascontiguousarray(gx_c, dtype=dtype),
        gy=np.ascontiguousarray(gy_c, dtype=dtype),
        gz=np.ascontiguousarray(gz_c, dtype=dtype),
        boundary_diag=np.ascontiguousarray(boundary_c, dtype=dtype),
        stiffness_diag=np.ascontiguousarray(diagonal_view.reshape(-1), dtype=dtype),
    )


def _assemble_structured_csc(
    operator: "StructuredThermalOperator",
    scaled_capacity: np.ndarray,
) -> sp.csc_matrix:
    """Assemble ``K + diag(scaled_capacity)`` on a coarse structured grid."""
    layers = operator.layer_count
    rows = operator.rows
    cols = operator.cols
    plane_cells = rows * cols
    node_count = layers * plane_cells
    diagonal = operator.stiffness_diag.astype(np.float64, copy=False) + np.asarray(
        scaled_capacity, dtype=np.float64
    )

    row_parts = [np.arange(node_count, dtype=np.int64)]
    col_parts = [np.arange(node_count, dtype=np.int64)]
    data_parts = [diagonal]

    if cols > 1:
        layer_base = np.arange(layers, dtype=np.int64)[:, None, None] * plane_cells
        row_base = np.arange(rows, dtype=np.int64)[None, :, None] * cols
        col_base = np.arange(cols - 1, dtype=np.int64)[None, None, :]
        left = (layer_base + row_base + col_base).reshape(-1)
        right = left + 1
        conductance = -operator.gx.astype(np.float64, copy=False).reshape(-1)
        row_parts.extend((left, right))
        col_parts.extend((right, left))
        data_parts.extend((conductance, conductance))

    if rows > 1:
        layer_base = np.arange(layers, dtype=np.int64)[:, None, None] * plane_cells
        row_base = np.arange(rows - 1, dtype=np.int64)[None, :, None] * cols
        col_base = np.arange(cols, dtype=np.int64)[None, None, :]
        upper = (layer_base + row_base + col_base).reshape(-1)
        lower = upper + cols
        conductance = -operator.gy.astype(np.float64, copy=False).reshape(-1)
        row_parts.extend((upper, lower))
        col_parts.extend((lower, upper))
        data_parts.extend((conductance, conductance))

    if layers > 1 and operator.gz.size:
        plane_idx = np.arange(plane_cells, dtype=np.int64)[None, :]
        top = (
            np.arange(layers - 1, dtype=np.int64)[:, None] * plane_cells + plane_idx
        ).reshape(-1)
        bottom = top + plane_cells
        conductance = -operator.gz.astype(np.float64, copy=False).reshape(-1)
        row_parts.extend((top, bottom))
        col_parts.extend((bottom, top))
        data_parts.extend((conductance, conductance))

    return sp.coo_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(row_parts), np.concatenate(col_parts)),
        ),
        shape=(node_count, node_count),
        dtype=np.float64,
    ).tocsc()


def _apply_with_diagonal_f32_into(
    layer_count: int,
    rows: int,
    cols: int,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    diag_mass: np.ndarray,
    vec: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """In-place float32 structured matvec into ``out``."""
    values = vec.reshape(layer_count, rows, cols)
    res = out.reshape(layer_count, rows, cols)
    np.multiply(diag_mass.reshape(values.shape), values, out=res)
    if cols > 1:
        diff = values[:, :, :-1] - values[:, :, 1:]
        diff *= gx
        res[:, :, :-1] += diff
        res[:, :, 1:] -= diff
    if rows > 1:
        diff = values[:, :-1, :] - values[:, 1:, :]
        diff *= gy
        res[:, :-1, :] += diff
        res[:, 1:, :] -= diff
    if layer_count > 1 and gz.size:
        diff = values[:-1, :, :] - values[1:, :, :]
        diff *= gz
        res[:-1, :, :] += diff
        res[1:, :, :] -= diff
    return out


def _smooth_f32(
    layer_count: int,
    rows: int,
    cols: int,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    diag_mass: np.ndarray,
    omega_inv: np.ndarray,
    rhs: np.ndarray,
    x: np.ndarray,
    scratch: np.ndarray,
    iterations: int,
    zero_init: bool,
) -> np.ndarray:
    """Run float32 damped Jacobi smoothing in-place on ``x``."""
    if damped_jacobi_f32_native is not None:
        native_out = damped_jacobi_f32_native(
            layer_count,
            rows,
            cols,
            gx,
            gy,
            gz,
            diag_mass,
            omega_inv,
            rhs,
            x,
            iterations,
            zero_init=zero_init,
            scratch=scratch,
        )
        if native_out is not None:
            if native_out is not x:
                np.copyto(x, native_out)
            return x

    if zero_init:
        np.multiply(omega_inv, rhs, out=x)
        remaining = max(0, iterations - 1)
    else:
        remaining = iterations
    for _ in range(remaining):
        _apply_with_diagonal_f32_into(
            layer_count, rows, cols, gx, gy, gz, diag_mass, x, scratch
        )
        np.subtract(rhs, scratch, out=scratch)
        scratch *= omega_inv
        x += scratch
    return x


def _residual_restrict_2x2_f32(
    layer_count: int,
    rows: int,
    cols: int,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    diag_mass: np.ndarray,
    rhs: np.ndarray,
    x: np.ndarray,
    scratch: np.ndarray,
    coarse_rhs: np.ndarray,
) -> np.ndarray:
    """Compute float32 residual ``rhs - A*x`` and 2x2-restrict into ``coarse_rhs``."""
    if residual_restrict_2x2_f32_native is not None:
        native_out = residual_restrict_2x2_f32_native(
            layer_count,
            rows,
            cols,
            gx,
            gy,
            gz,
            diag_mass,
            rhs,
            x,
            out=coarse_rhs,
        )
        if native_out is not None:
            if native_out is not coarse_rhs:
                np.copyto(coarse_rhs, native_out)
            return coarse_rhs

    _apply_with_diagonal_f32_into(
        layer_count, rows, cols, gx, gy, gz, diag_mass, x, scratch
    )
    np.subtract(rhs, scratch, out=scratch)
    _restrict_2x2_into(
        scratch.reshape(layer_count, rows, cols),
        coarse_rhs.reshape(layer_count, (rows + 1) // 2, (cols + 1) // 2),
    )
    return coarse_rhs


def _prolong_add_2x2_f32(
    layer_count: int,
    rows: int,
    cols: int,
    coarse_x: np.ndarray,
    fine_x: np.ndarray,
) -> None:
    """In-place add float32 piecewise-constant 2x2 prolongation into ``fine_x``."""
    if prolong_add_2x2_f32_native is not None and prolong_add_2x2_f32_native(
        layer_count, rows, cols, coarse_x, fine_x
    ):
        return
    coarse_rows = (rows + 1) // 2
    coarse_cols = (cols + 1) // 2
    coarse_view = coarse_x.reshape(layer_count, coarse_rows, coarse_cols)
    fine_view = fine_x.reshape(layer_count, rows, cols)
    fine_view[:, 0::2, 0::2] += coarse_view
    if cols > 1:
        fine_view[:, 0::2, 1::2] += coarse_view[:, :, : cols // 2]
    if rows > 1:
        fine_view[:, 1::2, 0::2] += coarse_view[:, : rows // 2, :]
    if rows > 1 and cols > 1:
        fine_view[:, 1::2, 1::2] += coarse_view[:, : rows // 2, : cols // 2]


def _build_coarse_solver(
    coarsest_op: "StructuredThermalOperator",
    scaled_capacity: np.ndarray,
):
    """Assemble and factorize the coarsest float64 CSC system."""
    scaled_cap_f64 = np.asarray(scaled_capacity, dtype=np.float64)
    coarse_matrix = _assemble_structured_csc(coarsest_op, scaled_cap_f64)
    try:
        return spla.factorized(coarse_matrix)
    except Exception:
        coarse_diag = (
            coarsest_op.stiffness_diag.astype(np.float64, copy=False) + scaled_cap_f64
        )
        coarse_inv = np.divide(
            1.0,
            coarse_diag,
            out=np.zeros_like(coarse_diag),
            where=np.abs(coarse_diag) > 1e-30,
        )
        return lambda rhs, _inv=coarse_inv: _inv * rhs


class _BaseGeometricMultigrid:
    """Shared float64 <-> float32 V-cycle entrypoints for multigrid preconditioners."""

    def _prepare_rhs(self, rhs: np.ndarray) -> np.ndarray:
        rhs_arr = _as_1d_contiguous_f64(rhs)
        if not (
            HAS_NATIVE_PCG
            and f64_to_f32_native is not None
            and f64_to_f32_native(rhs_arr, self._rhs_bufs[0])
        ):
            np.copyto(self._rhs_bufs[0], rhs_arr, casting="unsafe")
        return rhs_arr

    def solve_into(self, rhs: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Apply one symmetric V-cycle into preallocated float64 buffer ``out``."""
        self._prepare_rhs(rhs)
        self._cycle(0)
        np.copyto(out, self._x_bufs[0])
        return out

    def solve_into_and_dot(self, rhs_f64: np.ndarray, out_f64: np.ndarray) -> float:
        """Apply one V-cycle into ``out_f64`` and return ``dot(rhs_f64, out_f64)``."""
        rhs_arr = self._prepare_rhs(rhs_f64)
        self._cycle(0)
        x0_f32 = self._x_bufs[0]
        if HAS_NATIVE_PCG and f32_to_f64_dot_native is not None:
            dot_val = f32_to_f64_dot_native(x0_f32, rhs_arr, out_f64)
            if dot_val is not None:
                return float(dot_val)
        np.copyto(out_f64, x0_f32)
        return _dot_f64(rhs_arr, out_f64.reshape(-1))

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Apply one symmetric V-cycle and return a fresh float64 array."""
        out = np.empty(self._rhs_bufs[0].size, dtype=np.float64)
        return self.solve_into(rhs, out)


class _StructuredGeometricMultigrid(_BaseGeometricMultigrid):
    """Mixed-precision float32 V-cycle geometric multigrid preconditioner."""

    def __init__(
        self,
        operators: List["StructuredThermalOperator"],
        capacity: np.ndarray,
        derivative_scale: float,
        omega: float = 0.72,
        smooth_steps: int = 2,
    ):
        self.operators = operators
        self.omega = float(omega)
        self.smooth_steps = int(smooth_steps)
        self.gxs: List[np.ndarray] = []
        self.gys: List[np.ndarray] = []
        self.gzs: List[np.ndarray] = []
        self.diag_masses: List[np.ndarray] = []
        self.omega_inv_diags: List[np.ndarray] = []

        scaled_cap_f64 = np.ascontiguousarray(
            float(derivative_scale) * capacity, dtype=np.float64
        ).reshape(-1)
        scaled_cap_f32: Optional[np.ndarray] = None

        for level_idx, level_op in enumerate(self.operators):
            if level_idx == 0:
                diag_mass_f64 = level_op.boundary_diag + scaled_cap_f64
                diagonal_f64 = level_op.stiffness_diag + scaled_cap_f64
                inv_diag = np.divide(
                    1.0,
                    diagonal_f64,
                    out=np.zeros_like(diagonal_f64),
                    where=np.abs(diagonal_f64) > 1e-30,
                )
                diag_mass = np.ascontiguousarray(diag_mass_f64, dtype=np.float32)
                omega_inv = np.ascontiguousarray(
                    self.omega * inv_diag, dtype=np.float32
                )
                scaled_cap_f32 = scaled_cap_f64.astype(np.float32)
            else:
                prev_op = self.operators[level_idx - 1]
                scaled_cap_f32 = np.ascontiguousarray(
                    _restrict_2x2(
                        scaled_cap_f32.reshape(
                            prev_op.layer_count, prev_op.rows, prev_op.cols
                        ),
                        dtype=np.float32,
                    ).reshape(-1),
                    dtype=np.float32,
                )
                diag_mass = np.ascontiguousarray(
                    level_op.boundary_diag.astype(np.float32, copy=False)
                    + scaled_cap_f32,
                    dtype=np.float32,
                )
                diagonal = (
                    level_op.stiffness_diag.astype(np.float32, copy=False)
                    + scaled_cap_f32
                )
                inv_diag = np.divide(
                    np.float32(1.0),
                    diagonal,
                    out=np.zeros_like(diagonal),
                    where=np.abs(diagonal) > np.float32(1e-30),
                )
                omega_inv = np.ascontiguousarray(
                    np.float32(self.omega) * inv_diag, dtype=np.float32
                )

            self.gxs.append(np.ascontiguousarray(level_op.gx, dtype=np.float32))
            self.gys.append(np.ascontiguousarray(level_op.gy, dtype=np.float32))
            self.gzs.append(np.ascontiguousarray(level_op.gz, dtype=np.float32))
            self.diag_masses.append(diag_mass)
            self.omega_inv_diags.append(omega_inv)

        coarsest_cap = scaled_cap_f64 if len(self.operators) == 1 else scaled_cap_f32
        self.coarse_solver = _build_coarse_solver(self.operators[-1], coarsest_cap)

        self._rhs_bufs = [
            np.empty(op.shape[0], dtype=np.float32) for op in self.operators
        ]
        self._x_bufs = [
            np.empty(op.shape[0], dtype=np.float32) for op in self.operators
        ]
        self._scratch_bufs = [
            np.empty(op.shape[0], dtype=np.float32) for op in self.operators[:-1]
        ]

    def _cycle(self, level: int) -> None:
        if level == len(self.operators) - 1:
            sol = self.coarse_solver(
                self._rhs_bufs[level].astype(np.float64, copy=False)
            )
            np.copyto(self._x_bufs[level], np.asarray(sol, dtype=np.float64).reshape(-1), casting="unsafe")
            return

        op = self.operators[level]
        gx = self.gxs[level]
        gy = self.gys[level]
        gz = self.gzs[level]
        diag_mass = self.diag_masses[level]
        omega_inv = self.omega_inv_diags[level]
        rhs = self._rhs_bufs[level]
        x = self._x_bufs[level]
        scratch = self._scratch_bufs[level]

        _smooth_f32(
            op.layer_count,
            op.rows,
            op.cols,
            gx,
            gy,
            gz,
            diag_mass,
            omega_inv,
            rhs,
            x,
            scratch,
            self.smooth_steps,
            zero_init=True,
        )
        _residual_restrict_2x2_f32(
            op.layer_count,
            op.rows,
            op.cols,
            gx,
            gy,
            gz,
            diag_mass,
            rhs,
            x,
            scratch,
            self._rhs_bufs[level + 1],
        )
        self._cycle(level + 1)
        _prolong_add_2x2_f32(
            op.layer_count,
            op.rows,
            op.cols,
            self._x_bufs[level + 1],
            x,
        )
        _smooth_f32(
            op.layer_count,
            op.rows,
            op.cols,
            gx,
            gy,
            gz,
            diag_mass,
            omega_inv,
            rhs,
            x,
            scratch,
            self.smooth_steps,
            zero_init=False,
        )


def _capacity_cache_key(capacity: Optional[np.ndarray]):
    """Build a lightweight identity+shape+endpoint key without copying capacity."""
    if capacity is None:
        return None
    return (
        id(capacity),
        capacity.size,
        float(capacity[0]) if capacity.size else 0.0,
        float(capacity[-1]) if capacity.size else 0.0,
    )


@dataclass
class StructuredThermalOperator:
    """Matrix-free thermal operator for the regular multilayer grid."""

    layer_count: int
    rows: int
    cols: int
    gx: np.ndarray
    gy: np.ndarray
    gz: np.ndarray
    boundary_diag: np.ndarray
    stiffness_diag: np.ndarray

    def __post_init__(self):
        self._coarse_operators = None
        self._preconditioner_cache = {}

    @property
    def shape(self):
        """Return the equivalent square matrix shape."""
        node_count = self.layer_count * self.rows * self.cols
        return node_count, node_count

    @property
    def nnz_estimate(self):
        """Return the non-zero count of the equivalent assembled matrix."""
        edge_count = self.gx.size + self.gy.size + self.gz.size
        return int(self.shape[0] + 2 * edge_count)

    def clear_caches(self):
        """Release cached coarse operator hierarchies and preconditioners."""
        self._preconditioner_cache.clear()
        self._coarse_operators = None

    def _apply_with_diagonal(self, diag_mass: np.ndarray, vector) -> np.ndarray:
        """Apply conduction fluxes plus an arbitrary diagonal mass term."""
        if (
            isinstance(vector, np.ndarray)
            and vector.dtype == np.float64
            and vector.ndim == 1
            and vector.flags.c_contiguous
        ):
            flat_vector = vector
        else:
            flat_vector = np.ascontiguousarray(vector, dtype=np.float64).reshape(-1)

        if HAS_NATIVE_CORE:
            native_result = apply_structured_native(
                self.layer_count,
                self.rows,
                self.cols,
                self.gx,
                self.gy,
                self.gz,
                diag_mass,
                flat_vector,
            )
            if native_result is not None:
                return native_result

        values = flat_vector.reshape(self.layer_count, self.rows, self.cols)
        result = diag_mass.reshape(values.shape) * values

        if self.cols > 1:
            difference = values[:, :, :-1] - values[:, :, 1:]
            flux = self.gx * difference
            result[:, :, :-1] += flux
            result[:, :, 1:] -= flux
        if self.rows > 1:
            difference = values[:, :-1, :] - values[:, 1:, :]
            flux = self.gy * difference
            result[:, :-1, :] += flux
            result[:, 1:, :] -= flux
        if self.layer_count > 1 and self.gz.size:
            difference = values[:-1, :, :] - values[1:, :, :]
            flux = self.gz * difference
            result[:-1, :, :] += flux
            result[1:, :, :] -= flux
        return result.reshape(-1)

    def dot(self, vector):
        """Apply the stiffness operator without assembling a sparse matrix."""
        return self._apply_with_diagonal(self.boundary_diag, vector)

    def _get_multigrid_operators(
        self, coarse_max_nodes: int = 1024, max_levels: int = 10
    ) -> List["StructuredThermalOperator"]:
        """Build and cache the float32 coarsened stiffness operator hierarchy."""
        if self._coarse_operators is not None:
            return self._coarse_operators
        levels = [self]
        current = self
        for level_idx in range(max_levels - 1):
            if level_idx > 0 and current.shape[0] <= coarse_max_nodes:
                break
            if current.rows <= 2 and current.cols <= 2:
                break
            current = _coarsen_structured_operator(current, dtype=np.float32)
            levels.append(current)
        self._coarse_operators = levels
        return levels

    def implicit_linear_operator(self, capacity, derivative_scale):
        """Return ``K + derivative_scale * C`` as a SciPy LinearOperator."""
        if (
            isinstance(capacity, np.ndarray)
            and capacity.dtype == np.float64
            and capacity.ndim == 1
            and capacity.flags.c_contiguous
        ):
            cap_arr = capacity
        else:
            cap_arr = np.ascontiguousarray(capacity, dtype=np.float64).reshape(-1)
        scale = float(derivative_scale)
        scaled_capacity = scale * cap_arr
        diag_mass = np.ascontiguousarray(
            self.boundary_diag + scaled_capacity, dtype=np.float64
        )
        shape = self.shape

        def matvec(vector):
            return self._apply_with_diagonal(diag_mass, vector)

        operator = spla.LinearOperator(shape, matvec=matvec, dtype=np.float64)

        if self.rows < 8 or self.cols < 8:
            diagonal = self.stiffness_diag + scaled_capacity
            inverse_diagonal = np.divide(
                1.0,
                diagonal,
                out=np.zeros_like(diagonal),
                where=np.abs(diagonal) > 1e-30,
            )
            preconditioner = spla.LinearOperator(
                shape,
                matvec=lambda vector: inverse_diagonal * vector,
                dtype=np.float64,
            )
            preconditioner._solve_into = lambda v, out: np.multiply(
                inverse_diagonal, v, out=out
            )
            return operator, preconditioner

        cap_key = _capacity_cache_key(cap_arr)
        cached = self._preconditioner_cache.get(scale)
        if cached is not None and cached[0] == cap_key:
            multigrid = cached[1]
        else:
            multigrid = _StructuredGeometricMultigrid(
                self._get_multigrid_operators(),
                cap_arr,
                scale,
            )
            self._preconditioner_cache.clear()
            self._preconditioner_cache[scale] = (cap_key, multigrid)

        preconditioner = spla.LinearOperator(
            shape,
            matvec=multigrid.solve,
            dtype=np.float64,
        )
        preconditioner._solve_into = multigrid.solve_into
        preconditioner._solve_into_and_dot = multigrid.solve_into_and_dot
        return operator, preconditioner


def _compute_layer_luts(
    layer_count: int,
    t_cu: np.ndarray,
    t_fr4_eff: np.ndarray,
    k_cu: float,
    k_fr4: float,
    dx: float,
    dy: float,
    gap_m: List[float],
    rho_cu: float = 8960.0,
    cp_cu: float = 385.0,
    rho_fr4: float = 1850.0,
    cp_fr4: float = 1100.0,
    pad_cap_areal: float = 0.0,
):
    """Precompute compact 3-value edge conductance and 2-value capacity LUTs."""
    eps = 1e-12
    pixel_area = float(dx) * float(dy)
    k0 = float(k_fr4)
    k1 = float(k_cu)
    gx_lut = np.empty((layer_count, 3), dtype=np.float64)
    gy_lut = np.empty((layer_count, 3), dtype=np.float64)
    cap_lut = np.empty((layer_count, 2), dtype=np.float64)

    k_pair = (
        2.0 * k0 * k0 / (k0 + k0 + eps),
        2.0 * k0 * k1 / (k0 + k1 + eps),
        2.0 * k1 * k1 / (k1 + k1 + eps),
    )

    for layer_idx in range(layer_count):
        t0 = float(t_fr4_eff[layer_idx])
        t1 = float(t_cu[layer_idx])
        t_pair = (t0, 0.5 * (t0 + t1), t1)
        for m in range(3):
            gx_lut[layer_idx, m] = k_pair[m] * (t_pair[m] * dy) / dx
            gy_lut[layer_idx, m] = k_pair[m] * (t_pair[m] * dx) / dy
        cap_fr4 = float(rho_fr4) * float(cp_fr4) * pixel_area * t0
        cap_cu = float(rho_cu) * float(cp_cu) * pixel_area * t1
        cap_lut[layer_idx, 0] = cap_fr4
        cap_lut[layer_idx, 1] = cap_cu + cap_fr4

    if layer_count > 1 and gap_m:
        gz_scale = np.empty(layer_count - 1, dtype=np.float64)
        for layer_idx in range(layer_count - 1):
            gap_value = max(float(gap_m[layer_idx]), 1e-6)
            gz_scale[layer_idx] = k0 * pixel_area / gap_value
    else:
        gz_scale = np.empty(0, dtype=np.float64)

    pad_cap_cell = float(pad_cap_areal) * pixel_area
    return gx_lut, gy_lut, gz_scale, cap_lut, pad_cap_cell


def _coarsen_compact_to_f32_numpy(op: "CompactStructuredThermalOperator"):
    """Pure-NumPy layer-by-layer Level-0 -> Level-1 Galerkin coarsening in float32."""
    layers = op.layer_count
    rows = op.rows
    cols = op.cols
    c_rows = (rows + 1) // 2
    c_cols = (cols + 1) // 2
    even_rows = (rows // 2) * 2
    even_cols = (cols // 2) * 2
    hs_bool = op.heatsink_mask.astype(bool, copy=False)

    gx_c = np.empty((layers, c_rows, max(c_cols - 1, 0)), dtype=np.float32)
    gy_c = np.empty((layers, max(c_rows - 1, 0), c_cols), dtype=np.float32)
    gz_c = np.empty((max(layers - 1, 0), c_rows, c_cols), dtype=np.float32)
    boundary_c = np.empty((layers, c_rows, c_cols), dtype=np.float32)
    cap_c = np.empty((layers, c_rows, c_cols), dtype=np.float32)

    for layer_idx in range(layers):
        mask_l = op.copper_mask[layer_idx]
        if c_cols > 1:
            idx_x = (mask_l[:, :-1] + mask_l[:, 1:])[:, 1::2]
            gx_odd = op.gx_lut[layer_idx][idx_x]
            if rows == even_rows:
                gx_c[layer_idx] = (gx_odd[0::2, :] + gx_odd[1::2, :]).astype(np.float32)
            else:
                if even_rows > 0:
                    gx_c[layer_idx, :-1, :] = (
                        gx_odd[0:even_rows:2, :] + gx_odd[1:even_rows:2, :]
                    ).astype(np.float32)
                gx_c[layer_idx, -1, :] = gx_odd[-1, :].astype(np.float32)

        if c_rows > 1:
            idx_y = (mask_l[:-1, :] + mask_l[1:, :])[1::2, :]
            gy_odd = op.gy_lut[layer_idx][idx_y]
            if cols == even_cols:
                gy_c[layer_idx] = (gy_odd[:, 0::2] + gy_odd[:, 1::2]).astype(np.float32)
            else:
                if even_cols > 0:
                    gy_c[layer_idx, :, :-1] = (
                        gy_odd[:, 0:even_cols:2] + gy_odd[:, 1:even_cols:2]
                    ).astype(np.float32)
                gy_c[layer_idx, :, -1] = gy_odd[:, -1].astype(np.float32)

        cap_l = np.where(mask_l, op.cap_lut[layer_idx, 1], op.cap_lut[layer_idx, 0])
        if layer_idx == layers - 1 and op.pad_cap_cell != 0.0:
            cap_l[hs_bool] += op.pad_cap_cell
        _restrict_2x2_into(cap_l[None, :, :], cap_c[layer_idx : layer_idx + 1])

        b_l = np.zeros((rows, cols), dtype=np.float64)
        if layer_idx == 0:
            b_l += op.top_diag
        if layer_idx == layers - 1:
            b_l += np.where(hs_bool, op.bot_pad_diag, op.bot_air_diag)
        _restrict_2x2_into(b_l[None, :, :], boundary_c[layer_idx : layer_idx + 1])

    if layers > 1 and op.gz_scale.size:
        via_f64 = op.via_enh.astype(np.float64)
        for layer_idx in range(layers - 1):
            gz_l = op.gz_scale[layer_idx] * via_f64
            _restrict_2x2_into(gz_l[None, :, :], gz_c[layer_idx : layer_idx + 1])

    stiffness_view = np.array(boundary_c, dtype=np.float64, copy=True)
    if c_cols > 1:
        stiffness_view[:, :, :-1] += gx_c
        stiffness_view[:, :, 1:] += gx_c
    if c_rows > 1:
        stiffness_view[:, :-1, :] += gy_c
        stiffness_view[:, 1:, :] += gy_c
    if layers > 1 and gz_c.size:
        stiffness_view[:-1, :, :] += gz_c
        stiffness_view[1:, :, :] += gz_c

    return (
        np.ascontiguousarray(gx_c, dtype=np.float32),
        np.ascontiguousarray(gy_c, dtype=np.float32),
        np.ascontiguousarray(gz_c, dtype=np.float32),
        np.ascontiguousarray(boundary_c.reshape(-1), dtype=np.float32),
        np.ascontiguousarray(stiffness_view.reshape(-1), dtype=np.float32),
        np.ascontiguousarray(cap_c.reshape(-1), dtype=np.float32),
    )


def _compact_layer_full_diag(
    op: "CompactStructuredThermalOperator",
    layer_idx: int,
    scale: float,
    hs_bool: np.ndarray,
) -> np.ndarray:
    """Compute Level-0 ``K_ii + scale * C_i`` for a single 2D layer."""
    mask_l = op.copper_mask[layer_idx]
    diag = np.zeros((op.rows, op.cols), dtype=np.float64)
    if scale != 0.0:
        cap_l = np.where(mask_l, op.cap_lut[layer_idx, 1], op.cap_lut[layer_idx, 0])
        if layer_idx == op.layer_count - 1 and op.pad_cap_cell != 0.0:
            cap_l[hs_bool] += op.pad_cap_cell
        diag += scale * cap_l
    if layer_idx == 0:
        diag += op.top_diag
    if layer_idx == op.layer_count - 1:
        diag += np.where(hs_bool, op.bot_pad_diag, op.bot_air_diag)
    if op.cols > 1:
        gx_l = op.gx_lut[layer_idx][mask_l[:, :-1] + mask_l[:, 1:]]
        diag[:, :-1] += gx_l
        diag[:, 1:] += gx_l
    if op.rows > 1:
        gy_l = op.gy_lut[layer_idx][mask_l[:-1, :] + mask_l[1:, :]]
        diag[:-1, :] += gy_l
        diag[1:, :] += gy_l
    if op.layer_count > 1 and op.gz_scale.size:
        via_f64 = op.via_enh.astype(np.float64)
        if layer_idx > 0:
            diag += op.gz_scale[layer_idx - 1] * via_f64
        if layer_idx < op.layer_count - 1:
            diag += op.gz_scale[layer_idx] * via_f64
    return diag


def _compact_apply_f32_into(
    op: "CompactStructuredThermalOperator",
    scale: float,
    x_f32: np.ndarray,
    out_f32: np.ndarray,
) -> np.ndarray:
    """Layer-by-layer pure-NumPy Level-0 ``(K + scale*C) * x`` in float32."""
    layers, rows, cols = op.layer_count, op.rows, op.cols
    x_view = x_f32.reshape(layers, rows, cols)
    out_view = out_f32.reshape(layers, rows, cols)
    hs_bool = op.heatsink_mask.astype(bool, copy=False)
    via_f32 = op.via_enh if (layers > 1 and op.gz_scale.size) else None

    for l in range(layers):
        mask_l = op.copper_mask[l]
        x_l = x_view[l]
        r_l = out_view[l]
        diag_2d = np.zeros((rows, cols), dtype=np.float64)
        if scale != 0.0:
            c_l = np.where(mask_l, op.cap_lut[l, 1], op.cap_lut[l, 0])
            if l == layers - 1 and op.pad_cap_cell != 0.0:
                c_l[hs_bool] += op.pad_cap_cell
            diag_2d += scale * c_l
        if l == 0:
            diag_2d += op.top_diag
        if l == layers - 1:
            diag_2d += np.where(hs_bool, op.bot_pad_diag, op.bot_air_diag)
        np.multiply(diag_2d.astype(np.float32), x_l, out=r_l)

        if cols > 1:
            gx_l = op.gx_lut[l].astype(np.float32)[mask_l[:, :-1] + mask_l[:, 1:]]
            flux_x = gx_l * (x_l[:, :-1] - x_l[:, 1:])
            r_l[:, :-1] += flux_x
            r_l[:, 1:] -= flux_x
        if rows > 1:
            gy_l = op.gy_lut[l].astype(np.float32)[mask_l[:-1, :] + mask_l[1:, :]]
            flux_y = gy_l * (x_l[:-1, :] - x_l[1:, :])
            r_l[:-1, :] += flux_y
            r_l[1:, :] -= flux_y

    if via_f32 is not None:
        for l in range(layers - 1):
            gz_l = np.float32(op.gz_scale[l]) * via_f32
            flux_z = gz_l * (x_view[l] - x_view[l + 1])
            out_view[l] += flux_z
            out_view[l + 1] -= flux_z
    return out_f32


class _CompactGeometricMultigrid(_BaseGeometricMultigrid):
    """Zero-Level-0-materialization mixed-precision V-cycle preconditioner."""

    def __init__(
        self,
        compact_op: "CompactStructuredThermalOperator",
        coarse_operators: List["StructuredThermalOperator"],
        level1_cap_f32: np.ndarray,
        derivative_scale: float,
        omega: float = 0.72,
        smooth_steps: int = 2,
    ):
        self.compact_op = compact_op
        self.coarse_operators = coarse_operators
        self.omega = float(omega)
        self.smooth_steps = int(smooth_steps)
        self.scale = float(derivative_scale)

        self.gxs: List[np.ndarray] = []
        self.gys: List[np.ndarray] = []
        self.gzs: List[np.ndarray] = []
        self.diag_masses: List[np.ndarray] = []
        self.omega_inv_diags: List[np.ndarray] = []

        scaled_cap_f32 = np.ascontiguousarray(
            np.float32(self.scale) * level1_cap_f32, dtype=np.float32
        ).reshape(-1)

        for idx, level_op in enumerate(self.coarse_operators):
            if idx > 0:
                prev_op = self.coarse_operators[idx - 1]
                scaled_cap_f32 = np.ascontiguousarray(
                    _restrict_2x2(
                        scaled_cap_f32.reshape(
                            prev_op.layer_count, prev_op.rows, prev_op.cols
                        ),
                        dtype=np.float32,
                    ).reshape(-1),
                    dtype=np.float32,
                )
            diag_mass = np.ascontiguousarray(
                level_op.boundary_diag.astype(np.float32, copy=False) + scaled_cap_f32,
                dtype=np.float32,
            )
            diagonal = (
                level_op.stiffness_diag.astype(np.float32, copy=False) + scaled_cap_f32
            )
            inv_diag = np.divide(
                np.float32(1.0),
                diagonal,
                out=np.zeros_like(diagonal),
                where=np.abs(diagonal) > np.float32(1e-30),
            )
            self.gxs.append(np.ascontiguousarray(level_op.gx, dtype=np.float32))
            self.gys.append(np.ascontiguousarray(level_op.gy, dtype=np.float32))
            self.gzs.append(np.ascontiguousarray(level_op.gz, dtype=np.float32))
            self.diag_masses.append(diag_mass)
            self.omega_inv_diags.append(
                np.ascontiguousarray(np.float32(self.omega) * inv_diag, dtype=np.float32)
            )

        self.coarse_solver = _build_coarse_solver(
            self.coarse_operators[-1], scaled_cap_f32
        )

        # Level 0 + coarse levels (1..num_coarse)
        self._rhs_bufs = [
            np.empty(compact_op.shape[0], dtype=np.float32)
        ] + [np.empty(op.shape[0], dtype=np.float32) for op in self.coarse_operators]
        self._x_bufs = [
            np.empty(compact_op.shape[0], dtype=np.float32)
        ] + [np.empty(op.shape[0], dtype=np.float32) for op in self.coarse_operators]
        self._scratch_bufs = [
            np.empty(compact_op.shape[0], dtype=np.float32)
        ] + [np.empty(op.shape[0], dtype=np.float32) for op in self.coarse_operators[:-1]]

    def _smooth_level0(self, zero_init: bool) -> None:
        op = self.compact_op
        rhs = self._rhs_bufs[0]
        x = self._x_bufs[0]
        scratch = self._scratch_bufs[0]
        if damped_jacobi_compact_f32_native is not None:
            native_out = damped_jacobi_compact_f32_native(
                op.layer_count,
                op.rows,
                op.cols,
                op.copper_mask,
                op.via_enh,
                op.heatsink_mask,
                op.gx_lut,
                op.gy_lut,
                op.gz_scale,
                op.cap_lut,
                op.pad_cap_cell,
                op.top_diag,
                op.bot_air_diag,
                op.bot_pad_diag,
                self.scale,
                self.omega,
                rhs,
                x,
                self.smooth_steps,
                zero_init=zero_init,
                scratch=scratch,
            )
            if native_out is not None:
                if native_out is not x:
                    np.copyto(x, native_out)
                return

        hs_bool = op.heatsink_mask.astype(bool, copy=False)
        x_view = x.reshape(op.layer_count, op.rows, op.cols)
        rhs_view = rhs.reshape(op.layer_count, op.rows, op.cols)
        scratch_view = scratch.reshape(op.layer_count, op.rows, op.cols)

        def apply_omega_inv(src_view, dst_view, add_to_dst: bool):
            for l in range(op.layer_count):
                full_diag = _compact_layer_full_diag(op, l, self.scale, hs_bool)
                inv_d = np.divide(
                    self.omega,
                    full_diag,
                    out=np.zeros_like(full_diag),
                    where=np.abs(full_diag) > 1e-30,
                ).astype(np.float32)
                if add_to_dst:
                    dst_view[l] += inv_d * src_view[l]
                else:
                    np.multiply(inv_d, src_view[l], out=dst_view[l])

        if zero_init:
            apply_omega_inv(rhs_view, x_view, add_to_dst=False)
            remaining = max(0, self.smooth_steps - 1)
        else:
            remaining = self.smooth_steps

        for _ in range(remaining):
            _compact_apply_f32_into(op, self.scale, x, scratch)
            np.subtract(rhs, scratch, out=scratch)
            apply_omega_inv(scratch_view, x_view, add_to_dst=True)

    def _residual_restrict_level0(self) -> None:
        op = self.compact_op
        rhs = self._rhs_bufs[0]
        x = self._x_bufs[0]
        scratch = self._scratch_bufs[0]
        coarse_rhs = self._rhs_bufs[1]
        if residual_restrict_2x2_compact_f32_native is not None:
            native_out = residual_restrict_2x2_compact_f32_native(
                op.layer_count,
                op.rows,
                op.cols,
                op.copper_mask,
                op.via_enh,
                op.heatsink_mask,
                op.gx_lut,
                op.gy_lut,
                op.gz_scale,
                op.cap_lut,
                op.pad_cap_cell,
                op.top_diag,
                op.bot_air_diag,
                op.bot_pad_diag,
                self.scale,
                rhs,
                x,
                out=coarse_rhs,
            )
            if native_out is not None:
                if native_out is not coarse_rhs:
                    np.copyto(coarse_rhs, native_out)
                return

        _compact_apply_f32_into(op, self.scale, x, scratch)
        np.subtract(rhs, scratch, out=scratch)
        _restrict_2x2_into(
            scratch.reshape(op.layer_count, op.rows, op.cols),
            coarse_rhs.reshape(
                op.layer_count, (op.rows + 1) // 2, (op.cols + 1) // 2
            ),
        )

    def _cycle(self, level: int) -> None:
        num_total_levels = 1 + len(self.coarse_operators)
        if level == num_total_levels - 1:
            sol = self.coarse_solver(
                self._rhs_bufs[level].astype(np.float64, copy=False)
            )
            np.copyto(
                self._x_bufs[level],
                np.asarray(sol, dtype=np.float64).reshape(-1),
                casting="unsafe",
            )
            return

        if level == 0:
            op = self.compact_op
            self._smooth_level0(zero_init=True)
            self._residual_restrict_level0()
            self._cycle(1)
            _prolong_add_2x2_f32(
                op.layer_count,
                op.rows,
                op.cols,
                self._x_bufs[1],
                self._x_bufs[0],
            )
            self._smooth_level0(zero_init=False)
            return

        c_idx = level - 1
        op = self.coarse_operators[c_idx]
        gx = self.gxs[c_idx]
        gy = self.gys[c_idx]
        gz = self.gzs[c_idx]
        diag_mass = self.diag_masses[c_idx]
        omega_inv = self.omega_inv_diags[c_idx]
        rhs = self._rhs_bufs[level]
        x = self._x_bufs[level]
        scratch = self._scratch_bufs[level]

        _smooth_f32(
            op.layer_count,
            op.rows,
            op.cols,
            gx,
            gy,
            gz,
            diag_mass,
            omega_inv,
            rhs,
            x,
            scratch,
            self.smooth_steps,
            zero_init=True,
        )
        _residual_restrict_2x2_f32(
            op.layer_count,
            op.rows,
            op.cols,
            gx,
            gy,
            gz,
            diag_mass,
            rhs,
            x,
            scratch,
            self._rhs_bufs[level + 1],
        )
        self._cycle(level + 1)
        _prolong_add_2x2_f32(
            op.layer_count,
            op.rows,
            op.cols,
            self._x_bufs[level + 1],
            x,
        )
        _smooth_f32(
            op.layer_count,
            op.rows,
            op.cols,
            gx,
            gy,
            gz,
            diag_mass,
            omega_inv,
            rhs,
            x,
            scratch,
            self.smooth_steps,
            zero_init=False,
        )


@dataclass
class CompactStructuredThermalOperator:
    """Ultra-compact matrix-free operator storing only masks, 2D via map, and LUTs."""

    layer_count: int
    rows: int
    cols: int
    copper_mask: np.ndarray
    via_enh: np.ndarray
    heatsink_mask: np.ndarray
    gx_lut: np.ndarray
    gy_lut: np.ndarray
    gz_scale: np.ndarray
    cap_lut: np.ndarray
    pad_cap_cell: float
    top_diag: float
    bot_air_diag: float
    bot_pad_diag: float
    top_rhs: float
    bot_air_rhs: float
    bot_pad_rhs: float
    top_ha: float
    bot_air_ha: float
    bot_pad_ha: float
    heatsink_mode_active: bool

    def __post_init__(self):
        self.copper_mask = np.ascontiguousarray(
            np.asarray(self.copper_mask, dtype=bool), dtype=np.uint8
        )
        self.via_enh = np.ascontiguousarray(self.via_enh, dtype=np.float32)
        self.heatsink_mask = np.ascontiguousarray(
            np.asarray(self.heatsink_mask, dtype=bool), dtype=np.uint8
        )
        self.gx_lut = np.ascontiguousarray(self.gx_lut, dtype=np.float64)
        self.gy_lut = np.ascontiguousarray(self.gy_lut, dtype=np.float64)
        self.gz_scale = np.ascontiguousarray(self.gz_scale, dtype=np.float64)
        self.cap_lut = np.ascontiguousarray(self.cap_lut, dtype=np.float64)
        self._coarse_operators = None
        self._preconditioner_cache = {}

    @property
    def shape(self):
        """Return the equivalent square matrix shape."""
        node_count = self.layer_count * self.rows * self.cols
        return node_count, node_count

    @property
    def nnz_estimate(self):
        """Return the non-zero count of the equivalent assembled matrix."""
        x_edges = self.layer_count * self.rows * max(self.cols - 1, 0)
        y_edges = self.layer_count * max(self.rows - 1, 0) * self.cols
        z_edges = (
            (self.layer_count - 1) * self.rows * self.cols
            if (self.layer_count > 1 and self.gz_scale.size)
            else 0
        )
        return int(self.shape[0] + 2 * (x_edges + y_edges + z_edges))

    def clear_caches(self):
        """Release cached coarse operator hierarchies and preconditioners."""
        self._preconditioner_cache.clear()
        self._coarse_operators = None

    def _apply_with_scale(
        self, scale: float, vector, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply ``(K + scale * C) * vector`` without materializing 3D conductance arrays."""
        flat_vector = _as_1d_contiguous_f64(vector)

        if out is None:
            out = np.empty(self.shape[0], dtype=np.float64)

        if HAS_NATIVE_COMPACT and apply_compact_native is not None:
            native_result = apply_compact_native(
                self.layer_count,
                self.rows,
                self.cols,
                self.copper_mask,
                self.via_enh,
                self.heatsink_mask,
                self.gx_lut,
                self.gy_lut,
                self.gz_scale,
                self.cap_lut,
                self.pad_cap_cell,
                self.top_diag,
                self.bot_air_diag,
                self.bot_pad_diag,
                float(scale),
                flat_vector,
                out=out,
            )
            if native_result is not None:
                if native_result is not out:
                    np.copyto(out, native_result)
                return out

        layers, rows, cols = self.layer_count, self.rows, self.cols
        values = flat_vector.reshape(layers, rows, cols)
        res_view = out.reshape(layers, rows, cols)
        hs_bool = self.heatsink_mask.astype(bool, copy=False)
        scale_f = float(scale)

        for l in range(layers):
            mask_l = self.copper_mask[l]
            v_l = values[l]
            r_l = res_view[l]
            if scale_f != 0.0:
                diag_2d = np.where(mask_l, self.cap_lut[l, 1], self.cap_lut[l, 0])
                if l == layers - 1 and self.pad_cap_cell != 0.0:
                    diag_2d[hs_bool] += self.pad_cap_cell
                diag_2d *= scale_f
            else:
                diag_2d = np.zeros((rows, cols), dtype=np.float64)
            if l == 0:
                diag_2d += self.top_diag
            if l == layers - 1:
                diag_2d += np.where(hs_bool, self.bot_pad_diag, self.bot_air_diag)
            np.multiply(diag_2d, v_l, out=r_l)

            if cols > 1:
                gx_l = self.gx_lut[l][mask_l[:, :-1] + mask_l[:, 1:]]
                flux_x = gx_l * (v_l[:, :-1] - v_l[:, 1:])
                r_l[:, :-1] += flux_x
                r_l[:, 1:] -= flux_x
            if rows > 1:
                gy_l = self.gy_lut[l][mask_l[:-1, :] + mask_l[1:, :]]
                flux_y = gy_l * (v_l[:-1, :] - v_l[1:, :])
                r_l[:-1, :] += flux_y
                r_l[1:, :] -= flux_y

        if layers > 1 and self.gz_scale.size:
            via_f64 = self.via_enh.astype(np.float64)
            for l in range(layers - 1):
                gz_l = self.gz_scale[l] * via_f64
                flux_z = gz_l * (values[l] - values[l + 1])
                res_view[l] += flux_z
                res_view[l + 1] -= flux_z
        return out

    def _apply_with_scale_and_dot(
        self, scale: float, vector, out: np.ndarray
    ) -> float:
        """Apply ``(K + scale * C) * vector`` into ``out`` and return ``dot(vector, out)``."""
        flat_vector = _as_1d_contiguous_f64(vector)
        if HAS_NATIVE_COMPACT and HAS_NATIVE_PCG and apply_compact_dot_native is not None:
            native_result = apply_compact_dot_native(
                self.layer_count,
                self.rows,
                self.cols,
                self.copper_mask,
                self.via_enh,
                self.heatsink_mask,
                self.gx_lut,
                self.gy_lut,
                self.gz_scale,
                self.cap_lut,
                self.pad_cap_cell,
                self.top_diag,
                self.bot_air_diag,
                self.bot_pad_diag,
                float(scale),
                flat_vector,
                out=out,
            )
            if native_result is not None:
                res_arr, dot_val = native_result
                if res_arr is not out:
                    np.copyto(out, res_arr)
                return float(dot_val)

        self._apply_with_scale(scale, flat_vector, out=out)
        return _dot_f64(flat_vector, out.reshape(-1))

    def dot(self, vector):
        """Apply the stiffness operator ``K * vector``."""
        return self._apply_with_scale(0.0, vector)

    def add_boundary_rhs(self, out: np.ndarray) -> np.ndarray:
        """Add boundary condition RHS ``b`` in-place (only touches layers 0 and L-1)."""
        plane_size = self.rows * self.cols
        out[:plane_size] += self.top_rhs
        if self.bot_air_rhs != 0.0 or self.bot_pad_rhs != 0.0:
            hs_bool = self.heatsink_mask.reshape(-1).astype(bool, copy=False)
            out[-plane_size:] += np.where(hs_bool, self.bot_pad_rhs, self.bot_air_rhs)
        return out

    def add_scaled_capacity(
        self, out: np.ndarray, vec: np.ndarray, scale: float
    ) -> np.ndarray:
        """Accumulate ``scale * C * vec`` into ``out`` layer-by-layer without a 3D C array."""
        scale_f = float(scale)
        if scale_f == 0.0:
            return out
        plane_size = self.rows * self.cols
        out_layers = out.reshape(self.layer_count, plane_size)
        vec_layers = np.asarray(vec, dtype=np.float64).reshape(
            self.layer_count, plane_size
        )
        hs_bool = self.heatsink_mask.reshape(-1).astype(bool, copy=False)
        for l in range(self.layer_count):
            c0 = self.cap_lut[l, 0] * scale_f
            c1 = self.cap_lut[l, 1] * scale_f
            cap_scaled = np.where(self.copper_mask[l].reshape(-1), c1, c0)
            if l == self.layer_count - 1 and self.pad_cap_cell != 0.0:
                cap_scaled[hs_bool] += self.pad_cap_cell * scale_f
            out_layers[l] += cap_scaled * vec_layers[l]
        return out

    def divide_by_capacity(
        self, vec: np.ndarray, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute ``vec / C`` layer-by-layer without allocating a 3D capacity array."""
        if out is None:
            out = np.empty(self.shape[0], dtype=np.float64)
        plane_size = self.rows * self.cols
        vec_layers = np.asarray(vec, dtype=np.float64).reshape(
            self.layer_count, plane_size
        )
        out_layers = out.reshape(self.layer_count, plane_size)
        hs_bool = self.heatsink_mask.reshape(-1).astype(bool, copy=False)
        for l in range(self.layer_count):
            cap_l = np.where(
                self.copper_mask[l].reshape(-1),
                self.cap_lut[l, 1],
                self.cap_lut[l, 0],
            )
            if l == self.layer_count - 1 and self.pad_cap_cell != 0.0:
                cap_l[hs_bool] += self.pad_cap_cell
            np.divide(vec_layers[l], cap_l, out=out_layers[l])
        return out

    def thermal_energy(self, T: np.ndarray, amb: float) -> float:
        """Compute ``sum(C * (T - amb))`` layer-by-layer without a 3D C array."""
        plane_size = self.rows * self.cols
        t_layers = np.asarray(T, dtype=np.float64).reshape(
            self.layer_count, plane_size
        )
        hs_bool = self.heatsink_mask.reshape(-1).astype(bool, copy=False)
        amb_f = float(amb)
        total = 0.0
        for l in range(self.layer_count):
            cap_l = np.where(
                self.copper_mask[l].reshape(-1),
                self.cap_lut[l, 1],
                self.cap_lut[l, 0],
            )
            if l == self.layer_count - 1 and self.pad_cap_cell != 0.0:
                cap_l[hs_bool] += self.pad_cap_cell
            total += float(np.dot(cap_l, t_layers[l] - amb_f))
        return total

    def convection_power(self, T: np.ndarray, amb: float) -> float:
        """Compute ``sum(hA * (T - amb))`` touching only top and bottom layers."""
        plane_size = self.rows * self.cols
        t_flat = np.asarray(T, dtype=np.float64).reshape(-1)
        amb_f = float(amb)
        total = self.top_ha * float(np.sum(t_flat[:plane_size] - amb_f))
        if self.bot_air_ha != 0.0 or self.bot_pad_ha != 0.0:
            hs_bool = self.heatsink_mask.reshape(-1).astype(bool, copy=False)
            ha_bot = np.where(hs_bool, self.bot_pad_ha, self.bot_air_ha)
            total += float(np.dot(ha_bot, t_flat[-plane_size:] - amb_f))
        return total

    def build_heatsink_model(
        self, settings: Dict, amb: float
    ) -> Optional[HeatsinkModel]:
        """Build a compact 2D bottom-layer HeatsinkModel when dynamic heatsink is active."""
        if not bool(settings.get("use_heatsink", True)):
            return None
        sink_mode = str(settings.get("pad_sink_mode", "fixed") or "fixed").strip().lower()
        if sink_mode != "heatsink":
            return None
        if not np.any(self.heatsink_mask):
            return None
        rth_val = settings.get("heatsink_rth_kw", 5.0)
        rth_kw = max(1e-6, float(5.0 if rth_val is None else rth_val))
        pad_conductance = (
            self.heatsink_mask.astype(np.float64).reshape(-1) * self.bot_pad_diag
        )
        return HeatsinkModel(
            pad_conductance=pad_conductance,
            rth_kw=rth_kw,
            capacity_j_k=compute_heatsink_capacity_j_k(settings),
            initial_temp_c=float(amb),
            bottom_layer_only=True,
        )

    def _get_coarse_operators(
        self, coarse_max_nodes: int = 1024, max_levels: int = 10
    ):
        """Build and cache float32 coarse operators (Levels >= 1) and Level-1 capacity."""
        if self._coarse_operators is not None:
            return self._coarse_operators

        coarsened = None
        if HAS_NATIVE_COMPACT and coarsen_compact_to_f32_native is not None:
            coarsened = coarsen_compact_to_f32_native(
                self.layer_count,
                self.rows,
                self.cols,
                self.copper_mask,
                self.via_enh,
                self.heatsink_mask,
                self.gx_lut,
                self.gy_lut,
                self.gz_scale,
                self.cap_lut,
                self.pad_cap_cell,
                self.top_diag,
                self.bot_air_diag,
                self.bot_pad_diag,
            )
        if coarsened is None:
            coarsened = _coarsen_compact_to_f32_numpy(self)

        gx_c, gy_c, gz_c, boundary_c, stiffness_diag_c, level1_cap_f32 = coarsened
        c_rows = (self.rows + 1) // 2
        c_cols = (self.cols + 1) // 2
        level1_op = StructuredThermalOperator(
            layer_count=self.layer_count,
            rows=c_rows,
            cols=c_cols,
            gx=np.ascontiguousarray(gx_c, dtype=np.float32),
            gy=np.ascontiguousarray(gy_c, dtype=np.float32),
            gz=np.ascontiguousarray(gz_c, dtype=np.float32),
            boundary_diag=np.ascontiguousarray(boundary_c, dtype=np.float32),
            stiffness_diag=np.ascontiguousarray(stiffness_diag_c, dtype=np.float32),
        )
        coarse_levels = [level1_op]
        current = level1_op
        for _ in range(1, max_levels - 1):
            if current.shape[0] <= coarse_max_nodes:
                break
            if current.rows <= 2 and current.cols <= 2:
                break
            current = _coarsen_structured_operator(current, dtype=np.float32)
            coarse_levels.append(current)

        self._coarse_operators = (
            coarse_levels,
            np.ascontiguousarray(level1_cap_f32, dtype=np.float32).reshape(-1),
        )
        return self._coarse_operators

    def implicit_linear_operator(self, capacity, derivative_scale):
        """Return ``K + derivative_scale * C`` as a SciPy LinearOperator and MG preconditioner."""
        scale = float(derivative_scale)
        shape = self.shape

        def matvec(vector):
            return self._apply_with_scale(scale, vector)

        operator = spla.LinearOperator(shape, matvec=matvec, dtype=np.float64)
        operator._matvec_into = lambda vector, out: self._apply_with_scale(
            scale, vector, out=out
        )
        operator._matvec_into_and_dot = lambda v, out: self._apply_with_scale_and_dot(
            scale, v, out
        )

        if self.rows < 8 or self.cols < 8:
            hs_bool = self.heatsink_mask.astype(bool, copy=False)
            inv_diag = np.empty(shape[0], dtype=np.float64)
            inv_view = inv_diag.reshape(self.layer_count, self.rows, self.cols)
            for l in range(self.layer_count):
                full_diag = _compact_layer_full_diag(self, l, scale, hs_bool)
                np.divide(
                    1.0,
                    full_diag,
                    out=inv_view[l],
                    where=np.abs(full_diag) > 1e-30,
                )
            preconditioner = spla.LinearOperator(
                shape,
                matvec=lambda vector: inv_diag * vector,
                dtype=np.float64,
            )
            preconditioner._solve_into = lambda v, out: np.multiply(
                inv_diag, v, out=out
            )
            return operator, preconditioner

        cap_key = _capacity_cache_key(capacity)
        cached = self._preconditioner_cache.get(scale)
        if cached is not None and cached[0] == cap_key:
            multigrid = cached[1]
        else:
            coarse_levels, level1_cap_f32 = self._get_coarse_operators()
            if capacity is not None:
                cap_arr = np.asarray(capacity, dtype=np.float64).reshape(
                    self.layer_count, self.rows, self.cols
                )
                level1_cap_used = _restrict_2x2(
                    cap_arr, dtype=np.float32
                ).reshape(-1)
            else:
                level1_cap_used = level1_cap_f32
            multigrid = _CompactGeometricMultigrid(
                self,
                coarse_levels,
                level1_cap_used,
                scale,
            )
            self._preconditioner_cache.clear()
            self._preconditioner_cache[scale] = (cap_key, multigrid)

        preconditioner = spla.LinearOperator(
            shape,
            matvec=multigrid.solve,
            dtype=np.float64,
        )
        preconditioner._solve_into = multigrid.solve_into
        preconditioner._solve_into_and_dot = multigrid.solve_into_and_dot
        return operator, preconditioner


def build_compact_structured_operator(
    layer_count: int,
    rows: int,
    cols: int,
    copper_mask: np.ndarray,
    t_cu: np.ndarray,
    t_fr4_eff: np.ndarray,
    k_cu: float,
    k_fr4: float,
    dx: float,
    dy: float,
    V_map: np.ndarray,
    gap_m: List[float],
    H_map: np.ndarray,
    settings: Dict,
    amb: float,
    rho_cu: float = 8960.0,
    cp_cu: float = 385.0,
    rho_fr4: float = 1850.0,
    cp_fr4: float = 1100.0,
    pad_cap_areal: float = 0.0,
) -> CompactStructuredThermalOperator:
    """Build an ultra-compact Level-0 operator from geometry masks and LUTs."""
    layer_count = int(layer_count)
    rows = int(rows)
    cols = int(cols)
    pixel_area = float(dx) * float(dy)
    gx_lut, gy_lut, gz_scale, cap_lut, pad_cap_cell = _compute_layer_luts(
        layer_count=layer_count,
        t_cu=t_cu,
        t_fr4_eff=t_fr4_eff,
        k_cu=k_cu,
        k_fr4=k_fr4,
        dx=dx,
        dy=dy,
        gap_m=gap_m,
        rho_cu=rho_cu,
        cp_cu=cp_cu,
        rho_fr4=rho_fr4,
        cp_fr4=cp_fr4,
        pad_cap_areal=pad_cap_areal,
    )

    hs_mask_u8 = np.ascontiguousarray(np.asarray(H_map, dtype=bool), dtype=np.uint8)
    via_enh = np.ascontiguousarray(
        np.clip(np.asarray(V_map, dtype=np.float32), 1.0, 50.0),
        dtype=np.float32,
    )
    copper_u8 = np.ascontiguousarray(
        np.asarray(copper_mask, dtype=bool), dtype=np.uint8
    )

    h_top = float(settings.get("h_conv", 10.0))
    top_diag = h_top * pixel_area
    top_rhs = top_diag * float(amb)
    top_ha = top_diag

    h_air_bot = float(settings.get("h_conv", 10.0))
    pad_thick_m = max(0.0001, float(settings.get("pad_th", 1.0)) * 1e-3)
    pad_k = float(settings.get("pad_k", 3.0))
    contact_factor = max(1e-6, float(settings.get("pad_contact_factor", 0.2)))
    h_sink = (pad_k / pad_thick_m) * contact_factor
    bot_air_diag = h_air_bot * pixel_area
    bot_pad_diag = h_sink * pixel_area

    sink_mode = str(settings.get("pad_sink_mode", "fixed") or "fixed").strip().lower()
    use_heatsink = bool(settings.get("use_heatsink", True))
    heatsink_mode_active = bool(
        use_heatsink and sink_mode == "heatsink" and np.any(hs_mask_u8)
    )
    if heatsink_mode_active:
        bot_air_rhs = bot_air_diag * float(amb)
        bot_pad_rhs = 0.0
        bot_air_ha = bot_air_diag
        bot_pad_ha = 0.0
    else:
        use_amb = bool(settings.get("pad_sink_temp_use_ambient", True))
        sink_temp_raw = settings.get("pad_sink_temp_c")
        sink_temp = (
            float(amb)
            if (use_amb or sink_temp_raw is None)
            else float(sink_temp_raw)
        )
        bot_air_rhs = bot_air_diag * float(amb)
        bot_pad_rhs = bot_pad_diag * sink_temp
        bot_air_ha = bot_air_diag
        bot_pad_ha = bot_pad_diag

    return CompactStructuredThermalOperator(
        layer_count=layer_count,
        rows=rows,
        cols=cols,
        copper_mask=copper_u8,
        via_enh=via_enh,
        heatsink_mask=hs_mask_u8,
        gx_lut=gx_lut,
        gy_lut=gy_lut,
        gz_scale=gz_scale,
        cap_lut=cap_lut,
        pad_cap_cell=pad_cap_cell,
        top_diag=top_diag,
        bot_air_diag=bot_air_diag,
        bot_pad_diag=bot_pad_diag,
        top_rhs=top_rhs,
        bot_air_rhs=bot_air_rhs,
        bot_pad_rhs=bot_pad_rhs,
        top_ha=top_ha,
        bot_air_ha=bot_air_ha,
        bot_pad_ha=bot_pad_ha,
        heatsink_mode_active=heatsink_mode_active,
    )


def _compute_bottom_boundary_terms(
    H_map: np.ndarray,
    pixel_area: float,
    settings: Dict,
    amb: float,
):
    """Compute bottom-surface diagonal conductance, RHS contribution, and hA."""
    h_air_bot = float(settings.get("h_conv", 10.0))
    pad_thick_m = max(0.0001, float(settings.get("pad_th", 1.0)) * 1e-3)
    pad_k = float(settings.get("pad_k", 3.0))
    contact_factor = max(1e-6, float(settings.get("pad_contact_factor", 0.2)))
    h_sink = (pad_k / pad_thick_m) * contact_factor

    h_map_arr = np.asarray(H_map, dtype=np.float64)
    g_air_bot = ((1.0 - h_map_arr) * h_air_bot * pixel_area).ravel()
    g_pad_bot = (h_map_arr * h_sink * pixel_area).ravel()
    diag_add_bot = g_air_bot + g_pad_bot

    sink_mode = str(settings.get("pad_sink_mode", "fixed") or "fixed").strip().lower()
    use_heatsink = bool(settings.get("use_heatsink", True))
    if use_heatsink and sink_mode == "heatsink" and np.any(h_map_arr):
        b_add_bot = g_air_bot * float(amb)
        hA_add_bot = g_air_bot
    else:
        use_amb = bool(settings.get("pad_sink_temp_use_ambient", True))
        sink_temp_raw = settings.get("pad_sink_temp_c")
        sink_temp = (
            float(amb)
            if (use_amb or sink_temp_raw is None)
            else float(sink_temp_raw)
        )
        b_add_bot = g_air_bot * float(amb) + g_pad_bot * sink_temp
        hA_add_bot = diag_add_bot

    return diag_add_bot, b_add_bot, hA_add_bot


def build_structured_operator(
    layer_count: int,
    rows: int,
    cols: int,
    copper_mask: np.ndarray,
    t_cu: np.ndarray,
    t_fr4_eff: np.ndarray,
    k_cu: float,
    k_fr4: float,
    dx: float,
    dy: float,
    V_map: np.ndarray,
    gap_m: List[float],
    H_map: np.ndarray,
    settings: Dict,
    amb: float,
):
    """Build the matrix-free equivalent of :func:`build_stiffness_matrix`."""
    eps = 1e-12
    pixel_area = dx * dy
    node_count = layer_count * rows * cols
    gx = np.empty((layer_count, rows, max(cols - 1, 0)), dtype=np.float64)
    gy = np.empty((layer_count, max(rows - 1, 0), cols), dtype=np.float64)

    for layer_idx in range(layer_count):
        mask = copper_mask[layer_idx]
        conductivity = np.where(mask, k_cu, k_fr4)
        thickness = np.where(mask, t_cu[layer_idx], t_fr4_eff[layer_idx])
        if cols > 1:
            harmonic_k = (
                2.0
                * conductivity[:, :-1]
                * conductivity[:, 1:]
                / (conductivity[:, :-1] + conductivity[:, 1:] + eps)
            )
            edge_thickness = 0.5 * (thickness[:, :-1] + thickness[:, 1:])
            gx[layer_idx] = harmonic_k * (edge_thickness * dy) / dx
        if rows > 1:
            harmonic_k = (
                2.0
                * conductivity[:-1, :]
                * conductivity[1:, :]
                / (conductivity[:-1, :] + conductivity[1:, :] + eps)
            )
            edge_thickness = 0.5 * (thickness[:-1, :] + thickness[1:, :])
            gy[layer_idx] = harmonic_k * (edge_thickness * dx) / dy

    if layer_count > 1 and gap_m:
        gz = np.empty((layer_count - 1, rows, cols), dtype=np.float64)
        via_enhancement = np.clip(V_map, 1.0, 50.0)
        for layer_idx in range(layer_count - 1):
            gap_value = max(gap_m[layer_idx], 1e-6)
            gz[layer_idx] = (
                k_fr4 * pixel_area / gap_value
            ) * via_enhancement
    else:
        gz = np.empty((0, rows, cols), dtype=np.float64)

    boundary_diag = np.zeros(node_count, dtype=np.float64)
    boundary_rhs = np.zeros(node_count, dtype=np.float64)
    h_area = np.zeros(node_count, dtype=np.float64)
    plane_cells = rows * cols
    plane_indices = np.arange(plane_cells, dtype=np.int64)

    h_top = float(settings.get("h_conv", 10.0))
    top_addition = h_top * pixel_area
    boundary_diag[plane_indices] += top_addition
    boundary_rhs[plane_indices] += top_addition * amb
    h_area[plane_indices] += top_addition

    diag_add_bot, b_add_bot, hA_add_bot = _compute_bottom_boundary_terms(
        H_map, pixel_area, settings, amb
    )
    bottom_indices = (layer_count - 1) * plane_cells + plane_indices
    boundary_diag[bottom_indices] += diag_add_bot
    boundary_rhs[bottom_indices] += b_add_bot
    h_area[bottom_indices] += hA_add_bot

    diagonal_view = boundary_diag.reshape(layer_count, rows, cols).copy()
    if cols > 1:
        diagonal_view[:, :, :-1] += gx
        diagonal_view[:, :, 1:] += gx
    if rows > 1:
        diagonal_view[:, :-1, :] += gy
        diagonal_view[:, 1:, :] += gy
    if layer_count > 1 and gz.size:
        diagonal_view[:-1, :, :] += gz
        diagonal_view[1:, :, :] += gz

    operator = StructuredThermalOperator(
        layer_count=layer_count,
        rows=rows,
        cols=cols,
        gx=gx,
        gy=gy,
        gz=gz,
        boundary_diag=boundary_diag,
        stiffness_diag=diagonal_view.reshape(-1),
    )
    return operator, boundary_rhs, h_area, boundary_diag


def build_stiffness_matrix(
    layer_count: int,
    rows: int,
    cols: int,
    copper_mask: np.ndarray,
    t_cu: np.ndarray,
    t_fr4_eff: np.ndarray,
    k_cu: float,
    k_fr4: float,
    dx: float,
    dy: float,
    V_map: np.ndarray,
    gap_m: List[float],
    H_map: np.ndarray,
    settings: Dict,
    amb: float
):
    """
    Build the sparse stiffness matrix for thermal conduction.

    Parameters
    ----------
    layer_count : int
        Number of copper layers.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    copper_mask : np.ndarray
        Boolean mask of copper locations, shape (layers, rows, cols).
    t_cu : np.ndarray
        Copper thickness per layer in meters.
    t_fr4_eff : np.ndarray
        Effective FR4 thickness per layer in meters.
    k_cu : float
        Thermal conductivity of copper (W/m-K).
    k_fr4 : float
        Thermal conductivity of FR4 (W/m-K).
    dx : float
        Grid spacing in x direction (meters).
    dy : float
        Grid spacing in y direction (meters).
    V_map : np.ndarray
        Via enhancement factors, shape (rows, cols).
    gap_m : List[float]
        Dielectric gap between layers in meters.
    H_map : np.ndarray
        Heatsink mask, shape (rows, cols).
    settings : Dict
        Simulation settings.
    amb : float
        Ambient temperature.

    Returns
    -------
    tuple
        (K, b, hA, diag_extra) where:
        - K : scipy.sparse.csr_matrix
            Stiffness matrix including boundary conditions.
        - b : np.ndarray
            Right-hand side contribution from boundaries.
        - hA : np.ndarray
            Heat transfer coefficients times area for each node.
        - diag_extra : np.ndarray
            Diagonal additions for boundary conditions.
    """
    eps = 1e-12
    pixel_area = dx * dy
    RC = rows * cols
    N = RC * layer_count

    col_right = np.arange(cols - 1)[None, :]
    row_all = np.arange(rows)[:, None]
    col_all = np.arange(cols)[None, :]
    row_down = np.arange(rows - 1)[:, None]
    plane_idx = np.arange(RC, dtype=np.int64)

    x_edges = layer_count * rows * max(cols - 1, 0)
    y_edges = layer_count * max(rows - 1, 0) * cols
    z_edges = ((layer_count - 1) * RC) if (layer_count > 1 and gap_m) else 0
    pair_count = x_edges + y_edges + z_edges

    row_idx = np.empty(pair_count * 4, dtype=np.int64)
    col_idx = np.empty(pair_count * 4, dtype=np.int64)
    data = np.empty(pair_count * 4, dtype=np.float64)
    offset = 0

    # Build in-plane conduction terms
    for l in range(layer_count):
        base = l * RC
        mask = copper_mask[l]
        k_layer = np.where(mask, k_cu, k_fr4)
        t_layer = np.where(mask, t_cu[l], t_fr4_eff[l])

        # X-direction coupling
        k_h = 2.0 * k_layer[:, :-1] * k_layer[:, 1:] / (k_layer[:, :-1] + k_layer[:, 1:] + eps)
        t_edge = 0.5 * (t_layer[:, :-1] + t_layer[:, 1:])
        Gx = k_h * (t_edge * dy) / dx

        idx_left = base + row_all * cols + col_right
        idx_right = idx_left + 1

        g = Gx.ravel()
        i_idx = idx_left.ravel()
        j_idx = idx_right.ravel()
        n = g.size
        row_idx[offset:offset + n] = i_idx
        col_idx[offset:offset + n] = i_idx
        data[offset:offset + n] = g
        offset += n
        row_idx[offset:offset + n] = j_idx
        col_idx[offset:offset + n] = j_idx
        data[offset:offset + n] = g
        offset += n
        row_idx[offset:offset + n] = i_idx
        col_idx[offset:offset + n] = j_idx
        data[offset:offset + n] = -g
        offset += n
        row_idx[offset:offset + n] = j_idx
        col_idx[offset:offset + n] = i_idx
        data[offset:offset + n] = -g
        offset += n

        # Y-direction coupling
        k_h = 2.0 * k_layer[:-1, :] * k_layer[1:, :] / (k_layer[:-1, :] + k_layer[1:, :] + eps)
        t_edge = 0.5 * (t_layer[:-1, :] + t_layer[1:, :])
        Gy = k_h * (t_edge * dx) / dy

        idx_up = base + row_down * cols + col_all
        idx_down = idx_up + cols

        g = Gy.ravel()
        i_idx = idx_up.ravel()
        j_idx = idx_down.ravel()
        n = g.size
        row_idx[offset:offset + n] = i_idx
        col_idx[offset:offset + n] = i_idx
        data[offset:offset + n] = g
        offset += n
        row_idx[offset:offset + n] = j_idx
        col_idx[offset:offset + n] = j_idx
        data[offset:offset + n] = g
        offset += n
        row_idx[offset:offset + n] = i_idx
        col_idx[offset:offset + n] = j_idx
        data[offset:offset + n] = -g
        offset += n
        row_idx[offset:offset + n] = j_idx
        col_idx[offset:offset + n] = i_idx
        data[offset:offset + n] = -g
        offset += n

    # Inter-layer coupling through vias
    if layer_count > 1 and gap_m:
        V_enh = np.clip(V_map, 1.0, 50.0)
        for l in range(layer_count - 1):
            gap_val = max(gap_m[l], 1e-6)
            Gz_base = k_fr4 * pixel_area / gap_val
            Gz = (Gz_base * V_enh).ravel()
            i_idx = l * RC + plane_idx
            j_idx = (l + 1) * RC + plane_idx
            n = Gz.size
            row_idx[offset:offset + n] = i_idx
            col_idx[offset:offset + n] = i_idx
            data[offset:offset + n] = Gz
            offset += n
            row_idx[offset:offset + n] = j_idx
            col_idx[offset:offset + n] = j_idx
            data[offset:offset + n] = Gz
            offset += n
            row_idx[offset:offset + n] = i_idx
            col_idx[offset:offset + n] = j_idx
            data[offset:offset + n] = -Gz
            offset += n
            row_idx[offset:offset + n] = j_idx
            col_idx[offset:offset + n] = i_idx
            data[offset:offset + n] = -Gz
            offset += n

    if offset != data.size:
        row_idx = row_idx[:offset]
        col_idx = col_idx[:offset]
        data = data[:offset]

    K_base = sp.coo_matrix(
        (data, (row_idx, col_idx)),
        shape=(N, N),
        dtype=np.float64
    ).tocsr()

    # Boundary conditions
    diag_extra = np.zeros(N, dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    # Top surface convection
    h_top = float(settings.get('h_conv', 10.0))
    diag_add_top = h_top * pixel_area
    top_idx = np.arange(RC, dtype=np.int64)
    diag_extra[top_idx] += diag_add_top
    b[top_idx] += diag_add_top * amb

    # Bottom surface: air convection + heatsink
    diag_add_bot, b_add_bot, hA_add_bot = _compute_bottom_boundary_terms(
        H_map, pixel_area, settings, amb
    )
    bot_base = (layer_count - 1) * RC
    bot_idx = bot_base + top_idx
    diag_extra[bot_idx] += diag_add_bot
    b[bot_idx] += b_add_bot

    hA = np.zeros(N, dtype=np.float64)
    hA[top_idx] += diag_add_top
    hA[bot_idx] += hA_add_bot

    K = K_base + sp.diags(diag_extra, format="csr")

    return K, b, hA, diag_extra


def _release_linear_solver(owner):
    """Release native resources associated with an optional solver owner."""
    if owner is not None and hasattr(owner, "free_memory"):
        try:
            owner.free_memory(everything=True)
        except Exception:
            pass


def _factor_linear_system(A: sp.csr_matrix):
    """Factor a thermal linear system using SciPy SuperLU."""
    owner = spla.splu(A.tocsc())
    return owner.solve, owner, "SciPy", None


class _CancelledIterativeSolve(RuntimeError):
    """Internal signal used to stop SciPy CG from its callback."""


_OPENBLAS_THREAD_SYMBOLS = (
    ("scipy_openblas_get_num_threads64_", "scipy_openblas_set_num_threads64_"),
    ("scipy_openblas_get_num_threads", "scipy_openblas_set_num_threads"),
    ("openblas_get_num_threads64_", "openblas_set_num_threads64_"),
    ("openblas_get_num_threads", "openblas_set_num_threads"),
)


@contextlib.contextmanager
def _limit_blas_threads(max_threads: int = 1):
    """Temporarily cap BLAS worker threads to avoid OpenMP spin-barrier contention.

    Uses ``threadpoolctl`` when available, otherwise scans ``/proc/self/maps``
    on Linux for already-loaded OpenBLAS shared libraries and restores their
    original thread counts on exit. Silently no-ops on unsupported platforms.
    """
    limit = max(1, int(max_threads))
    tpc_ctx = None
    try:
        import threadpoolctl

        tpc_ctx = threadpoolctl.threadpool_limits(limits=limit, user_api="blas")
        tpc_ctx.__enter__()
    except Exception:
        tpc_ctx = None

    if tpc_ctx is not None:
        try:
            yield
        finally:
            try:
                tpc_ctx.__exit__(None, None, None)
            except Exception:
                pass
        return

    restorers = []
    if sys.platform.startswith("linux"):
        try:
            seen_paths = set()
            with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if "openblas" not in line.lower() or ".so" not in line:
                        continue
                    parts = line.strip().split()
                    if not parts:
                        continue
                    lib_path = parts[-1]
                    if (
                        not lib_path.startswith("/")
                        or "openblas" not in lib_path.lower()
                        or lib_path in seen_paths
                    ):
                        continue
                    seen_paths.add(lib_path)
                    try:
                        lib = ctypes.CDLL(lib_path)
                    except OSError:
                        continue
                    for get_name, set_name in _OPENBLAS_THREAD_SYMBOLS:
                        get_fn = getattr(lib, get_name, None)
                        set_fn = getattr(lib, set_name, None)
                        if get_fn is None or set_fn is None:
                            continue
                        try:
                            get_fn.argtypes = []
                            get_fn.restype = ctypes.c_int
                            set_fn.argtypes = [ctypes.c_int]
                            set_fn.restype = None
                            previous = int(get_fn())
                            if previous > limit:
                                set_fn(limit)
                                restorers.append((set_fn, previous))
                        except Exception:
                            pass
                        break
        except Exception:
            pass

    try:
        yield
    finally:
        for set_fn, previous in restorers:
            try:
                set_fn(previous)
            except Exception:
                pass


def _solve_pcg(
    operator,
    preconditioner,
    rhs,
    initial,
    cancel_check=None,
    relative_tolerance=1e-6,
    maxiter=100,
    retry_maxiter=200,
    out=None,
    workspaces=None,
):
    """Solve one implicit step with preconditioned conjugate gradients."""
    rhs_arr = _as_1d_contiguous_f64(rhs)
    n = rhs_arr.shape[0]
    if workspaces is None:
        r = np.empty(n, dtype=np.float64)
        z = np.empty(n, dtype=np.float64)
        p = np.empty(n, dtype=np.float64)
        ap = np.empty(n, dtype=np.float64)
    else:
        r, z, p, ap = workspaces

    if out is None:
        x = np.array(initial, dtype=np.float64, copy=True).reshape(-1)
    else:
        x = out
        if x is not initial:
            np.copyto(x, np.asarray(initial, dtype=np.float64).reshape(-1))

    bnorm = _norm_f64(rhs_arr)
    if bnorm == 0.0:
        x.fill(0.0)
        return x, 0, False, 0.0

    tol = float(relative_tolerance) * bnorm
    matvec_into = getattr(operator, "_matvec_into", None)
    apply_a_and_dot = getattr(operator, "_matvec_into_and_dot", None)
    solve_into = getattr(preconditioner, "_solve_into", None)
    apply_m_and_dot = getattr(preconditioner, "_solve_into_and_dot", None)

    def apply_a(vec, dst):
        if matvec_into is not None:
            matvec_into(vec, dst)
        else:
            np.copyto(dst, operator.matvec(vec))

    def apply_m(vec, dst):
        if solve_into is not None:
            solve_into(vec, dst)
        else:
            np.copyto(dst, preconditioner.matvec(vec))

    apply_a(x, ap)
    np.subtract(rhs_arr, ap, out=r)
    rnorm = _norm_f64(r)
    if rnorm <= tol:
        return x, 0, False, rnorm / bnorm

    iterations = 0
    retried = False
    converged = False

    for attempt, limit in ((False, int(maxiter)), (True, int(retry_maxiter))):
        if attempt:
            retried = True
            apply_a(x, ap)
            np.subtract(rhs_arr, ap, out=r)
            rnorm = _norm_f64(r)
            if rnorm <= tol:
                converged = True
                break

        rz_old = 0.0
        for it in range(limit):
            if apply_m_and_dot is not None:
                rz_new = float(apply_m_and_dot(r, z))
            else:
                apply_m(r, z)
                rz_new = _dot_f64(r, z)
            if rz_new < 0.0:
                raise RuntimeError("PCG failed with solver status -1.")
            if rz_new == 0.0:
                break
            if it == 0:
                np.copyto(p, z)
            else:
                beta = rz_new / rz_old
                if not (
                    HAS_NATIVE_PCG
                    and pcg_update_p_native is not None
                    and pcg_update_p_native(p, z, beta)
                ):
                    np.multiply(p, beta, out=p)
                    np.add(p, z, out=p)

            if apply_a_and_dot is not None:
                p_ap = float(apply_a_and_dot(p, ap))
            else:
                apply_a(p, ap)
                p_ap = _dot_f64(p, ap)
            if p_ap <= 0.0:
                raise RuntimeError("PCG failed with solver status -1.")
            alpha = rz_new / p_ap

            r_norm2 = (
                pcg_update_xr_norm2_native(x, r, p, ap, alpha)
                if (HAS_NATIVE_PCG and pcg_update_xr_norm2_native is not None)
                else None
            )
            if r_norm2 is None:
                np.multiply(p, alpha, out=z)
                np.add(x, z, out=x)
                np.multiply(ap, alpha, out=z)
                np.subtract(r, z, out=r)
                r_norm2 = _dot_f64(r, r)
            rnorm = math.sqrt(r_norm2) if not (r_norm2 < 0.0) else 0.0

            rz_old = rz_new
            iterations += 1
            if cancel_check and cancel_check():
                raise _CancelledIterativeSolve()

            if rnorm <= tol:
                converged = True
                break

        if converged:
            break

    if not converged:
        raise RuntimeError(
            f"PCG did not converge after {iterations} iterations "
            f"(relative tolerance {relative_tolerance:g})."
        )

    apply_a(x, ap)
    np.subtract(rhs_arr, ap, out=r)
    relative_residual = _norm_f64(r) / max(bnorm, 1e-30)
    return x, iterations, retried, relative_residual


@dataclass
class _HeatsinkPhaseState:
    """Cached Sherman-Morrison rank-1 coupling state for one linear operator."""

    g: np.ndarray
    g_sum: float
    g_hs_amb: float
    c_hs: float
    alpha_hs: float
    u: np.ndarray
    delta: float
    bottom_layer_only: bool = False

    def dot_g(self, vec: np.ndarray) -> float:
        """Compute ``dot(g, vec)`` supporting both 3D and 2D bottom-layer g."""
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if self.bottom_layer_only:
            return float(np.dot(self.g, v[-self.g.size :]))
        return float(np.dot(self.g, v))

    def add_scaled_g(self, out: np.ndarray, coeff: float) -> np.ndarray:
        """Add ``coeff * g`` in-place into ``out``."""
        if self.bottom_layer_only:
            out[-self.g.size :] += coeff * self.g
        else:
            out += coeff * self.g
        return out

    @classmethod
    def from_u(
        cls,
        heatsink_model: HeatsinkModel,
        gamma: float,
        u: np.ndarray,
    ) -> "_HeatsinkPhaseState":
        bottom_only = bool(getattr(heatsink_model, "bottom_layer_only", False))
        g = heatsink_model.pad_conductance
        g_sum = float(np.sum(g))
        g_hs_amb = heatsink_model.g_hs_amb
        c_hs = max(0.0, float(heatsink_model.capacity_j_k))
        alpha_hs = (gamma * c_hs if c_hs > 0.0 else 0.0) + g_hs_amb + g_sum
        u_vec = np.asarray(u, dtype=np.float64).reshape(-1)
        g_dot_u = (
            float(np.dot(g, u_vec[-g.size :]))
            if bottom_only
            else float(np.dot(g, u_vec))
        )
        delta = alpha_hs - g_dot_u
        if delta <= 1e-18:
            raise RuntimeError(
                f"Sherman-Morrison heatsink denominator is non-positive ({delta:.3e})."
            )
        return cls(
            g=g,
            g_sum=g_sum,
            g_hs_amb=g_hs_amb,
            c_hs=c_hs,
            alpha_hs=alpha_hs,
            u=u_vec,
            delta=delta,
            bottom_layer_only=bottom_only,
        )

    @classmethod
    def prepare_direct(
        cls,
        heatsink_model: HeatsinkModel,
        gamma: float,
        solve_fn: Callable[[np.ndarray], np.ndarray],
        node_count: Optional[int] = None,
    ) -> "_HeatsinkPhaseState":
        g = heatsink_model.pad_conductance
        if getattr(heatsink_model, "bottom_layer_only", False):
            if node_count is None:
                raise ValueError("node_count is required for bottom_layer_only heatsink.")
            g_rhs = np.zeros(node_count, dtype=np.float64)
            g_rhs[-g.size :] = g
            u = solve_fn(g_rhs)
        else:
            u = solve_fn(g)
        return cls.from_u(heatsink_model, gamma, u)

    @classmethod
    def prepare_pcg(
        cls,
        heatsink_model: HeatsinkModel,
        gamma: float,
        operator,
        preconditioner,
        cancel_check=None,
        rtol: float = 1e-8,
        workspaces=None,
    ):
        g = heatsink_model.pad_conductance
        node_count = operator.shape[0]
        if getattr(heatsink_model, "bottom_layer_only", False):
            g_rhs = np.zeros(node_count, dtype=np.float64)
            g_rhs[-g.size :] = g
        else:
            g_rhs = g
        u = np.zeros(node_count, dtype=np.float64)
        _, _, _, rel_res = _solve_pcg(
            operator,
            preconditioner,
            g_rhs,
            u,
            cancel_check=cancel_check,
            relative_tolerance=rtol,
            maxiter=300,
            retry_maxiter=600,
            out=u,
            workspaces=workspaces,
        )
        return cls.from_u(heatsink_model, gamma, u), rel_res

    def scalar_rhs(
        self,
        amb: float,
        t_hs_n: float = 0.0,
        t_hs_nm1: float = 0.0,
        dt: float = 0.0,
        bdf2: bool = False,
    ) -> float:
        r_hs = self.g_hs_amb * float(amb)
        if self.c_hs > 0.0 and dt > 0.0:
            d_hs = self.c_hs / dt
            if bdf2:
                r_hs += 2.0 * d_hs * t_hs_n - 0.5 * d_hs * t_hs_nm1
            else:
                r_hs += d_hs * t_hs_n
        return float(r_hs)

    def modified_rhs(
        self, pcb_rhs: np.ndarray, r_hs: float, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if out is None:
            dst = np.array(pcb_rhs, dtype=np.float64, copy=True)
        else:
            dst = out
            if dst is not pcb_rhs:
                np.copyto(dst, pcb_rhs)
        return self.add_scaled_g(dst, r_hs / self.alpha_hs)

    def warm_start_x1(
        self, t_guess: np.ndarray, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        coeff = self.dot_g(t_guess) / self.alpha_hs
        if out is None:
            return t_guess - coeff * self.u
        if out is t_guess:
            out -= coeff * self.u
            return out
        np.multiply(self.u, -coeff, out=out)
        np.add(t_guess, out, out=out)
        return out

    def recover(
        self,
        x1: np.ndarray,
        r_hs: float,
        out: Optional[np.ndarray] = None,
        scratch: Optional[np.ndarray] = None,
    ):
        x1_vec = np.asarray(x1, dtype=np.float64).reshape(-1)
        g_dot_x1 = self.dot_g(x1_vec)
        scale = g_dot_x1 / self.delta
        if out is None:
            t_new = x1_vec + scale * self.u
        elif scratch is not None:
            np.multiply(self.u, scale, out=scratch)
            np.add(x1_vec, scratch, out=out)
            t_new = out
        else:
            t_new = out
            if t_new is not x1_vec:
                np.multiply(self.u, scale, out=t_new)
                np.add(x1_vec, t_new, out=t_new)
            else:
                t_new += scale * self.u
        t_hs_new = (r_hs + self.dot_g(t_new)) / self.alpha_hs
        return t_new, float(t_hs_new)


def _validate_heatsink_model(
    heatsink_model: Optional[HeatsinkModel],
    node_count: int,
    plane_size: Optional[int] = None,
) -> bool:
    """Return True if heatsink_model is active, validating vector shape."""
    if heatsink_model is None or not heatsink_model.is_active:
        return False
    if getattr(heatsink_model, "bottom_layer_only", False):
        expected = (plane_size,) if plane_size is not None else None
        if expected is not None and heatsink_model.pad_conductance.shape != expected:
            raise ValueError(
                f"heatsink_model.pad_conductance shape {heatsink_model.pad_conductance.shape} "
                f"does not match bottom layer size {expected}."
            )
        if heatsink_model.pad_conductance.size > node_count:
            raise ValueError(
                f"heatsink_model.pad_conductance size {heatsink_model.pad_conductance.size} "
                f"exceeds node count ({node_count})."
            )
        return True
    if heatsink_model.pad_conductance.shape != (node_count,):
        raise ValueError(
            f"heatsink_model.pad_conductance shape {heatsink_model.pad_conductance.shape} "
            f"does not match node count ({node_count},)."
        )
    return True


def _heatsink_diagnostics(
    heatsink_model: Optional[HeatsinkModel],
    T_flat: np.ndarray,
    t_hs: float,
    hA: Optional[np.ndarray],
    amb: float,
    operator: Optional[Any] = None,
):
    """Compute total heat output (W) and heatsink metadata for k_norm_info."""
    if hA is None and operator is not None and hasattr(operator, "convection_power"):
        p_pcb_air = float(operator.convection_power(T_flat, amb))
    else:
        p_pcb_air = float(np.dot(hA, T_flat - amb))
    if heatsink_model is None or not heatsink_model.is_active:
        return p_pcb_air, {}
    g = heatsink_model.pad_conductance
    g_hs_amb = heatsink_model.g_hs_amb
    c_hs = max(0.0, float(heatsink_model.capacity_j_k))
    p_hs_amb = float(g_hs_amb * (t_hs - amb))
    if getattr(heatsink_model, "bottom_layer_only", False):
        p_pcb_to_hs = float(np.dot(g, T_flat[-g.size :] - t_hs))
    else:
        p_pcb_to_hs = float(np.dot(g, T_flat - t_hs))
    p_total = p_pcb_air + p_hs_amb
    hs_frac = float(p_hs_amb / p_total) if p_total > 1e-12 else 0.0
    return p_total, {
        "heatsink_enabled": True,
        "heatsink_mode": str(heatsink_model.mode or "heatsink"),
        "heatsink_temp_c": float(t_hs),
        "heatsink_temp_rise_c": float(t_hs - amb),
        "heatsink_rth_kw": float(heatsink_model.rth_kw),
        "heatsink_capacity_j_k": float(c_hs),
        "heatsink_tau_s": float(heatsink_model.tau_s),
        "heatsink_power_w": p_hs_amb,
        "heatsink_pad_flux_w": p_pcb_to_hs,
        "pcb_convection_power_w": p_pcb_air,
        "heatsink_power_fraction": hs_frac,
    }


def _power_sum(source) -> float:
    """Return total power (W) for either a SparsePowerVector or dense ndarray."""
    if source is None:
        return 0.0
    if isinstance(source, SparsePowerVector):
        return source.total_power()
    return float(np.sum(source))


def run_steady_state(
    config: SolverConfig,
    K: sp.csr_matrix,
    Q: np.ndarray,
    b: np.ndarray,
    hA: np.ndarray,
    layer_count: int,
    rows: int,
    cols: int,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    heatsink_model: Optional[HeatsinkModel] = None,
) -> SolverResult:
    """Solve the steady-state thermal system ``K * T = Q + b`` directly."""
    N = K.shape[0]
    amb = config.amb
    pin = _power_sum(Q)
    hs_active = _validate_heatsink_model(heatsink_model, N, plane_size=rows * cols)
    t_hs = (
        float(amb if heatsink_model.initial_temp_c is None else heatsink_model.initial_temp_c)
        if hs_active
        else float(amb)
    )

    if cancel_check and cancel_check():
        T_abort = np.full((layer_count, rows, cols), amb, dtype=np.float64)
        _, hs_info = _heatsink_diagnostics(
            heatsink_model if hs_active else None,
            T_abort.reshape(-1),
            t_hs,
            hA,
            amb,
        )
        k_norm_info = {
            "strategy": "direct_steady_state",
            "backend": "SciPy",
            "multi_phase": False,
            "time_stepping": "steady_state",
            "N": N,
            "nnz_K": int(K.nnz),
            "dt_base": config.dt_base,
            "steps_target": 1,
            "steps_total": 0,
            "factorization_s": 0.0,
            "factorizations": 0,
            "factorization_cache_hit": False,
            "avg_solve_s": 0.0,
            "pin_w": pin,
            "pout_final_w": 0.0,
            "steady_rel_diff": None,
            "steady_state": True,
            "early_steady_state_stop": False,
        }
        k_norm_info.update(hs_info)
        return SolverResult(
            T=T_abort,
            aborted=True,
            step_counter=0,
            total_solve_time=0.0,
            total_factor_time=0.0,
            factor_count=0,
            snapshot_stats=[],
            snapshot_files=[],
            phase_metrics=[],
            balance_history=[],
            k_norm_info=k_norm_info,
        )

    q_dense = Q.to_dense() if isinstance(Q, SparsePowerVector) else np.asarray(Q, dtype=np.float64)
    rhs = q_dense + np.asarray(b, dtype=np.float64)
    factor_start = time.perf_counter()
    solve_fn, owner, backend, _ = _factor_linear_system(K)
    factor_time = time.perf_counter() - factor_start
    try:
        solve_start = time.perf_counter()
        if hs_active:
            phase_hs = _HeatsinkPhaseState.prepare_direct(
                heatsink_model, 0.0, solve_fn, node_count=N
            )
            r_hs = phase_hs.scalar_rhs(amb)
            x1 = solve_fn(phase_hs.modified_rhs(rhs, r_hs))
            Tn, t_hs = phase_hs.recover(x1, r_hs)
        else:
            Tn = np.asarray(solve_fn(rhs), dtype=np.float64).reshape(-1)
        solve_time = time.perf_counter() - solve_start
    finally:
        _release_linear_solver(owner)

    aborted = False
    if progress_callback and not progress_callback(1, 1):
        aborted = True

    T = Tn.reshape((layer_count, rows, cols))
    pout_final, hs_info = _heatsink_diagnostics(
        heatsink_model if hs_active else None, Tn, t_hs, hA, amb
    )
    steady_rel_diff = abs(pin - pout_final) / max(abs(pin), 1e-9)

    phase_metrics = [{
        "phase": "steady_state",
        "dt": 0.0,
        "steps": 1,
        "assembly_s": 0.0,
        "factorization_s": factor_time,
        "avg_solve_s": solve_time,
        "backend": backend,
    }]
    k_norm_info = {
        "strategy": "direct_steady_state",
        "backend": backend,
        "multi_phase": False,
        "time_stepping": "steady_state",
        "N": N,
        "nnz_K": int(K.nnz),
        "dt_base": config.dt_base,
        "steps_target": 1,
        "steps_total": 1,
        "factorization_s": factor_time,
        "factorizations": 1,
        "factorization_cache_hit": False,
        "avg_solve_s": solve_time,
        "pin_w": pin,
        "pout_final_w": pout_final,
        "steady_rel_diff": steady_rel_diff,
        "steady_state": True,
        "early_steady_state_stop": False,
    }
    k_norm_info.update(hs_info)
    return SolverResult(
        T=T,
        aborted=aborted,
        step_counter=1,
        total_solve_time=solve_time,
        total_factor_time=factor_time,
        factor_count=1,
        snapshot_stats=[],
        snapshot_files=[],
        phase_metrics=phase_metrics,
        balance_history=[],
        k_norm_info=k_norm_info,
    )


def run_steady_state_matrix_free(
    config: SolverConfig,
    operator: StructuredThermalOperator,
    C: Optional[np.ndarray],
    Q,
    b: Optional[np.ndarray],
    hA: Optional[np.ndarray],
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    heatsink_model: Optional[HeatsinkModel] = None,
) -> SolverResult:
    """Solve ``K * T = Q + b`` directly using a matrix-free or adaptive operator."""
    layer_count = operator.layer_count
    rows = operator.rows
    cols = operator.cols
    node_count = operator.shape[0]
    is_compact = isinstance(operator, CompactStructuredThermalOperator)
    amb = config.amb
    pin = _power_sum(Q)
    hs_active = _validate_heatsink_model(
        heatsink_model, node_count, plane_size=rows * cols
    )
    t_hs = (
        float(amb if heatsink_model.initial_temp_c is None else heatsink_model.initial_temp_c)
        if hs_active
        else float(amb)
    )
    if bool(getattr(operator, "prefer_derivative_start", False)):
        backend_name = "Adaptive-Multigrid-PCG"
    elif HAS_NATIVE_CORE:
        backend_name = "Native-MatrixFree-PCG"
    else:
        backend_name = "MatrixFree-PCG"

    if cancel_check and cancel_check():
        T_abort = np.full((layer_count, rows, cols), amb, dtype=np.float64)
        _, hs_info = _heatsink_diagnostics(
            heatsink_model if hs_active else None,
            T_abort.reshape(-1),
            t_hs,
            hA,
            amb,
            operator=operator,
        )
        k_norm_info = {
            "strategy": "direct_steady_state",
            "backend": backend_name,
            "multi_phase": False,
            "time_stepping": "steady_state",
            "N": node_count,
            "nnz_K": operator.nnz_estimate,
            "dt_base": config.dt_base,
            "steps_target": 1,
            "steps_total": 0,
            "factorization_s": 0.0,
            "factorizations": 0,
            "factorization_cache_hit": False,
            "avg_solve_s": 0.0,
            "avg_pcg_iterations": 0.0,
            "pcg_retry_count": 0,
            "pcg_relative_residual_max": 0.0,
            "pin_w": pin,
            "pout_final_w": 0.0,
            "steady_rel_diff": None,
            "steady_state": True,
            "early_steady_state_stop": False,
        }
        k_norm_info.update(hs_info)
        return SolverResult(
            T=T_abort,
            aborted=True,
            step_counter=0,
            total_solve_time=0.0,
            total_factor_time=0.0,
            factor_count=0,
            snapshot_stats=[],
            snapshot_files=[],
            phase_metrics=[],
            balance_history=[],
            k_norm_info=k_norm_info,
        )

    capacity = (
        None
        if (C is None and is_compact)
        else (
            np.zeros(node_count, dtype=np.float64)
            if C is None
            else np.asarray(C, dtype=np.float64)
        )
    )
    rhs = np.zeros(node_count, dtype=np.float64)
    if isinstance(Q, SparsePowerVector):
        Q.add_to(rhs)
    elif Q is not None:
        rhs += np.asarray(Q, dtype=np.float64)
    if b is not None:
        rhs += np.asarray(b, dtype=np.float64)
    elif is_compact:
        operator.add_boundary_rhs(rhs)

    initial = np.full(node_count, amb, dtype=np.float64)
    sol_buf = np.empty(node_count, dtype=np.float64)
    pcg_workspaces = (
        np.empty(node_count, dtype=np.float64),
        np.empty(node_count, dtype=np.float64),
        np.empty(node_count, dtype=np.float64),
        np.empty(node_count, dtype=np.float64),
    )

    setup_start = time.perf_counter()
    aborted = False
    step_counter = 0
    iterations = 0
    retried = False
    relative_residual = 0.0
    solve_time = 0.0
    setup_time = 0.0
    temperature = initial
    use_blas_limit = (
        HAS_NATIVE_CORE
        and not bool(getattr(operator, "prefer_derivative_start", False))
    )
    blas_guard = (
        _limit_blas_threads(1) if use_blas_limit else contextlib.nullcontext()
    )

    with blas_guard:
        steady_operator, steady_preconditioner = operator.implicit_linear_operator(
            capacity, 0.0
        )
        phase_hs = None
        u_rel_res = 0.0
        try:
            if hs_active:
                phase_hs, u_rel_res = _HeatsinkPhaseState.prepare_pcg(
                    heatsink_model,
                    0.0,
                    steady_operator,
                    steady_preconditioner,
                    cancel_check=cancel_check,
                    rtol=1e-8,
                    workspaces=pcg_workspaces,
                )
            setup_time = time.perf_counter() - setup_start

            solve_start = time.perf_counter()
            if phase_hs is not None:
                r_hs = phase_hs.scalar_rhs(amb)
                phase_hs.modified_rhs(rhs, r_hs, out=rhs)
                phase_hs.warm_start_x1(initial, out=sol_buf)
                solve_init = sol_buf
            else:
                r_hs = 0.0
                solve_init = initial
            solution, iterations, retried, relative_residual = _solve_pcg(
                steady_operator,
                steady_preconditioner,
                rhs,
                solve_init,
                cancel_check=cancel_check,
                relative_tolerance=1e-7,
                maxiter=250,
                retry_maxiter=500,
                out=sol_buf,
                workspaces=pcg_workspaces,
            )
            if phase_hs is not None:
                temperature, t_hs = phase_hs.recover(
                    solution, r_hs, out=sol_buf, scratch=pcg_workspaces[1]
                )
                relative_residual = max(relative_residual, u_rel_res)
            else:
                temperature = solution
            solve_time = time.perf_counter() - solve_start
            step_counter = 1
        except _CancelledIterativeSolve:
            solve_time = time.perf_counter() - setup_start
            aborted = True

    if not aborted and progress_callback and not progress_callback(1, 1):
        aborted = True

    final_view = temperature.reshape(layer_count, rows, cols)
    pout_final, hs_info = _heatsink_diagnostics(
        heatsink_model if hs_active else None,
        temperature,
        t_hs,
        hA,
        amb,
        operator=operator,
    )
    steady_rel_diff = (
        abs(pin - pout_final) / max(abs(pin), 1e-9) if step_counter > 0 else None
    )

    phase_metrics = [{
        "phase": "steady_state",
        "dt": 0.0,
        "steps": step_counter,
        "assembly_s": setup_time,
        "factorization_s": 0.0,
        "avg_solve_s": solve_time,
        "avg_pcg_iterations": float(iterations),
        "backend": backend_name,
    }]
    k_norm_info = {
        "strategy": "direct_steady_state",
        "backend": backend_name,
        "multi_phase": False,
        "time_stepping": "steady_state",
        "N": node_count,
        "nnz_K": operator.nnz_estimate,
        "dt_base": config.dt_base,
        "steps_target": 1,
        "steps_total": step_counter,
        "factorization_s": 0.0,
        "factorizations": 0,
        "factorization_cache_hit": False,
        "avg_solve_s": solve_time,
        "avg_pcg_iterations": float(iterations),
        "pcg_retry_count": int(retried),
        "pcg_relative_residual_max": relative_residual,
        "pin_w": pin,
        "pout_final_w": pout_final,
        "steady_rel_diff": steady_rel_diff,
        "steady_state": True,
        "early_steady_state_stop": False,
    }
    k_norm_info.update(hs_info)
    return SolverResult(
        T=final_view,
        aborted=aborted,
        step_counter=step_counter,
        total_solve_time=solve_time,
        total_factor_time=0.0,
        factor_count=0,
        snapshot_stats=[],
        snapshot_files=[],
        phase_metrics=phase_metrics,
        balance_history=[],
        k_norm_info=k_norm_info,
    )


def run_simulation_matrix_free(
    config: SolverConfig,
    operator: StructuredThermalOperator,
    C: Optional[np.ndarray],
    Q,
    b: Optional[np.ndarray],
    hA: Optional[np.ndarray],
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    snapshot_callback: Optional[Callable[[np.ndarray, float, int], str]] = None,
    Q_func: Optional[Callable[[float], np.ndarray]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    heatsink_model: Optional[HeatsinkModel] = None,
) -> SolverResult:
    """Run BDF2 using a matrix-free structured operator and iterative solves."""
    if config.steady_state and Q_func is None:
        return run_steady_state_matrix_free(
            config=config,
            operator=operator,
            C=C,
            Q=Q,
            b=b,
            hA=hA,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            heatsink_model=heatsink_model,
        )

    layer_count = operator.layer_count
    rows = operator.rows
    cols = operator.cols
    node_count = operator.shape[0]
    is_compact = isinstance(operator, CompactStructuredThermalOperator)
    if C is None and not is_compact:
        raise ValueError("C cannot be None unless operator is CompactStructuredThermalOperator.")
    ambient = config.amb
    hs_active = _validate_heatsink_model(
        heatsink_model, node_count, plane_size=rows * cols
    )
    c_hs = max(0.0, float(heatsink_model.capacity_j_k)) if hs_active else 0.0
    t_hs = (
        float(ambient if heatsink_model.initial_temp_c is None else heatsink_model.initial_temp_c)
        if hs_active
        else float(ambient)
    )
    t_hs_prev = t_hs
    temperature = np.full(node_count, ambient, dtype=np.float64)
    previous_temperature = temperature.copy()
    next_temperature = np.empty(node_count, dtype=np.float64)
    guess_workspace = np.empty(node_count, dtype=np.float64)
    rhs_workspace = np.empty(node_count, dtype=np.float64)
    pcg_workspaces = (
        np.empty(node_count, dtype=np.float64),
        np.empty(node_count, dtype=np.float64),
        np.empty(node_count, dtype=np.float64),
        np.empty(node_count, dtype=np.float64),
    )

    def compute_energy(temp_vec: np.ndarray) -> float:
        if C is None and is_compact:
            pcb_energy = operator.thermal_energy(temp_vec, ambient)
        else:
            np.subtract(temp_vec, ambient, out=pcg_workspaces[0])
            pcb_energy = float(np.dot(C, pcg_workspaces[0]))
        return pcb_energy + (c_hs * (t_hs - ambient) if hs_active else 0.0)

    def compute_pout(temp_vec: np.ndarray) -> float:
        if hA is None and is_compact:
            pcb_pout = operator.convection_power(temp_vec, ambient)
        else:
            np.subtract(temp_vec, ambient, out=pcg_workspaces[0])
            pcb_pout = float(np.dot(hA, pcg_workspaces[0]))
        return pcb_pout + (
            heatsink_model.g_hs_amb * (t_hs - ambient) if hs_active else 0.0
        )

    def add_b(dst: np.ndarray) -> None:
        if b is not None:
            dst += b
        elif is_compact:
            operator.add_boundary_rhs(dst)

    def add_q(dst: np.ndarray, q_src) -> None:
        if isinstance(q_src, SparsePowerVector):
            q_src.add_to(dst)
        elif q_src is not None:
            dst += q_src

    initial_energy = compute_energy(temperature)
    current_time = 0.0
    step_counter = 0
    aborted = False
    early_stopped = False
    total_solve_time = 0.0
    total_iterations = 0
    retry_count = 0
    max_relative_residual = 0.0
    snapshot_stats = []
    snapshot_files = []
    phase_metrics = []
    balance_history = []
    previous_snapshot_time = 0.0
    previous_snapshot_energy = initial_energy
    balance_integral = 0.0
    pout_step = compute_pout(temperature)
    source = Q_func(0.0) if Q_func is not None else Q
    pin = _power_sum(source)
    next_snapshot_idx = 0
    snapshot_number = 1
    if bool(getattr(operator, "prefer_derivative_start", False)):
        backend_name = "Adaptive-Multigrid-PCG"
    elif HAS_NATIVE_CORE:
        backend_name = "Native-MatrixFree-PCG"
    else:
        backend_name = "MatrixFree-PCG"

    time_stepping_used, phase_plan = _build_phase_plan(config, node_count)
    total_steps = sum(item[2] for item in phase_plan)
    update_interval = max(1, total_steps // 100)

    def should_early_stop():
        if (
            not config.early_stop
            or Q_func is not None
            or step_counter < 5
            or pin <= 1e-12
        ):
            return False
        if config.snapshots_enabled and next_snapshot_idx < len(config.snap_times):
            return False
        if abs(pin - pout_step) / pin >= 1e-4:
            return False
        np.subtract(temperature, previous_temperature, out=pcg_workspaces[0])
        np.abs(pcg_workspaces[0], out=pcg_workspaces[0])
        max_delta = float(np.max(pcg_workspaces[0]))
        if hs_active:
            max_delta = max(max_delta, abs(t_hs - t_hs_prev))
        return max_delta < 1e-4

    def record_snapshot(t_elapsed):
        nonlocal previous_snapshot_time, previous_snapshot_energy, snapshot_number
        view = temperature.reshape(layer_count, rows, cols)
        energy = compute_energy(temperature)
        energy_delta = energy - initial_energy
        absolute_error = abs(energy_delta - balance_integral)
        relative_error = absolute_error / max(abs(t_elapsed * pin), 1e-9)
        interval = t_elapsed - previous_snapshot_time
        if interval > 0.0:
            balance_history.append({
                "delta_t": interval,
                "dE": energy - previous_snapshot_energy,
            })
        previous_snapshot_time = t_elapsed
        previous_snapshot_energy = energy
        snap_entry = {
            "t": t_elapsed,
            "max_top": float(np.max(view[0])),
            "max_bottom": float(np.max(view[-1])),
            "energy": energy,
            "pin": pin,
            "pout": pout_step,
            "eps_abs": absolute_error,
            "eps_rel": relative_error,
            "energy_warn": relative_error > 0.01 or absolute_error > 0.01,
        }
        if hs_active:
            snap_entry["heatsink_temp_c"] = float(t_hs)
        snapshot_stats.append(snap_entry)
        if snapshot_callback:
            path = snapshot_callback(view, t_elapsed, snapshot_number)
            if path:
                snapshot_files.append((t_elapsed, path))
        snapshot_number += 1

    def advance(
        rhs,
        implicit_operator,
        preconditioner,
        dt_value,
        extrapolate=False,
        phase_hs: Optional[_HeatsinkPhaseState] = None,
        bdf2_hs: bool = False,
    ):
        nonlocal temperature, previous_temperature, next_temperature, t_hs, t_hs_prev
        nonlocal current_time, step_counter
        nonlocal total_solve_time, total_iterations, retry_count
        nonlocal max_relative_residual, pout_step, balance_integral
        if extrapolate:
            np.multiply(2.0, temperature, out=guess_workspace)
            np.subtract(guess_workspace, previous_temperature, out=guess_workspace)
            t_guess = guess_workspace
        else:
            t_guess = temperature
        if phase_hs is not None:
            r_hs = phase_hs.scalar_rhs(
                ambient, t_hs, t_hs_prev, dt_value, bdf2=bdf2_hs
            )
            solve_rhs = phase_hs.modified_rhs(rhs, r_hs, out=rhs)
            initial_guess = phase_hs.warm_start_x1(t_guess, out=next_temperature)
        else:
            r_hs = 0.0
            solve_rhs = rhs
            initial_guess = t_guess
        solve_started = time.perf_counter()
        solution, iterations, retried, relative_residual = _solve_pcg(
            implicit_operator,
            preconditioner,
            solve_rhs,
            initial_guess,
            cancel_check=cancel_check,
            out=next_temperature,
            workspaces=pcg_workspaces,
        )
        elapsed = time.perf_counter() - solve_started
        if phase_hs is not None:
            solution, t_hs_new = phase_hs.recover(
                solution, r_hs, out=solution, scratch=pcg_workspaces[1]
            )
            t_hs_prev, t_hs = t_hs, t_hs_new
        old_prev = previous_temperature
        previous_temperature = temperature
        temperature = solution
        if solution is next_temperature:
            next_temperature = old_prev
        current_time += dt_value
        step_counter += 1
        total_solve_time += elapsed
        total_iterations += iterations
        retry_count += int(retried)
        max_relative_residual = max(max_relative_residual, relative_residual)
        pout_step = compute_pout(temperature)
        balance_integral += dt_value * (pin - pout_step)
        return elapsed, iterations

    use_blas_limit = (
        HAS_NATIVE_CORE
        and not bool(getattr(operator, "prefer_derivative_start", False))
    )
    blas_guard = (
        _limit_blas_threads(1) if use_blas_limit else contextlib.nullcontext()
    )

    with blas_guard:
        for phase_definition, _, phase_steps, phase_dt in phase_plan:
            if phase_steps <= 0 or aborted or early_stopped:
                continue
            phase_started = time.perf_counter()
            inv_dt = 1.0 / phase_dt
            phase_solve_time = 0.0
            phase_iterations = 0
            phase_steps_done = 0
            derivative_start = (
                bool(phase_definition.get("derivative_start"))
                or (
                    time_stepping_used == "uniform"
                    and (
                        node_count >= 250_000
                        or bool(getattr(operator, "prefer_derivative_start", False))
                    )
                )
            )

            def build_be_rhs(q_src):
                if C is None and is_compact:
                    rhs_workspace.fill(0.0)
                    operator.add_scaled_capacity(rhs_workspace, temperature, inv_dt)
                else:
                    np.multiply(C, inv_dt, out=rhs_workspace)
                    np.multiply(rhs_workspace, temperature, out=rhs_workspace)
                add_q(rhs_workspace, q_src)
                add_b(rhs_workspace)
                return rhs_workspace

            def build_bdf2_rhs(q_src):
                if C is None and is_compact:
                    rhs_workspace.fill(0.0)
                    operator.add_scaled_capacity(rhs_workspace, temperature, 2.0 * inv_dt)
                    operator.add_scaled_capacity(
                        rhs_workspace, previous_temperature, -0.5 * inv_dt
                    )
                else:
                    np.multiply(temperature, 2.0 * inv_dt, out=rhs_workspace)
                    np.multiply(previous_temperature, -0.5 * inv_dt, out=pcg_workspaces[0])
                    np.add(rhs_workspace, pcg_workspaces[0], out=rhs_workspace)
                    np.multiply(rhs_workspace, C, out=rhs_workspace)
                add_q(rhs_workspace, q_src)
                add_b(rhs_workspace)
                return rhs_workspace

            try:
                if derivative_start:
                    bdf_operator, bdf_preconditioner = operator.implicit_linear_operator(
                        C, 1.5 / phase_dt
                    )
                    phase_hs_bdf = None
                    if hs_active:
                        phase_hs_bdf, u_rel = _HeatsinkPhaseState.prepare_pcg(
                            heatsink_model,
                            1.5 / phase_dt,
                            bdf_operator,
                            bdf_preconditioner,
                            cancel_check=cancel_check,
                            rtol=1e-8,
                            workspaces=pcg_workspaces,
                        )
                        max_relative_residual = max(max_relative_residual, u_rel)
                    if Q_func is not None:
                        source = Q_func(current_time)
                        pin = _power_sum(source)
                    if is_compact:
                        operator._apply_with_scale(0.0, temperature, out=rhs_workspace)
                    else:
                        np.copyto(rhs_workspace, operator.dot(temperature))
                    rhs_workspace *= -1.0
                    add_q(rhs_workspace, source)
                    add_b(rhs_workspace)
                    if hs_active:
                        g = heatsink_model.pad_conductance
                        bottom_only = bool(getattr(heatsink_model, "bottom_layer_only", False))
                        t_dot_g = (
                            float(np.dot(g, temperature[-g.size :]))
                            if bottom_only
                            else float(np.dot(g, temperature))
                        )
                        t_hs_0 = (
                            t_hs
                            if c_hs > 0.0
                            else (heatsink_model.g_hs_amb * ambient + t_dot_g)
                            / (heatsink_model.g_hs_amb + float(np.sum(g)))
                        )
                        if bottom_only:
                            rhs_workspace[-g.size :] += g * t_hs_0
                        else:
                            rhs_workspace += g * t_hs_0
                        if c_hs > 0.0:
                            t_diff_g = (
                                float(np.dot(g, temperature[-g.size :] - t_hs))
                                if bottom_only
                                else float(np.dot(g, temperature - t_hs))
                            )
                            d_t_hs = (
                                heatsink_model.g_hs_amb * (ambient - t_hs) + t_diff_g
                            ) / c_hs
                            t_hs_prev = t_hs - phase_dt * d_t_hs
                    if C is None and is_compact:
                        operator.divide_by_capacity(rhs_workspace, out=rhs_workspace)
                    else:
                        rhs_workspace /= C
                    np.multiply(rhs_workspace, phase_dt, out=rhs_workspace)
                    np.subtract(temperature, rhs_workspace, out=previous_temperature)

                    while phase_steps_done < phase_steps and not aborted and not early_stopped:
                        if cancel_check and cancel_check():
                            aborted = True
                            break
                        if Q_func is not None:
                            source = Q_func(current_time + phase_dt)
                            pin = _power_sum(source)
                        rhs = build_bdf2_rhs(source)
                        elapsed, iterations = advance(
                            rhs,
                            bdf_operator,
                            bdf_preconditioner,
                            phase_dt,
                            extrapolate=(phase_steps_done >= 1),
                            phase_hs=phase_hs_bdf,
                            bdf2_hs=True,
                        )
                        phase_solve_time += elapsed
                        phase_iterations += iterations
                        phase_steps_done += 1
                        if (
                            step_counter % update_interval == 0
                            or step_counter == total_steps
                        ):
                            if progress_callback and not progress_callback(
                                step_counter, total_steps
                            ):
                                aborted = True
                                break
                        if config.snapshots_enabled:
                            while (
                                next_snapshot_idx < len(config.snap_times)
                                and current_time >= config.snap_times[next_snapshot_idx]
                            ):
                                record_snapshot(config.snap_times[next_snapshot_idx])
                                next_snapshot_idx += 1
                        if should_early_stop():
                            early_stopped = True
                            break
                    phase_metrics.append({
                        "phase": phase_definition["name"],
                        "dt": phase_dt,
                        "steps": phase_steps_done,
                        "assembly_s": time.perf_counter() - phase_started - phase_solve_time,
                        "factorization_s": 0.0,
                        "avg_solve_s": phase_solve_time / max(phase_steps_done, 1),
                        "avg_pcg_iterations": phase_iterations / max(phase_steps_done, 1),
                        "startup": "derivative_bdf2",
                        "backend": backend_name,
                    })
                    continue

                be_operator, be_preconditioner = operator.implicit_linear_operator(
                    C, 1.0 / phase_dt
                )
                phase_hs_be = None
                if hs_active:
                    phase_hs_be, u_rel = _HeatsinkPhaseState.prepare_pcg(
                        heatsink_model,
                        1.0 / phase_dt,
                        be_operator,
                        be_preconditioner,
                        cancel_check=cancel_check,
                        rtol=1e-8,
                        workspaces=pcg_workspaces,
                    )
                    max_relative_residual = max(max_relative_residual, u_rel)
                bdf_operator = None
                bdf_preconditioner = None
                phase_hs_bdf = None
                if phase_steps > 1:
                    bdf_operator, bdf_preconditioner = operator.implicit_linear_operator(
                        C, 1.5 / phase_dt
                    )
                    if hs_active:
                        phase_hs_bdf, u_rel = _HeatsinkPhaseState.prepare_pcg(
                            heatsink_model,
                            1.5 / phase_dt,
                            bdf_operator,
                            bdf_preconditioner,
                            cancel_check=cancel_check,
                            rtol=1e-8,
                            workspaces=pcg_workspaces,
                        )
                        max_relative_residual = max(max_relative_residual, u_rel)
                if Q_func is not None:
                    source = Q_func(current_time + phase_dt)
                    pin = _power_sum(source)
                rhs = build_be_rhs(source)
                elapsed, iterations = advance(
                    rhs,
                    be_operator,
                    be_preconditioner,
                    phase_dt,
                    extrapolate=False,
                    phase_hs=phase_hs_be,
                    bdf2_hs=False,
                )
                phase_solve_time += elapsed
                phase_iterations += iterations
                phase_steps_done += 1
                if step_counter % update_interval == 0 or step_counter == total_steps:
                    if progress_callback and not progress_callback(step_counter, total_steps):
                        aborted = True
                if config.snapshots_enabled:
                    while (
                        next_snapshot_idx < len(config.snap_times)
                        and current_time >= config.snap_times[next_snapshot_idx]
                    ):
                        record_snapshot(config.snap_times[next_snapshot_idx])
                        next_snapshot_idx += 1
                if not aborted and should_early_stop():
                    early_stopped = True

                while phase_steps_done < phase_steps and not aborted and not early_stopped:
                    if cancel_check and cancel_check():
                        aborted = True
                        break
                    if Q_func is not None:
                        source = Q_func(current_time + phase_dt)
                        pin = _power_sum(source)
                    rhs = build_bdf2_rhs(source)
                    elapsed, iterations = advance(
                        rhs,
                        bdf_operator,
                        bdf_preconditioner,
                        phase_dt,
                        extrapolate=(phase_steps_done >= 1),
                        phase_hs=phase_hs_bdf,
                        bdf2_hs=True,
                    )
                    phase_solve_time += elapsed
                    phase_iterations += iterations
                    phase_steps_done += 1

                    if step_counter % update_interval == 0 or step_counter == total_steps:
                        if progress_callback and not progress_callback(step_counter, total_steps):
                            aborted = True
                            break
                    if config.snapshots_enabled:
                        while (
                            next_snapshot_idx < len(config.snap_times)
                            and current_time >= config.snap_times[next_snapshot_idx]
                        ):
                            record_snapshot(config.snap_times[next_snapshot_idx])
                            next_snapshot_idx += 1
                    if should_early_stop():
                        early_stopped = True
                        break
            except _CancelledIterativeSolve:
                aborted = True

            phase_metrics.append({
                "phase": phase_definition["name"],
                "dt": phase_dt,
                "steps": phase_steps_done,
                "assembly_s": time.perf_counter() - phase_started - phase_solve_time,
                "factorization_s": 0.0,
                "avg_solve_s": phase_solve_time / max(phase_steps_done, 1),
                "avg_pcg_iterations": phase_iterations / max(phase_steps_done, 1),
                "backend": backend_name,
            })

    final_view = temperature.reshape(layer_count, rows, cols)
    pout_final, hs_info = _heatsink_diagnostics(
        heatsink_model if hs_active else None,
        temperature,
        t_hs,
        hA,
        ambient,
        operator=operator,
    )
    average_solve = total_solve_time / max(step_counter, 1)
    steady_rel_diff = (
        abs(pin - pout_final) / max(abs(pin), 1e-9) if early_stopped else None
    )
    k_norm_info = {
        "strategy": "matrix_free_implicit_fvm_bdf2",
        "backend": backend_name,
        "multi_phase": time_stepping_used != "uniform",
        "time_stepping": time_stepping_used,
        "N": node_count,
        "nnz_K": operator.nnz_estimate,
        "dt_base": config.dt_base,
        "steps_target": config.steps_target,
        "steps_total": step_counter,
        "factorization_s": 0.0,
        "factorizations": 0,
        "factorization_cache_hit": False,
        "avg_solve_s": average_solve,
        "avg_pcg_iterations": total_iterations / max(step_counter, 1),
        "pcg_retry_count": retry_count,
        "pcg_relative_residual_max": max_relative_residual,
        "pin_w": pin,
        "pout_final_w": pout_final,
        "steady_rel_diff": steady_rel_diff,
        "early_steady_state_stop": early_stopped,
    }
    k_norm_info.update(hs_info)
    return SolverResult(
        T=final_view,
        aborted=aborted,
        step_counter=step_counter,
        total_solve_time=total_solve_time,
        total_factor_time=0.0,
        factor_count=0,
        snapshot_stats=snapshot_stats,
        snapshot_files=snapshot_files,
        phase_metrics=phase_metrics,
        balance_history=balance_history,
        k_norm_info=k_norm_info,
    )


def run_simulation(
    config: SolverConfig,
    K: sp.csr_matrix,
    C: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    hA: np.ndarray,
    layer_count: int,
    rows: int,
    cols: int,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    snapshot_callback: Optional[Callable[[np.ndarray, float, int], str]] = None,
    Q_func: Optional[Callable[[float], np.ndarray]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    factorization_cache: Optional[Any] = None,
    factorization_cache_key: Optional[str] = None,
    heatsink_model: Optional[HeatsinkModel] = None,
) -> SolverResult:
    """
    Run the transient thermal simulation using BDF2 time integration.

    Parameters
    ----------
    config : SolverConfig
        Solver configuration.
    K : scipy.sparse.csr_matrix
        Stiffness matrix (N x N).
    C : np.ndarray
        Heat capacity per node (N,).
    Q : np.ndarray
        Heat source per node (N,).
    b : np.ndarray
        Boundary condition contribution (N,).
    hA : np.ndarray
        Heat transfer coefficient times area per node (N,).
    layer_count : int
        Number of copper layers.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    progress_callback : callable, optional
        Function(current_step, total_steps) -> bool.
        Returns False to abort simulation.
    snapshot_callback : callable, optional
        Function(T_view, t_elapsed, snap_index) -> filename.
        Called to save snapshots at specified times.
    Q_func : callable, optional
        Function(t) -> np.ndarray returning heat source vector at time t.
        When provided, Q is updated each time step for time-varying power.
        When None, the constant Q array is used throughout.
    factorization_cache : object, optional
        Single-entry cache exposing ``get(key)`` and ``put(key, value)``.
        Used only by the source-independent uniform BDF2 operator.
    factorization_cache_key : str, optional
        Complete identity for the cached operator and time-step configuration.
    heatsink_model : HeatsinkModel, optional
        Lumped 0D dynamic heatsink model coupled via exact Sherman-Morrison.

    Returns
    -------
    SolverResult
        Simulation results including final temperature and diagnostics.
    """
    if config.steady_state and Q_func is None:
        return run_steady_state(
            config=config,
            K=K,
            Q=Q,
            b=b,
            hA=hA,
            layer_count=layer_count,
            rows=rows,
            cols=cols,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            heatsink_model=heatsink_model,
        )

    N = K.shape[0]
    RC = rows * cols
    sim_time = config.sim_time
    amb = config.amb
    hs_active = _validate_heatsink_model(heatsink_model, N, plane_size=RC)
    c_hs = max(0.0, float(heatsink_model.capacity_j_k)) if hs_active else 0.0
    t_hs = (
        float(amb if heatsink_model.initial_temp_c is None else heatsink_model.initial_temp_c)
        if hs_active
        else float(amb)
    )
    t_hs_prev = t_hs

    backend = "SciPy"
    backend_modes = []

    # Initialize temperature
    Tn = np.ones(N, dtype=np.float64) * amb
    Tnm1 = Tn.copy()
    e0 = float(np.sum(C * (Tn - amb))) + (c_hs * (t_hs - amb) if hs_active else 0.0)
    balance_integral = 0.0
    pout_step = float(np.sum(hA * (Tn - amb))) + (
        heatsink_model.g_hs_amb * (t_hs - amb) if hs_active else 0.0
    )
    prev_snap_time = 0.0
    prev_snap_energy = e0
    Q_current = Q_func(0.0) if Q_func is not None else Q
    pin = _power_sum(Q_current)

    # Tracking variables
    step_counter = 0
    snap_cnt = 1
    next_snap_idx = 0
    current_time = 0.0
    aborted = False
    early_stopped = False
    total_solve_time = 0.0
    total_factor_time = 0.0
    factor_count = 0
    factorization_cache_used = False
    snapshot_stats = []
    snapshot_files = []
    phase_metrics = []
    balance_history = []

    time_stepping_used, phase_plan = _build_phase_plan(config, N)
    total_steps = sum(item[2] for item in phase_plan)

    update_interval = max(1, total_steps // 100)

    def should_early_stop():
        if (
            not config.early_stop
            or Q_func is not None
            or step_counter < 5
            or pin <= 1e-12
        ):
            return False
        if config.snapshots_enabled and next_snap_idx < len(config.snap_times):
            return False
        if abs(pin - pout_step) / pin >= 1e-4:
            return False
        max_delta = float(np.max(np.abs(Tn - Tnm1)))
        if hs_active:
            max_delta = max(max_delta, abs(t_hs - t_hs_prev))
        return max_delta < 1e-4

    def record_snapshot(t_elapsed):
        nonlocal prev_snap_time, prev_snap_energy, snap_cnt
        T_view = Tn.reshape((layer_count, rows, cols))
        max_top = float(np.max(T_view[0]))
        max_bot = float(np.max(T_view[-1]))
        delta_t = Tn - amb
        energy = float(np.sum(C * delta_t)) + (
            c_hs * (t_hs - amb) if hs_active else 0.0
        )
        dE = energy - e0
        eps_abs = abs(dE - balance_integral)
        eps_rel = eps_abs / max(abs(t_elapsed * pin), 1e-9)
        balance_warn = eps_rel > 0.01 or eps_abs > 0.01

        print(
            "[ThermalSim][EnergyBalance] "
            f"t={t_elapsed:.3f}s E={energy:.6f}J Pin={pin:.6f}W "
            f"Pout={pout_step:.6f}W eps_abs={eps_abs:.6f}J eps_rel={eps_rel:.6f}"
        )
        if balance_warn:
            print(
                "[ThermalSim][EnergyBalance][WARN] "
                f"eps_rel={eps_rel:.6f} eps_abs={eps_abs:.6f}J"
            )

        interval_t = t_elapsed - prev_snap_time
        if interval_t > 0:
            balance_history.append({
                "delta_t": interval_t,
                "dE": energy - prev_snap_energy
            })
        prev_snap_time = t_elapsed
        prev_snap_energy = energy

        snap_entry = {
            "t": t_elapsed,
            "max_top": max_top,
            "max_bottom": max_bot,
            "energy": energy,
            "pin": pin,
            "pout": pout_step,
            "eps_abs": eps_abs,
            "eps_rel": eps_rel,
            "energy_warn": balance_warn
        }
        if hs_active:
            snap_entry["heatsink_temp_c"] = float(t_hs)
        snapshot_stats.append(snap_entry)

        if snapshot_callback:
            snap_path = snapshot_callback(T_view, t_elapsed, snap_cnt)
            if snap_path:
                snapshot_files.append((t_elapsed, snap_path))
        snap_cnt += 1

    def advance_step(
        rhs,
        solver,
        dt_k,
        phase_hs: Optional[_HeatsinkPhaseState] = None,
        bdf2_hs: bool = False,
    ):
        nonlocal Tnm1, Tn, t_hs, t_hs_prev, current_time, step_counter
        nonlocal pout_step, balance_integral, total_solve_time
        if phase_hs is not None:
            r_hs = phase_hs.scalar_rhs(amb, t_hs, t_hs_prev, dt_k, bdf2=bdf2_hs)
            solve_rhs = phase_hs.modified_rhs(rhs, r_hs)
        else:
            r_hs = 0.0
            solve_rhs = rhs
        solve_start = time.perf_counter()
        Tnp1 = solver(solve_rhs)
        solve_elapsed = time.perf_counter() - solve_start
        total_solve_time += solve_elapsed
        if phase_hs is not None:
            Tnp1, t_hs_new = phase_hs.recover(Tnp1, r_hs)
            t_hs_prev, t_hs = t_hs, t_hs_new
        Tnm1, Tn = Tn, Tnp1
        current_time += dt_k
        step_counter += 1
        delta_t = Tn - amb
        pout_step = float(np.sum(hA * delta_t)) + (
            heatsink_model.g_hs_amb * (t_hs - amb) if hs_active else 0.0
        )
        balance_integral += dt_k * (pin - pout_step)
        return solve_elapsed

    # Main time-stepping loop
    for pdef, phase_time, phase_step_count, phase_dt in phase_plan:
        if phase_step_count <= 0 or aborted or early_stopped:
            continue

        # Build system matrices for this phase
        assembly_start = time.perf_counter()
        D = C / phase_dt
        two_D = 2.0 * D
        half_D = 0.5 * D
        rhs_workspace = np.empty(N, dtype=np.float64)
        A_bdf2 = K + sp.diags(1.5 * D, format="csr")
        A_be = None if pdef.get("derivative_start") else (K + sp.diags(D, format="csr"))
        assembly_time = time.perf_counter() - assembly_start

        def build_bdf2_rhs(q_value):
            """Populate the reusable BDF2 right-hand-side buffer."""
            np.multiply(two_D, Tn, out=rhs_workspace)
            rhs_workspace[...] -= half_D * Tnm1
            rhs_workspace[...] += q_value
            rhs_workspace[...] += b
            return rhs_workspace

        if pdef.get("derivative_start"):
            # A derivative-consistent fictitious T(-dt) gives BDF2 its
            # second-order startup without a separate one-shot BE matrix.
            # This is used by Auto on large uniform direct-solver systems.
            Q_current = Q_func(current_time) if Q_func is not None else Q
            pin = _power_sum(Q_current)
            if hs_active:
                g = heatsink_model.pad_conductance
                bottom_only = bool(getattr(heatsink_model, "bottom_layer_only", False))
                t_dot_g = (
                    float(np.dot(g, Tn[-g.size :]))
                    if bottom_only
                    else float(np.dot(g, Tn))
                )
                t_hs_0 = (
                    t_hs
                    if c_hs > 0.0
                    else (heatsink_model.g_hs_amb * amb + t_dot_g)
                    / (heatsink_model.g_hs_amb + float(np.sum(g)))
                )
                rhs0 = Q_current + b - K.dot(Tn)
                if bottom_only:
                    rhs0[-g.size :] += g * t_hs_0
                else:
                    rhs0 += g * t_hs_0
                derivative0 = rhs0 / C
                Tnm1 = Tn - phase_dt * derivative0
                if c_hs > 0.0:
                    t_diff_g = (
                        float(np.dot(g, Tn[-g.size :] - t_hs))
                        if bottom_only
                        else float(np.dot(g, Tn - t_hs))
                    )
                    d_t_hs = (
                        heatsink_model.g_hs_amb * (amb - t_hs) + t_diff_g
                    ) / c_hs
                    t_hs_prev = t_hs - phase_dt * d_t_hs
            else:
                derivative0 = (Q_current + b - K.dot(Tn)) / C
                Tnm1 = Tn - phase_dt * derivative0

            factor_cache_hit = False
            factor_cached = False
            cached_factor = (
                factorization_cache.get(factorization_cache_key)
                if factorization_cache is not None and factorization_cache_key
                else None
            )
            if cached_factor is not None:
                solve_bdf2 = cached_factor.solve
                bdf_owner = cached_factor.owner
                phase_backend = cached_factor.backend
                factor_time = 0.0
                factor_cache_hit = True
                factor_cached = True
                factorization_cache_used = True
            else:
                factor_start = time.perf_counter()
                solve_bdf2, bdf_owner, phase_backend, _ = _factor_linear_system(A_bdf2)
                factor_time = time.perf_counter() - factor_start
                total_factor_time += factor_time
                factor_count += 1
                if factorization_cache is not None and factorization_cache_key:
                    factorization_cache.put(
                        factorization_cache_key,
                        CachedLinearSolver(solve_bdf2, bdf_owner, phase_backend),
                    )
                    factor_cached = True
            phase_hs_bdf = (
                _HeatsinkPhaseState.prepare_direct(
                    heatsink_model, 1.5 / phase_dt, solve_bdf2, node_count=N
                )
                if hs_active
                else None
            )
            backend_modes.append(phase_backend)
            phase_solve_time = 0.0
            phase_steps_done = 0

            while not aborted and not early_stopped and phase_steps_done < phase_step_count:
                if cancel_check and cancel_check():
                    aborted = True
                    break
                if Q_func is not None:
                    Q_current = Q_func(current_time + phase_dt)
                    pin = _power_sum(Q_current)
                rhs = build_bdf2_rhs(Q_current)
                phase_solve_time += advance_step(
                    rhs,
                    solve_bdf2,
                    phase_dt,
                    phase_hs=phase_hs_bdf,
                    bdf2_hs=True,
                )
                phase_steps_done += 1

                if step_counter % update_interval == 0 or step_counter == total_steps:
                    if progress_callback and not progress_callback(step_counter, total_steps):
                        aborted = True
                        print("[ThermalSim] Simulation cancelled by user.")
                        break
                if config.snapshots_enabled:
                    while next_snap_idx < len(config.snap_times) and current_time >= config.snap_times[next_snap_idx]:
                        record_snapshot(config.snap_times[next_snap_idx])
                        next_snap_idx += 1
                if should_early_stop():
                    early_stopped = True
                    break

            if not factor_cached:
                _release_linear_solver(bdf_owner)
            del solve_bdf2, bdf_owner, A_bdf2
            phase_metrics.append({
                "phase": pdef["name"],
                "dt": phase_dt,
                "steps": phase_steps_done,
                "assembly_s": assembly_time,
                "factorization_s": factor_time,
                "avg_solve_s": phase_solve_time / max(phase_steps_done, 1),
                "startup": "derivative_bdf2",
                "backend": phase_backend,
                "factorization_cache_hit": factor_cache_hit,
            })
            if aborted or early_stopped:
                break
            continue

        phase_solve_time = 0.0
        phase_factor_time = 0.0
        phase_steps_done = 0
        phase_backend = None

        # Backward Euler bootstraps BDF2. It is a one-shot matrix, so release
        # its factorization before preparing the reusable BDF2 system.
        factor_start = time.perf_counter()
        solve_be, be_owner, be_backend, _ = _factor_linear_system(A_be)
        factor_elapsed = time.perf_counter() - factor_start
        phase_factor_time += factor_elapsed
        total_factor_time += factor_elapsed
        factor_count += 1
        backend_modes.append(be_backend)
        phase_backend = be_backend
        phase_hs_be = (
            _HeatsinkPhaseState.prepare_direct(
                heatsink_model, 1.0 / phase_dt, solve_be, node_count=N
            )
            if hs_active
            else None
        )

        if Q_func is not None:
            Q_current = Q_func(current_time + phase_dt)
            pin = _power_sum(Q_current)
        np.multiply(D, Tn, out=rhs_workspace)
        rhs_workspace += Q_current
        rhs_workspace += b
        rhs1 = rhs_workspace
        solve_elapsed = advance_step(
            rhs1,
            solve_be,
            phase_dt,
            phase_hs=phase_hs_be,
            bdf2_hs=False,
        )
        phase_solve_time += solve_elapsed
        phase_steps_done += 1

        _release_linear_solver(be_owner)
        del solve_be, be_owner, A_be

        if cancel_check and cancel_check():
            aborted = True

        if step_counter % update_interval == 0 or step_counter == total_steps:
            if progress_callback and not progress_callback(step_counter, total_steps):
                aborted = True
                print("[ThermalSim] Simulation cancelled by user.")
                break

        if config.snapshots_enabled:
            while next_snap_idx < len(config.snap_times) and current_time >= config.snap_times[next_snap_idx]:
                record_snapshot(config.snap_times[next_snap_idx])
                next_snap_idx += 1

        if not aborted and should_early_stop():
            early_stopped = True

        # Explicitly factor the BDF2 system once and reuse that exact owner.
        solve_bdf2 = None
        bdf_owner = None
        phase_hs_bdf = None
        if not aborted and not early_stopped and phase_step_count > 1:
            factor_start = time.perf_counter()
            solve_bdf2, bdf_owner, bdf_backend, _ = _factor_linear_system(A_bdf2)
            factor_elapsed = time.perf_counter() - factor_start
            phase_factor_time += factor_elapsed
            total_factor_time += factor_elapsed
            factor_count += 1
            backend_modes.append(bdf_backend)
            phase_backend = bdf_backend
            if hs_active:
                phase_hs_bdf = _HeatsinkPhaseState.prepare_direct(
                    heatsink_model, 1.5 / phase_dt, solve_bdf2, node_count=N
                )

        while not aborted and not early_stopped and phase_steps_done < phase_step_count:
            if cancel_check and cancel_check():
                aborted = True
                break
            if Q_func is not None:
                Q_current = Q_func(current_time + phase_dt)
                pin = float(np.sum(Q_current))
            rhs = build_bdf2_rhs(Q_current)
            solve_elapsed = advance_step(
                rhs,
                solve_bdf2,
                phase_dt,
                phase_hs=phase_hs_bdf,
                bdf2_hs=True,
            )
            phase_solve_time += solve_elapsed
            phase_steps_done += 1

            if step_counter % update_interval == 0 or step_counter == total_steps:
                if progress_callback and not progress_callback(step_counter, total_steps):
                    aborted = True
                    print("[ThermalSim] Simulation cancelled by user.")
                    break

            if config.snapshots_enabled:
                while next_snap_idx < len(config.snap_times) and current_time >= config.snap_times[next_snap_idx]:
                    record_snapshot(config.snap_times[next_snap_idx])
                    next_snap_idx += 1

            if should_early_stop():
                early_stopped = True
                break

            if aborted:
                break

        _release_linear_solver(bdf_owner)
        if solve_bdf2 is not None:
            del solve_bdf2
        if bdf_owner is not None:
            del bdf_owner
        del A_bdf2

        phase_avg_solve = phase_solve_time / max(phase_steps_done, 1)
        phase_metrics.append({
            "phase": pdef["name"],
            "dt": phase_dt,
            "steps": phase_steps_done,
            "assembly_s": assembly_time,
            "factorization_s": phase_factor_time,
            "avg_solve_s": phase_avg_solve,
            "backend": phase_backend,
        })

        if aborted or early_stopped:
            break

    # Compute final statistics
    avg_solve_time = total_solve_time / max(step_counter, 1)
    T = Tn.reshape((layer_count, rows, cols))
    pout_final, hs_info = _heatsink_diagnostics(
        heatsink_model if hs_active else None, Tn, t_hs, hA, amb
    )

    steady_ok = early_stopped
    rel_diff = None
    if not steady_ok and balance_history:
        recent = balance_history[-min(3, len(balance_history)):]
        steady_ok = all(
            abs(item["dE"]) / max(abs(item["delta_t"] * pin), 1e-9) < 0.01
            for item in recent
        )
    if steady_ok:
        rel_diff = abs(pin - pout_final) / max(abs(pin), 1e-9)
        print(
            "[ThermalSim][EnergyBalance][Final] "
            f"Pin={pin:.6f}W Pout={pout_final:.6f}W rel_diff={rel_diff:.6f}"
        )

    if backend_modes:
        backend = backend_modes[-1] if len(set(backend_modes)) == 1 else " -> ".join(dict.fromkeys(backend_modes))

    k_norm_info = {
        "strategy": "implicit_fvm_bdf2",
        "backend": backend,
        "multi_phase": time_stepping_used != "uniform",
        "time_stepping": time_stepping_used,
        "N": N,
        "nnz_K": int(K.nnz),
        "dt_base": config.dt_base,
        "steps_target": config.steps_target,
        "steps_total": step_counter,
        "factorization_s": total_factor_time,
        "factorizations": factor_count,
        "factorization_cache_hit": factorization_cache_used,
        "avg_solve_s": avg_solve_time,
        "pin_w": pin,
        "pout_final_w": pout_final,
        "steady_rel_diff": rel_diff,
        "early_steady_state_stop": early_stopped,
    }
    k_norm_info.update(hs_info)

    return SolverResult(
        T=T,
        aborted=aborted,
        step_counter=step_counter,
        total_solve_time=total_solve_time,
        total_factor_time=total_factor_time,
        factor_count=factor_count,
        snapshot_stats=snapshot_stats,
        snapshot_files=snapshot_files,
        phase_metrics=phase_metrics,
        balance_history=balance_history,
        k_norm_info=k_norm_info
    )
