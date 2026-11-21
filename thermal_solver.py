"""
Thermal solver using finite volume method with BDF2 time integration.

This module implements the core numerical solver for transient thermal
simulation, including sparse matrix assembly and time-stepping.
"""

import contextlib
import ctypes
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
        damped_jacobi_native,
        residual_restrict_2x2_native,
        prolong_add_2x2_native,
    )
except ImportError:
    damped_jacobi_native = None
    residual_restrict_2x2_native = None
    prolong_add_2x2_native = None


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


def _restrict_2x2(array: np.ndarray) -> np.ndarray:
    """Conservatively sum 2x2 spatial blocks on a ``(layers, rows, cols)`` array."""
    layers, rows, cols = array.shape
    coarse_rows = (rows + 1) // 2
    coarse_cols = (cols + 1) // 2
    even_rows = (rows // 2) * 2
    if rows == even_rows:
        row_sum = array[:, 0::2, :] + array[:, 1::2, :]
    else:
        row_sum = np.empty((layers, coarse_rows, cols), dtype=np.float64)
        if even_rows > 0:
            row_sum[:, :-1, :] = (
                array[:, 0:even_rows:2, :] + array[:, 1:even_rows:2, :]
            )
        row_sum[:, -1, :] = array[:, -1, :]

    even_cols = (cols // 2) * 2
    if cols == even_cols:
        return row_sum[:, :, 0::2] + row_sum[:, :, 1::2]
    out = np.empty((layers, coarse_rows, coarse_cols), dtype=np.float64)
    if even_cols > 0:
        out[:, :, :-1] = (
            row_sum[:, :, 0:even_cols:2] + row_sum[:, :, 1:even_cols:2]
        )
    out[:, :, -1] = row_sum[:, :, -1]
    return out


def _coarsen_structured_operator(
    operator: "StructuredThermalOperator",
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
            gx_c = np.empty((layers, coarse_rows, coarse_cols - 1), dtype=np.float64)
            if even_rows > 0:
                gx_c[:, :-1, :] = (
                    gx_odd[:, 0:even_rows:2, :] + gx_odd[:, 1:even_rows:2, :]
                )
            gx_c[:, -1, :] = gx_odd[:, -1, :]
    else:
        gx_c = np.empty((layers, coarse_rows, 0), dtype=np.float64)

    if coarse_rows > 1:
        gy_odd = operator.gy[:, 1::2, :]
        if cols == even_cols:
            gy_c = gy_odd[:, :, 0::2] + gy_odd[:, :, 1::2]
        else:
            gy_c = np.empty((layers, coarse_rows - 1, coarse_cols), dtype=np.float64)
            if even_cols > 0:
                gy_c[:, :, :-1] = (
                    gy_odd[:, :, 0:even_cols:2] + gy_odd[:, :, 1:even_cols:2]
                )
            gy_c[:, :, -1] = gy_odd[:, :, -1]
    else:
        gy_c = np.empty((layers, 0, coarse_cols), dtype=np.float64)

    if layers > 1 and operator.gz.size:
        gz_c = _restrict_2x2(operator.gz)
    else:
        gz_c = np.empty((0, coarse_rows, coarse_cols), dtype=np.float64)

    boundary_c = _restrict_2x2(
        operator.boundary_diag.reshape(layers, rows, cols)
    ).reshape(-1)
    diagonal_view = boundary_c.reshape(layers, coarse_rows, coarse_cols).copy()
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
        gx=np.ascontiguousarray(gx_c, dtype=np.float64),
        gy=np.ascontiguousarray(gy_c, dtype=np.float64),
        gz=np.ascontiguousarray(gz_c, dtype=np.float64),
        boundary_diag=np.ascontiguousarray(boundary_c, dtype=np.float64),
        stiffness_diag=np.ascontiguousarray(diagonal_view.reshape(-1), dtype=np.float64),
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
    diagonal = operator.stiffness_diag + scaled_capacity

    row_parts = [np.arange(node_count, dtype=np.int64)]
    col_parts = [np.arange(node_count, dtype=np.int64)]
    data_parts = [diagonal]

    if cols > 1:
        layer_base = np.arange(layers, dtype=np.int64)[:, None, None] * plane_cells
        row_base = np.arange(rows, dtype=np.int64)[None, :, None] * cols
        col_base = np.arange(cols - 1, dtype=np.int64)[None, None, :]
        left = (layer_base + row_base + col_base).reshape(-1)
        right = left + 1
        conductance = -operator.gx.reshape(-1)
        row_parts.extend((left, right))
        col_parts.extend((right, left))
        data_parts.extend((conductance, conductance))

    if rows > 1:
        layer_base = np.arange(layers, dtype=np.int64)[:, None, None] * plane_cells
        row_base = np.arange(rows - 1, dtype=np.int64)[None, :, None] * cols
        col_base = np.arange(cols, dtype=np.int64)[None, None, :]
        upper = (layer_base + row_base + col_base).reshape(-1)
        lower = upper + cols
        conductance = -operator.gy.reshape(-1)
        row_parts.extend((upper, lower))
        col_parts.extend((lower, upper))
        data_parts.extend((conductance, conductance))

    if layers > 1 and operator.gz.size:
        plane_idx = np.arange(plane_cells, dtype=np.int64)[None, :]
        top = (
            np.arange(layers - 1, dtype=np.int64)[:, None] * plane_cells + plane_idx
        ).reshape(-1)
        bottom = top + plane_cells
        conductance = -operator.gz.reshape(-1)
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


class _StructuredGeometricMultigrid:
    """Symmetric V-cycle geometric multigrid preconditioner for regular 3D grids."""

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
        self.diag_masses: List[np.ndarray] = []
        self.omega_inv_diags: List[np.ndarray] = []

        scaled_capacity = np.ascontiguousarray(
            float(derivative_scale) * capacity, dtype=np.float64
        ).reshape(-1)

        for level_idx, level_op in enumerate(self.operators):
            if level_idx > 0:
                prev_op = self.operators[level_idx - 1]
                scaled_capacity = np.ascontiguousarray(
                    _restrict_2x2(
                        scaled_capacity.reshape(
                            prev_op.layer_count, prev_op.rows, prev_op.cols
                        )
                    ).reshape(-1),
                    dtype=np.float64,
                )
            diag_mass = np.ascontiguousarray(
                level_op.boundary_diag + scaled_capacity, dtype=np.float64
            )
            diagonal = level_op.stiffness_diag + scaled_capacity
            inv_diag = np.divide(
                1.0,
                diagonal,
                out=np.zeros_like(diagonal),
                where=np.abs(diagonal) > 1e-30,
            )
            self.diag_masses.append(diag_mass)
            self.omega_inv_diags.append(
                np.ascontiguousarray(self.omega * inv_diag, dtype=np.float64)
            )

        coarse_matrix = _assemble_structured_csc(self.operators[-1], scaled_capacity)
        try:
            self.coarse_solver = spla.factorized(coarse_matrix)
        except Exception:
            coarse_diag = self.operators[-1].stiffness_diag + scaled_capacity
            coarse_inv = np.divide(
                1.0,
                coarse_diag,
                out=np.zeros_like(coarse_diag),
                where=np.abs(coarse_diag) > 1e-30,
            )
            self.coarse_solver = lambda rhs, _inv=coarse_inv: _inv * rhs

    def _cycle(self, level: int, rhs: np.ndarray) -> np.ndarray:
        if level == len(self.operators) - 1:
            return np.ascontiguousarray(
                self.coarse_solver(rhs), dtype=np.float64
            ).reshape(-1)

        op = self.operators[level]
        diag_mass = self.diag_masses[level]
        omega_inv = self.omega_inv_diags[level]

        # Pre-smoothing starting from zero avoids an unnecessary matvec on step 1.
        value = None
        if damped_jacobi_native is not None:
            value = damped_jacobi_native(
                op.layer_count,
                op.rows,
                op.cols,
                op.gx,
                op.gy,
                op.gz,
                diag_mass,
                omega_inv,
                rhs,
                None,
                self.smooth_steps,
                zero_init=True,
            )
        if value is None:
            value = omega_inv * rhs
            for _ in range(self.smooth_steps - 1):
                residual = rhs - op._apply_with_diagonal(diag_mass, value)
                residual *= omega_inv
                value += residual

        coarse_rhs = None
        if residual_restrict_2x2_native is not None:
            coarse_rhs = residual_restrict_2x2_native(
                op.layer_count,
                op.rows,
                op.cols,
                op.gx,
                op.gy,
                op.gz,
                diag_mass,
                rhs,
                value,
            )
        if coarse_rhs is None:
            residual = rhs - op._apply_with_diagonal(diag_mass, value)
            coarse_rhs = _restrict_2x2(
                residual.reshape(op.layer_count, op.rows, op.cols)
            ).reshape(-1)

        coarse_correction = self._cycle(level + 1, coarse_rhs)

        prolonged = False
        if prolong_add_2x2_native is not None:
            prolonged = bool(
                prolong_add_2x2_native(
                    op.layer_count,
                    op.rows,
                    op.cols,
                    coarse_correction,
                    value,
                )
            )
        if not prolonged:
            coarse_op = self.operators[level + 1]
            coarse_view = coarse_correction.reshape(
                coarse_op.layer_count, coarse_op.rows, coarse_op.cols
            )
            value_view = value.reshape(op.layer_count, op.rows, op.cols)
            value_view[:, 0::2, 0::2] += coarse_view
            if op.cols > 1:
                value_view[:, 0::2, 1::2] += coarse_view[:, :, : op.cols // 2]
            if op.rows > 1:
                value_view[:, 1::2, 0::2] += coarse_view[:, : op.rows // 2, :]
            if op.rows > 1 and op.cols > 1:
                value_view[:, 1::2, 1::2] += coarse_view[
                    :, : op.rows // 2, : op.cols // 2
                ]

        post_value = None
        if damped_jacobi_native is not None:
            post_value = damped_jacobi_native(
                op.layer_count,
                op.rows,
                op.cols,
                op.gx,
                op.gy,
                op.gz,
                diag_mass,
                omega_inv,
                rhs,
                value,
                self.smooth_steps,
                zero_init=False,
            )
        if post_value is not None:
            return post_value

        for _ in range(self.smooth_steps):
            residual = rhs - op._apply_with_diagonal(diag_mass, value)
            residual *= omega_inv
            value += residual
        return value

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Apply one symmetric V-cycle."""
        return self._cycle(
            0, np.ascontiguousarray(rhs, dtype=np.float64).reshape(-1)
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
        """Build and cache the coarsened stiffness operator hierarchy."""
        if self._coarse_operators is not None:
            return self._coarse_operators
        levels = [self]
        current = self
        for level_idx in range(max_levels - 1):
            if level_idx > 0 and current.shape[0] <= coarse_max_nodes:
                break
            if current.rows <= 2 and current.cols <= 2:
                break
            current = _coarsen_structured_operator(current)
            levels.append(current)
        self._coarse_operators = levels
        return levels

    def implicit_linear_operator(self, capacity, derivative_scale):
        """Return ``K + derivative_scale * C`` as a SciPy LinearOperator."""
        capacity = np.ascontiguousarray(capacity, dtype=np.float64).reshape(-1)
        scale = float(derivative_scale)
        scaled_capacity = scale * capacity
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
            return operator, preconditioner

        cached = self._preconditioner_cache.get(scale)
        if cached is not None and np.array_equal(cached[0], capacity):
            multigrid = cached[1]
        else:
            multigrid = _StructuredGeometricMultigrid(
                self._get_multigrid_operators(),
                capacity,
                scale,
            )
            if len(self._preconditioner_cache) >= 4:
                self._preconditioner_cache.clear()
            self._preconditioner_cache[scale] = (capacity.copy(), multigrid)

        preconditioner = spla.LinearOperator(
            shape,
            matvec=multigrid.solve,
            dtype=np.float64,
        )
        return operator, preconditioner


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

    h_air_bottom = float(settings.get("h_conv", 10.0))
    pad_thickness = max(0.0001, settings["pad_th"] * 1e-3)
    h_sink = (settings["pad_k"] / pad_thickness) * 0.2
    bottom_effective = (
        (1.0 - H_map) * h_air_bottom + H_map * h_sink
    )
    bottom_addition = (bottom_effective * pixel_area).reshape(-1)
    bottom_indices = (layer_count - 1) * plane_cells + plane_indices
    boundary_diag[bottom_indices] += bottom_addition
    boundary_rhs[bottom_indices] += bottom_addition * amb
    h_area[bottom_indices] += bottom_addition

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
    h_air_bot = float(settings.get('h_conv', 10.0))
    pad_thick_m = max(0.0001, settings['pad_th'] * 1e-3)
    pad_k = settings['pad_k']
    contact_factor = 0.2
    h_sink = (pad_k / pad_thick_m) * contact_factor
    h_bot_eff = (1.0 - H_map) * h_air_bot + H_map * h_sink
    diag_add_bot = (h_bot_eff * pixel_area).ravel()
    bot_base = (layer_count - 1) * RC
    bot_idx = bot_base + top_idx
    diag_extra[bot_idx] += diag_add_bot
    b[bot_idx] += diag_add_bot * amb

    hA = np.zeros(N, dtype=np.float64)
    hA[top_idx] += diag_add_top
    hA[bot_idx] += diag_add_bot

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
):
    """Solve one implicit step with preconditioned conjugate gradients."""
    iterations = 0

    def callback(_value):
        nonlocal iterations
        iterations += 1
        if cancel_check and cancel_check():
            raise _CancelledIterativeSolve()

    solution, info = spla.cg(
        operator,
        rhs,
        x0=initial,
        M=preconditioner,
        rtol=relative_tolerance,
        atol=0.0,
        maxiter=maxiter,
        callback=callback,
    )
    retried = False
    if info > 0:
        retried = True
        solution, retry_info = spla.cg(
            operator,
            rhs,
            x0=solution,
            M=preconditioner,
            rtol=relative_tolerance,
            atol=0.0,
            maxiter=retry_maxiter,
            callback=callback,
        )
        info = retry_info
    if info < 0:
        raise RuntimeError(f"PCG failed with solver status {info}.")
    if info > 0:
        raise RuntimeError(
            f"PCG did not converge after {iterations} iterations "
            f"(relative tolerance {relative_tolerance:g})."
        )
    residual = operator.matvec(solution) - rhs
    relative_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-30)
    )
    return solution, iterations, retried, relative_residual


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
) -> SolverResult:
    """Solve the steady-state thermal system ``K * T = Q + b`` directly."""
    N = K.shape[0]
    amb = config.amb
    pin = float(np.sum(Q))

    if cancel_check and cancel_check():
        T_abort = np.full((layer_count, rows, cols), amb, dtype=np.float64)
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
            k_norm_info={
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
            },
        )

    rhs = np.asarray(Q, dtype=np.float64) + np.asarray(b, dtype=np.float64)
    factor_start = time.perf_counter()
    solve_fn, owner, backend, _ = _factor_linear_system(K)
    factor_time = time.perf_counter() - factor_start
    try:
        solve_start = time.perf_counter()
        Tn = np.asarray(solve_fn(rhs), dtype=np.float64).reshape(-1)
        solve_time = time.perf_counter() - solve_start
    finally:
        _release_linear_solver(owner)

    aborted = False
    if progress_callback and not progress_callback(1, 1):
        aborted = True

    T = Tn.reshape((layer_count, rows, cols))
    pout_final = float(np.sum(hA * (Tn - amb)))
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
    C: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    hA: np.ndarray,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> SolverResult:
    """Solve ``K * T = Q + b`` directly using a matrix-free or adaptive operator."""
    layer_count = operator.layer_count
    rows = operator.rows
    cols = operator.cols
    node_count = operator.shape[0]
    amb = config.amb
    pin = float(np.sum(Q))
    if bool(getattr(operator, "prefer_derivative_start", False)):
        backend_name = "Adaptive-Multigrid-PCG"
    elif HAS_NATIVE_CORE:
        backend_name = "Native-MatrixFree-PCG"
    else:
        backend_name = "MatrixFree-PCG"

    if cancel_check and cancel_check():
        T_abort = np.full((layer_count, rows, cols), amb, dtype=np.float64)
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
            k_norm_info={
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
            },
        )

    capacity = (
        np.zeros(node_count, dtype=np.float64)
        if C is None
        else np.asarray(C, dtype=np.float64)
    )
    rhs = np.asarray(Q, dtype=np.float64) + np.asarray(b, dtype=np.float64)
    initial = np.full(node_count, amb, dtype=np.float64)

    setup_start = time.perf_counter()
    aborted = False
    step_counter = 0
    iterations = 0
    retried = False
    relative_residual = 0.0
    solve_time = 0.0
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
        setup_time = time.perf_counter() - setup_start

        solve_start = time.perf_counter()
        try:
            temperature, iterations, retried, relative_residual = _solve_pcg(
                steady_operator,
                steady_preconditioner,
                rhs,
                initial,
                cancel_check=cancel_check,
                relative_tolerance=1e-7,
                maxiter=250,
                retry_maxiter=500,
            )
            solve_time = time.perf_counter() - solve_start
            step_counter = 1
        except _CancelledIterativeSolve:
            solve_time = time.perf_counter() - solve_start
            aborted = True

    if not aborted and progress_callback and not progress_callback(1, 1):
        aborted = True

    final_view = temperature.reshape(layer_count, rows, cols)
    pout_final = float(np.sum(hA * (temperature - amb)))
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
    C: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    hA: np.ndarray,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    snapshot_callback: Optional[Callable[[np.ndarray, float, int], str]] = None,
    Q_func: Optional[Callable[[float], np.ndarray]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
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
        )

    layer_count = operator.layer_count
    rows = operator.rows
    cols = operator.cols
    node_count = operator.shape[0]
    ambient = config.amb
    temperature = np.full(node_count, ambient, dtype=np.float64)
    previous_temperature = temperature.copy()
    guess_workspace = np.empty(node_count, dtype=np.float64)
    initial_energy = float(np.sum(C * (temperature - ambient)))
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
    pout_step = 0.0
    source = Q_func(0.0) if Q_func is not None else Q
    pin = float(np.sum(source))
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
        return float(np.max(np.abs(temperature - previous_temperature))) < 1e-4

    def record_snapshot(t_elapsed):
        nonlocal previous_snapshot_time, previous_snapshot_energy, snapshot_number
        view = temperature.reshape(layer_count, rows, cols)
        energy = float(np.sum(C * (temperature - ambient)))
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
        snapshot_stats.append({
            "t": t_elapsed,
            "max_top": float(np.max(view[0])),
            "max_bottom": float(np.max(view[-1])),
            "energy": energy,
            "pin": pin,
            "pout": pout_step,
            "eps_abs": absolute_error,
            "eps_rel": relative_error,
            "energy_warn": relative_error > 0.01 or absolute_error > 0.01,
        })
        if snapshot_callback:
            path = snapshot_callback(view, t_elapsed, snapshot_number)
            if path:
                snapshot_files.append((t_elapsed, path))
        snapshot_number += 1

    def advance(rhs, implicit_operator, preconditioner, dt_value, extrapolate=False):
        nonlocal temperature, previous_temperature, current_time, step_counter
        nonlocal total_solve_time, total_iterations, retry_count
        nonlocal max_relative_residual, pout_step, balance_integral
        if extrapolate:
            np.multiply(2.0, temperature, out=guess_workspace)
            np.subtract(guess_workspace, previous_temperature, out=guess_workspace)
            initial_guess = guess_workspace
        else:
            initial_guess = temperature
        solve_started = time.perf_counter()
        solution, iterations, retried, relative_residual = _solve_pcg(
            implicit_operator,
            preconditioner,
            rhs,
            initial_guess,
            cancel_check=cancel_check,
        )
        elapsed = time.perf_counter() - solve_started
        previous_temperature, temperature = temperature, solution
        current_time += dt_value
        step_counter += 1
        total_solve_time += elapsed
        total_iterations += iterations
        retry_count += int(retried)
        max_relative_residual = max(max_relative_residual, relative_residual)
        pout_step = float(np.sum(hA * (temperature - ambient)))
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
            derivative = C / phase_dt
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

            try:
                if derivative_start:
                    bdf_operator, bdf_preconditioner = operator.implicit_linear_operator(
                        C, 1.5 / phase_dt
                    )
                    if Q_func is not None:
                        source = Q_func(current_time)
                        pin = float(np.sum(source))
                    initial_derivative = (
                        source + b - operator.dot(temperature)
                    ) / C
                    previous_temperature = temperature - phase_dt * initial_derivative
                    while phase_steps_done < phase_steps and not aborted and not early_stopped:
                        if cancel_check and cancel_check():
                            aborted = True
                            break
                        if Q_func is not None:
                            source = Q_func(current_time + phase_dt)
                            pin = float(np.sum(source))
                        rhs = (
                            2.0 * derivative * temperature
                            - 0.5 * derivative * previous_temperature
                            + source
                            + b
                        )
                        elapsed, iterations = advance(
                            rhs,
                            bdf_operator,
                            bdf_preconditioner,
                            phase_dt,
                            extrapolate=(phase_steps_done >= 1),
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
                bdf_operator, bdf_preconditioner = operator.implicit_linear_operator(
                    C, 1.5 / phase_dt
                )
                if Q_func is not None:
                    source = Q_func(current_time + phase_dt)
                    pin = float(np.sum(source))
                rhs = derivative * temperature + source + b
                elapsed, iterations = advance(
                    rhs, be_operator, be_preconditioner, phase_dt, extrapolate=False
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
                        pin = float(np.sum(source))
                    rhs = (
                        2.0 * derivative * temperature
                        - 0.5 * derivative * previous_temperature
                        + source
                        + b
                    )
                    elapsed, iterations = advance(
                        rhs,
                        bdf_operator,
                        bdf_preconditioner,
                        phase_dt,
                        extrapolate=(phase_steps_done >= 1),
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
    pout_final = float(np.sum(hA * (temperature - ambient)))
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
        )

    N = K.shape[0]
    RC = rows * cols
    sim_time = config.sim_time
    amb = config.amb

    backend = "SciPy"
    backend_modes = []

    # Initialize temperature
    Tn = np.ones(N, dtype=np.float64) * amb
    Tnm1 = Tn.copy()
    e0 = float(np.sum(C * (Tn - amb)))
    balance_integral = 0.0
    pout_step = float(np.sum(hA * (Tn - amb)))
    prev_snap_time = 0.0
    prev_snap_energy = e0
    Q_current = Q_func(0.0) if Q_func is not None else Q
    pin = float(np.sum(Q_current))

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
        return float(np.max(np.abs(Tn - Tnm1))) < 1e-4

    def record_snapshot(t_elapsed):
        nonlocal prev_snap_time, prev_snap_energy, snap_cnt
        T_view = Tn.reshape((layer_count, rows, cols))
        max_top = float(np.max(T_view[0]))
        max_bot = float(np.max(T_view[-1]))
        delta_t = Tn - amb
        energy = float(np.sum(C * delta_t))
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

        snapshot_stats.append({
            "t": t_elapsed,
            "max_top": max_top,
            "max_bottom": max_bot,
            "energy": energy,
            "pin": pin,
            "pout": pout_step,
            "eps_abs": eps_abs,
            "eps_rel": eps_rel,
            "energy_warn": balance_warn
        })

        if snapshot_callback:
            snap_path = snapshot_callback(T_view, t_elapsed, snap_cnt)
            if snap_path:
                snapshot_files.append((t_elapsed, snap_path))
        snap_cnt += 1

    def advance_step(rhs, solver, dt_k):
        nonlocal Tnm1, Tn, current_time, step_counter, pout_step, balance_integral, total_solve_time
        solve_start = time.perf_counter()
        Tnp1 = solver(rhs)
        solve_elapsed = time.perf_counter() - solve_start
        total_solve_time += solve_elapsed
        Tnm1, Tn = Tn, Tnp1
        current_time += dt_k
        step_counter += 1
        delta_t = Tn - amb
        pout_step = float(np.sum(hA * delta_t))
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
            pin = float(np.sum(Q_current))
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
            backend_modes.append(phase_backend)
            phase_solve_time = 0.0
            phase_steps_done = 0

            while not aborted and not early_stopped and phase_steps_done < phase_step_count:
                if cancel_check and cancel_check():
                    aborted = True
                    break
                if Q_func is not None:
                    Q_current = Q_func(current_time + phase_dt)
                    pin = float(np.sum(Q_current))
                rhs = build_bdf2_rhs(Q_current)
                phase_solve_time += advance_step(rhs, solve_bdf2, phase_dt)
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

        if Q_func is not None:
            Q_current = Q_func(current_time + phase_dt)
            pin = float(np.sum(Q_current))
        np.multiply(D, Tn, out=rhs_workspace)
        rhs_workspace += Q_current
        rhs_workspace += b
        rhs1 = rhs_workspace
        solve_elapsed = advance_step(rhs1, solve_be, phase_dt)
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
        if not aborted and not early_stopped and phase_step_count > 1:
            factor_start = time.perf_counter()
            solve_bdf2, bdf_owner, bdf_backend, _ = _factor_linear_system(A_bdf2)
            factor_elapsed = time.perf_counter() - factor_start
            phase_factor_time += factor_elapsed
            total_factor_time += factor_elapsed
            factor_count += 1
            backend_modes.append(bdf_backend)
            phase_backend = bdf_backend

        while not aborted and not early_stopped and phase_steps_done < phase_step_count:
            if cancel_check and cancel_check():
                aborted = True
                break
            if Q_func is not None:
                Q_current = Q_func(current_time + phase_dt)
                pin = float(np.sum(Q_current))
            rhs = build_bdf2_rhs(Q_current)
            solve_elapsed = advance_step(rhs, solve_bdf2, phase_dt)
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
    delta_t_final = Tn - amb
    pout_final = float(np.sum(hA * delta_t_final))

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
