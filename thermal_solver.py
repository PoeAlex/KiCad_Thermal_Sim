"""
Thermal solver using finite volume method with BDF2 time integration.

This module implements the core numerical solver for transient thermal
simulation, including sparse matrix assembly and time-stepping.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .capabilities import HAS_PARDISO

if HAS_PARDISO:
    import pypardiso


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
    use_pardiso : bool
        Whether to use PyPardiso solver if available.
    use_multi_phase : bool
        Whether to use multi-phase time stepping.
    snapshots_enabled : bool
        Whether to capture time-series snapshots.
    snap_times : List[float]
        Times at which to capture snapshots.
    """
    sim_time: float
    amb: float
    dt_base: float
    steps_target: int
    use_pardiso: bool = False
    use_multi_phase: bool = True
    snapshots_enabled: bool = False
    snap_times: List[float] = None

    def __post_init__(self):
        if self.snap_times is None:
            self.snap_times = []


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

    rows_list = []
    cols_list = []
    data_list = []

    col_right = np.arange(cols - 1)[None, :]
    row_all = np.arange(rows)[:, None]
    col_all = np.arange(cols)[None, :]
    row_down = np.arange(rows - 1)[:, None]

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
        rows_list.extend([i_idx, j_idx, i_idx, j_idx])
        cols_list.extend([i_idx, j_idx, j_idx, i_idx])
        data_list.extend([g, g, -g, -g])

        # Y-direction coupling
        k_h = 2.0 * k_layer[:-1, :] * k_layer[1:, :] / (k_layer[:-1, :] + k_layer[1:, :] + eps)
        t_edge = 0.5 * (t_layer[:-1, :] + t_layer[1:, :])
        Gy = k_h * (t_edge * dx) / dy

        idx_up = base + row_down * cols + col_all
        idx_down = idx_up + cols

        g = Gy.ravel()
        i_idx = idx_up.ravel()
        j_idx = idx_down.ravel()
        rows_list.extend([i_idx, j_idx, i_idx, j_idx])
        cols_list.extend([i_idx, j_idx, j_idx, i_idx])
        data_list.extend([g, g, -g, -g])

    # Inter-layer coupling through vias
    if layer_count > 1 and gap_m:
        V_enh = np.clip(V_map, 1.0, 50.0)
        plane_idx = np.arange(RC, dtype=np.int64)
        for l in range(layer_count - 1):
            gap_val = max(gap_m[l], 1e-6)
            Gz_base = k_fr4 * pixel_area / gap_val
            Gz = (Gz_base * V_enh).ravel()
            i_idx = l * RC + plane_idx
            j_idx = (l + 1) * RC + plane_idx
            rows_list.extend([i_idx, j_idx, i_idx, j_idx])
            cols_list.extend([i_idx, j_idx, j_idx, i_idx])
            data_list.extend([Gz, Gz, -Gz, -Gz])

    K_base = sp.coo_matrix(
        (np.concatenate(data_list), (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(N, N),
        dtype=np.float64
    ).tocsr()

    # Boundary conditions
    diag_extra = np.zeros(N, dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    # Top surface convection
    h_top = 10.0
    diag_add_top = h_top * pixel_area
    top_idx = np.arange(RC, dtype=np.int64)
    diag_extra[top_idx] += diag_add_top
    b[top_idx] += diag_add_top * amb

    # Bottom surface: air convection + heatsink
    h_air_bot = 10.0
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
    Q_func: Optional[Callable[[float], np.ndarray]] = None
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

    Returns
    -------
    SolverResult
        Simulation results including final temperature and diagnostics.
    """
    N = K.shape[0]
    RC = rows * cols
    dt = config.dt_base
    sim_time = config.sim_time
    amb = config.amb

    use_pardiso = config.use_pardiso and HAS_PARDISO
    backend = "PARDISO" if use_pardiso else "SciPy"

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
    total_solve_time = 0.0
    total_factor_time = 0.0
    factor_count = 0
    snapshot_stats = []
    snapshot_files = []
    phase_metrics = []
    balance_history = []

    # Set up phase plan
    if config.use_multi_phase:
        phase_defs = [
            {"name": "A", "frac": 0.08, "dt_scale": 0.5},
            {"name": "B", "frac": 0.35, "dt_scale": 1.0},
            {"name": "C", "frac": 0.57, "dt_scale": 2.0},
        ]
        phase_times = [sim_time * p["frac"] for p in phase_defs]
        phase_times[-1] = sim_time - sum(phase_times[:-1])
        phase_steps = []
        phase_dts = []
        for phase_time, pdef in zip(phase_times, phase_defs):
            phase_dt = dt * pdef["dt_scale"]
            step_count = max(1, int(round(phase_time / phase_dt)))
            phase_dt = phase_time / step_count if phase_time > 0 else dt
            phase_steps.append(step_count)
            phase_dts.append(phase_dt)
        total_steps = sum(phase_steps)
        phase_plan = list(zip(phase_defs, phase_times, phase_steps, phase_dts))
    else:
        total_steps = config.steps_target
        phase_plan = [({"name": "single"}, sim_time, total_steps, dt)]

    update_interval = max(1, total_steps // 100)

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
        if phase_step_count <= 0:
            continue

        # Build system matrices for this phase
        assembly_start = time.perf_counter()
        D = C / phase_dt
        A_be = K + sp.diags(D, format="csr")
        A_bdf2 = K + sp.diags(1.5 * D, format="csr")
        assembly_time = time.perf_counter() - assembly_start

        # Factor matrices
        factor_start = time.perf_counter()
        if use_pardiso and hasattr(pypardiso, "factorized"):
            solve_be = pypardiso.factorized(A_be.tocsc())
            solve_bdf2 = pypardiso.factorized(A_bdf2.tocsc())
        elif use_pardiso:
            solve_be = lambda rhs: pypardiso.spsolve(A_be.tocsc(), rhs)
            solve_bdf2 = lambda rhs: pypardiso.spsolve(A_bdf2.tocsc(), rhs)
        else:
            lu_be = spla.splu(A_be.tocsc())
            lu_bdf2 = spla.splu(A_bdf2.tocsc())
            solve_be = lu_be.solve
            solve_bdf2 = lu_bdf2.solve
        factor_time = time.perf_counter() - factor_start
        total_factor_time += factor_time
        factor_count += 1

        phase_solve_time = 0.0
        phase_steps_done = 0

        # First step: Backward Euler
        if Q_func is not None:
            Q_current = Q_func(current_time + phase_dt)
            pin = float(np.sum(Q_current))
        rhs1 = D * Tn + Q_current + b
        solve_elapsed = advance_step(rhs1, solve_be, phase_dt)
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

        # Remaining steps: BDF2
        while phase_steps_done < phase_step_count:
            if Q_func is not None:
                Q_current = Q_func(current_time + phase_dt)
                pin = float(np.sum(Q_current))
            rhs = (2.0 * D) * Tn - (0.5 * D) * Tnm1 + Q_current + b
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

            if aborted:
                break

        phase_avg_solve = phase_solve_time / max(phase_steps_done, 1)
        phase_metrics.append({
            "phase": pdef["name"],
            "dt": phase_dt,
            "steps": phase_steps_done,
            "assembly_s": assembly_time,
            "factorization_s": factor_time,
            "avg_solve_s": phase_avg_solve
        })

        if aborted:
            break

    # Compute final statistics
    avg_solve_time = total_solve_time / max(step_counter, 1)
    T = Tn.reshape((layer_count, rows, cols))
    delta_t_final = Tn - amb
    pout_final = float(np.sum(hA * delta_t_final))

    steady_ok = False
    rel_diff = None
    if balance_history:
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

    k_norm_info = {
        "strategy": "implicit_fvm_bdf2",
        "backend": backend,
        "multi_phase": config.use_multi_phase,
        "N": N,
        "nnz_K": int(K.nnz),
        "dt_base": config.dt_base,
        "steps_target": config.steps_target,
        "steps_total": step_counter,
        "factorization_s": total_factor_time,
        "factorizations": factor_count,
        "avg_solve_s": avg_solve_time,
        "pin_w": pin,
        "pout_final_w": pout_final,
        "steady_rel_diff": rel_diff
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
