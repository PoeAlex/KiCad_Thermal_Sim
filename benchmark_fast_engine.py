"""Synthetic benchmark for ThermalSim's large-grid CPU engine.

The default scenario creates an equivalent ten-million-node multilayer PCB
without requiring a proprietary board file. It measures structured-operator
construction, adaptive reduction, and optional transient solve time. Legacy
CSR assembly is run only below a configurable safety threshold.
"""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import time

PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLUGIN_DIR.parent))
try:
    import pcbnew  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PLUGIN_DIR / "tests"))
    from mocks.pcbnew_mock import install_mock
    from mocks.wx_mock import install_wx_mock

    install_mock()
    install_wx_mock()

import numpy as np

from KiCad_Thermal_Sim.adaptive_mesh import build_adaptive_mesh, build_adaptive_system
from KiCad_Thermal_Sim.thermal_solver import (
    SolverConfig,
    build_stiffness_matrix,
    build_structured_operator,
    run_simulation,
    run_simulation_matrix_free,
)


def _working_set_bytes():
    """Return the current Windows process working set, when available."""
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.argtypes = []
    get_process.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    ]
    get_memory.restype = ctypes.c_int
    process = get_process()
    if not get_memory(
        process, ctypes.byref(counters), counters.cb
    ):
        return None
    return int(counters.WorkingSetSize)


def synthetic_geometry(total_nodes=10_000_000, layers=4):
    """Create deterministic dense routing, planes, vias, and source masks."""
    cells_per_layer = max(1, int(math.ceil(total_nodes / max(layers, 1))))
    rows = max(8, int(math.sqrt(cells_per_layer * 0.7)))
    cols = max(8, int(math.ceil(cells_per_layer / rows)))
    yy, xx = np.ogrid[:rows, :cols]
    copper = np.zeros((layers, rows, cols), dtype=bool)
    for layer_idx in range(layers):
        horizontal = np.mod(yy + 11 * layer_idx, 53) < 2
        vertical = np.mod(xx + 17 * layer_idx, 67) < 2
        diagonal = np.mod(xx + yy * (layer_idx + 1), 109) < 2
        plane = (
            (xx > cols // 6)
            & (xx < 5 * cols // 6)
            & (yy > rows // 5)
            & (yy < 4 * rows // 5)
        )
        holes = ((xx - cols // 2) ** 2 + (yy - rows // 2) ** 2) < (
            min(rows, cols) // 9
        ) ** 2
        copper[layer_idx] = horizontal | vertical | diagonal | (plane & ~holes)

    via_map = np.ones((rows, cols), dtype=np.float64)
    via_map[
        (np.mod(yy, 64) < 2) & (np.mod(xx, 64) < 2)
    ] = 50.0
    heatsink = np.zeros((rows, cols), dtype=bool)
    heatsink[
        rows // 3:2 * rows // 3,
        cols // 3:2 * cols // 3,
    ] = True
    source = np.zeros((layers, rows, cols), dtype=bool)
    source[
        0,
        rows // 2 - 3:rows // 2 + 4,
        cols // 2 - 3:cols // 2 + 4,
    ] = True
    return copper, via_map, heatsink, source


def _timed(function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def benchmark(args):
    copper, via_map, heatsink, source_mask = synthetic_geometry(
        args.nodes, args.layers
    )
    layers, rows, cols = copper.shape
    node_count = copper.size
    settings = {
        "h_conv": 10.0,
        "pad_th": 1.0,
        "pad_k": 3.0,
    }
    t_cu = np.full(layers, 35e-6)
    t_fr4 = np.full(layers, 1.6e-3 / max(layers, 1))
    gaps = [1.6e-3 / max(layers - 1, 1)] * max(0, layers - 1)
    dx = dy = args.resolution_mm * 1e-3
    common = (
        layers, rows, cols, copper, t_cu, t_fr4,
        390.0, 0.3, dx, dy, via_map, gaps,
        heatsink.astype(np.float64), settings, 25.0,
    )
    memory_start = _working_set_bytes()
    (structured, boundary, h_area, _), structured_s = _timed(
        build_structured_operator, *common
    )
    memory_structured = _working_set_bytes()

    pixel_area = dx * dy
    capacity_layers = np.empty_like(copper, dtype=np.float64)
    for layer_idx in range(layers):
        cu_capacity = 8960.0 * 385.0 * pixel_area * t_cu[layer_idx]
        fr4_capacity = 1850.0 * 1100.0 * pixel_area * t_fr4[layer_idx]
        capacity_layers[layer_idx] = np.where(
            copper[layer_idx],
            cu_capacity + fr4_capacity,
            fr4_capacity,
        )
    capacity = capacity_layers.reshape(-1)
    power = np.zeros(node_count, dtype=np.float64)
    source_indices = np.flatnonzero(source_mask.reshape(-1))
    power[source_indices] = 10.0 / max(source_indices.size, 1)

    mesh, mesh_s = _timed(
        build_adaptive_mesh,
        copper,
        via_map,
        heatsink,
        source_mask,
        args.max_cell_ratio,
    )
    adaptive, reduction_s = _timed(
        build_adaptive_system,
        structured,
        capacity,
        power,
        boundary,
        h_area,
        mesh,
    )
    memory_adaptive = _working_set_bytes()

    result = {
        "grid": {
            "rows": rows,
            "cols": cols,
            "layers": layers,
            "equivalent_uniform_nodes": node_count,
            "adaptive_nodes": adaptive.operator.shape[0],
            "reduction_ratio": node_count / adaptive.operator.shape[0],
        },
        "timings": {
            "structured_operator_s": structured_s,
            "adaptive_mesh_s": mesh_s,
            "adaptive_operator_s": reduction_s,
        },
        "memory_mb": {
            "start": memory_start / 1024 ** 2 if memory_start else None,
            "structured": memory_structured / 1024 ** 2 if memory_structured else None,
            "adaptive": memory_adaptive / 1024 ** 2 if memory_adaptive else None,
        },
    }

    if args.solve_steps > 0:
        config = SolverConfig(
            sim_time=float(args.solve_steps),
            amb=25.0,
            dt_base=1.0,
            steps_target=args.solve_steps,
            use_multi_phase=False,
            time_stepping="uniform",
        )
        solved, solve_s = _timed(
            run_simulation_matrix_free,
            config,
            adaptive.operator,
            adaptive.capacity,
            adaptive.power,
            adaptive.boundary_rhs,
            adaptive.h_area,
        )
        result["timings"]["adaptive_solver_s"] = solve_s
        result["solver"] = {
            "steps": solved.step_counter,
            "avg_pcg_iterations": solved.k_norm_info["avg_pcg_iterations"],
            "max_temperature_c": float(np.max(solved.T)),
        }

    if node_count <= args.legacy_max_nodes:
        (legacy_matrix, legacy_b, legacy_h, _), legacy_build_s = _timed(
            build_stiffness_matrix, *common
        )
        result["timings"]["legacy_matrix_build_s"] = legacy_build_s
        if args.solve_steps > 0:
            legacy_config = SolverConfig(
                sim_time=float(args.solve_steps),
                amb=25.0,
                dt_base=1.0,
                steps_target=args.solve_steps,
                use_multi_phase=False,
                time_stepping="uniform",
            )
            _, legacy_solver_s = _timed(
                run_simulation,
                legacy_config,
                legacy_matrix,
                capacity,
                power,
                legacy_b,
                legacy_h,
                layers,
                rows,
                cols,
            )
            result["timings"]["legacy_solver_s"] = legacy_solver_s
    else:
        result["legacy_skipped"] = (
            f"{node_count:,} nodes exceeds the "
            f"{args.legacy_max_nodes:,}-node safety limit"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=10_000_000)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--resolution-mm", type=float, default=0.1)
    parser.add_argument("--max-cell-ratio", type=int, default=8)
    parser.add_argument("--solve-steps", type=int, default=3)
    parser.add_argument("--legacy-max-nodes", type=int, default=1_000_000)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = benchmark(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
