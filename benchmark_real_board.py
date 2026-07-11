"""Read-only ThermalSim benchmark using KiCad's real Python environment."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("THERMALSIM_HEADLESS", "1")
PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLUGIN_DIR.parent))

import numpy as np
import pcbnew

from KiCad_Thermal_Sim.capabilities import HAS_PARDISO
from KiCad_Thermal_Sim.geometry_mapper import build_geometry_state
from KiCad_Thermal_Sim.stackup_parser import parse_stackup_from_board_file
from KiCad_Thermal_Sim.thermal_solver import SolverConfig, build_stiffness_matrix, run_simulation
from KiCad_Thermal_Sim.workflow import GridEstimate


def _timed(func, *args, **kwargs):
    start = time.perf_counter()
    value = func(*args, **kwargs)
    return value, time.perf_counter() - start


def _enabled_copper_layers(board):
    """Return KiCad-enabled copper layers when a board has no saved stackup."""
    enabled = board.GetEnabledLayers()
    layers = []
    for layer_id in range(64):
        try:
            is_copper = pcbnew.IsCopperLayer(layer_id)
        except Exception:
            is_copper = layer_id < 32
        if is_copper and enabled.Contains(layer_id):
            layers.append(layer_id)
    try:
        return sorted(layers, key=lambda layer_id: int(pcbnew.CopperLayerToOrdinal(layer_id)))
    except Exception:
        return layers


def _estimate_solver_grid(bbox, requested_res, settings, layer_count, _pads):
    width = bbox.GetWidth() * 1e-6
    height = bbox.GetHeight() * 1e-6
    estimated = (width / requested_res) * (height / requested_res)
    coarsened = estimated > 200000
    actual = (width * height / 100000.0) ** 0.5 if coarsened else requested_res
    return GridEstimate(
        requested_res, actual, bbox.GetX() * 1e-6, bbox.GetY() * 1e-6,
        width, height, int(height / actual) + 4, int(width / actual) + 4,
        layer_count, coarsened, False, 200000, 100000,
    )


def _single_pad_power(board, copper_ids, pad, grid, total_nodes, watts):
    q = np.zeros(total_nodes, dtype=np.float64)
    bbox = pad.GetBoundingBox()
    x0, y0 = bbox.GetX() * 1e-6, bbox.GetY() * 1e-6
    width, height = bbox.GetWidth() * 1e-6, bbox.GetHeight() * 1e-6
    cs = max(0, int((x0 - grid.x_min_mm) / grid.actual_res_mm))
    rs = max(0, int((y0 - grid.y_min_mm) / grid.actual_res_mm))
    ce = min(grid.cols, int((x0 + width - grid.x_min_mm) / grid.actual_res_mm) + 1)
    re = min(grid.rows, int((y0 + height - grid.y_min_mm) / grid.actual_res_mm) + 1)
    layer = copper_ids.index(pad.GetLayer()) if pad.GetLayer() in copper_ids else 0
    indices = []
    for row in range(rs, re):
        for col in range(cs, ce):
            indices.append(layer * grid.base_cells + row * grid.cols + col)
    if indices:
        q[np.asarray(indices, dtype=np.int64)] = float(watts) / len(indices)
    return q


def _capacity_and_thickness(board, copper_ids, stack, settings, state, grid):
    copper_by_id = {
        item.get("layer_id"): item.get("thickness_mm")
        for item in stack.get("copper", [])
        if isinstance(item.get("thickness_mm"), (int, float))
    }
    copper_thickness = [float(copper_by_id.get(layer, 0.035)) for layer in copper_ids]
    gaps_mm = list(stack.get("dielectric_gaps_mm") or [])
    if len(gaps_mm) != max(0, len(copper_ids) - 1):
        gaps_mm = [float(settings["thick"]) / max(1, len(copper_ids) - 1)] * max(0, len(copper_ids) - 1)
    derived = {
        "copper_thickness_mm_used": copper_thickness,
        "gap_mm_used": gaps_mm,
        "total_thick_mm_used": float(stack.get("board_thickness_mm") or settings["thick"]),
    }
    t_cu = np.asarray([max(1e-9, value * 1e-3) for value in derived["copper_thickness_mm_used"]])
    gaps = [max(1e-9, value * 1e-3) for value in derived["gap_mm_used"]]
    layer_count = len(copper_ids)
    if layer_count > 1 and gaps:
        t_fr4 = [
            gaps[0] if idx == 0 else gaps[-1] if idx == layer_count - 1
            else 0.5 * (gaps[idx - 1] + gaps[idx])
            for idx in range(layer_count)
        ]
    else:
        t_fr4 = [max(derived["total_thick_mm_used"] * 1e-3, 1e-5)] * layer_count
    t_fr4 = np.clip(np.asarray(t_fr4), 1e-6, 5e-3)
    area = (grid.actual_res_mm * 1e-3) ** 2
    capacity = np.empty_like(state.copper_mask, dtype=np.float64)
    for idx in range(layer_count):
        mask = state.copper_mask[idx]
        v_cu = area * t_cu[idx]
        v_fr4 = area * t_fr4[idx]
        capacity[idx] = np.where(mask, 8960.0 * 385.0 * v_cu, 1850.0 * 1100.0 * v_fr4)
        capacity[idx] += mask * (1850.0 * 1100.0 * v_fr4)
    return capacity.reshape(-1), t_cu, t_fr4, gaps


def benchmark(args):
    board_path = Path(args.board).resolve()
    board, load_s = _timed(pcbnew.LoadBoard, str(board_path))
    stack, stackup_s = _timed(parse_stackup_from_board_file, board)
    copper_ids = list(stack.get("copper_ids") or [])
    stackup_fallback = False
    if not copper_ids:
        copper_ids = _enabled_copper_layers(board)
        stack = dict(stack) if isinstance(stack, dict) else {}
        stack["copper_ids"] = copper_ids
        stackup_fallback = True
    if not copper_ids:
        raise ValueError("Board has no enabled copper layers.")
    bbox = board.GetBoundingBox()
    pads = [pad for footprint in board.Footprints() for pad in footprint.Pads()]
    settings = {
        "res": args.resolution,
        "time": args.time,
        "amb": 25.0,
        "thick": 1.6,
        "h_conv": 10.0,
        "pad_th": 1.0,
        "pad_k": 3.0,
        "pad_cap_areal": 0.0,
        "ignore_traces": False,
        "ignore_polygons": False,
        "limit_area": False,
        "use_heatsink": True,
    }
    grid = _estimate_solver_grid(bbox, args.resolution, settings, len(copper_ids), pads)
    geometry_args = dict(
        board=board, copper_ids=copper_ids, rows=grid.rows, cols=grid.cols,
        x_min=grid.x_min_mm, y_min=grid.y_min_mm, res=grid.actual_res_mm,
        settings=settings, via_factor=390.0 / 0.3, pads_list=pads,
    )
    state, geometry_s = _timed(build_geometry_state, **geometry_args)
    _, repeated_lookup_s = _timed(lambda: state)

    result = {
        "board": str(board_path),
        "kiCad_version": pcbnew.GetBuildVersion(),
        "scenario": args.scenario,
        "stackup_fallback": stackup_fallback,
        "grid": {
            "requested_res_mm": args.resolution,
            "actual_res_mm": grid.actual_res_mm,
            "rows": grid.rows,
            "cols": grid.cols,
            "layers": len(copper_ids),
            "nodes": grid.nodes,
        },
        "timings": {
            "board_load_s": load_s,
            "stackup_parse_s": stackup_s,
            "geometry_maps_s": geometry_s,
            "repeated_cache_lookup_s": repeated_lookup_s,
        },
    }

    if args.scenario in {"thermal", "all"}:
        (capacity, t_cu, t_fr4, gaps), capacity_s = _timed(
            _capacity_and_thickness, board, copper_ids, stack, settings, state, grid
        )
        dx = grid.actual_res_mm * 1e-3
        (matrix, boundary, h_area, _), matrix_s = _timed(
            build_stiffness_matrix,
            len(copper_ids), grid.rows, grid.cols, state.copper_mask,
            t_cu, t_fr4, 390.0, 0.3, dx, dx, state.via_map, gaps,
            state.heatsink_mask.astype(np.float64), settings, 25.0,
        )
        power_pad = next((pad for fp in board.Footprints() if fp.GetReference() == "U1" for pad in fp.Pads()), pads[0])
        power = _single_pad_power(board, copper_ids, power_pad, grid, grid.nodes, 2.0)
        power_func = None
        steps = max(1, min(600, max(80, int(120 * (args.time ** 0.35)))))
        config = SolverConfig(
            sim_time=args.time,
            amb=25.0,
            dt_base=args.time / steps,
            steps_target=steps,
            use_pardiso=HAS_PARDISO and args.backend != "scipy",
            use_multi_phase=True,
            time_stepping=args.time_stepping,
        )
        solved, solver_s = _timed(
            run_simulation, config, matrix, capacity, power, boundary, h_area,
            len(copper_ids), grid.rows, grid.cols, Q_func=power_func,
        )
        result["timings"].update({
            "capacity_build_s": capacity_s,
            "stiffness_matrix_s": matrix_s,
            "factorization_s": solved.total_factor_time,
            "linear_solve_s": solved.total_solve_time,
            "solver_total_s": solver_s,
        })
        result["solver"] = {
            "backend": solved.k_norm_info.get("backend"),
            "time_stepping": solved.k_norm_info.get("time_stepping"),
            "steps": solved.step_counter,
            "factorizations": solved.factor_count,
            "max_temperature_c": float(np.max(solved.T)),
            "phase_metrics": solved.phase_metrics,
        }
        if args.compare_reference and args.time_stepping != "multi_phase":
            reference_config = SolverConfig(
                sim_time=args.time,
                amb=25.0,
                dt_base=args.time / steps,
                steps_target=steps,
                use_pardiso=HAS_PARDISO and args.backend != "scipy",
                use_multi_phase=True,
                time_stepping="multi_phase",
            )
            reference, reference_s = _timed(
                run_simulation, reference_config, matrix, capacity, power, boundary, h_area,
                len(copper_ids), grid.rows, grid.cols, Q_func=power_func,
            )
            delta = np.abs(solved.T - reference.T)
            result["reference_comparison"] = {
                "reference_solver_s": reference_s,
                "max_abs_delta_c": float(np.max(delta)),
                "mean_abs_delta_c": float(np.mean(delta)),
                "reference_max_temperature_c": float(np.max(reference.T)),
            }

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", help="Path to a .kicad_pcb file; it is never saved")
    parser.add_argument("--scenario", choices=("preview", "thermal", "all"), default="all")
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--backend", choices=("auto", "scipy", "pardiso"), default="auto")
    parser.add_argument(
        "--time-stepping", choices=("auto", "multi_phase", "two_phase", "uniform"), default="auto"
    )
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--compare-reference", action="store_true",
        help="Also run the legacy 3-phase plan and report cell-wise temperature deltas",
    )
    args = parser.parse_args()
    result = benchmark(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
