"""Headless Python API and semantic summary engine for ThermalSim.

This module provides a pure-headless, JSON-serializable interface to inspect
KiCad PCBs, validate simulation jobs, execute steady-state or transient
thermal simulations, run parameter sweeps, and compare simulation runs.

No GUI or ``wx`` modules are imported here.
"""

from dataclasses import asdict
import fnmatch
import inspect
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pcbnew

from .capabilities import (
    HAS_MATPLOTLIB,
    HAS_NATIVE_COMPACT,
    HAS_NATIVE_CORE,
    HAS_NATIVE_MG,
    HAS_NUMPY,
    HAS_SCIPY,
)
from .electrical_solver import (
    CurrentTerminal,
    ElectricalConfig,
    net_key_from_obj,
    net_key_from_values,
    solve_electrical_heating,
)
try:
    from .electrical_solver import EquipotentialBus
except ImportError:  # pragma: no cover - optional during staged updates
    EquipotentialBus = None
from .geometry_mapper import (
    build_geometry_state,
    get_pad_copper_layer_indices,
    get_pad_grid_mask,
    get_pad_pixels,
    get_pad_power_indices,
    get_pad_primary_layer_id,
)
from .native_core import NATIVE_CORE, clear_native_scratch
from .pwl_parser import parse_pwl_file, validate_pwl
from .stackup_parser import parse_stackup_from_board_file
from .thermal_solver import (
    HEATSINK_MATERIAL_CP,
    HeatsinkModel,
    SolverConfig,
    build_heatsink_model,
    build_stiffness_matrix,
    build_structured_operator,
    compute_heatsink_capacity_j_k,
    resolve_heatsink_cp as _resolve_heatsink_cp_from_settings,
    run_simulation,
    run_simulation_matrix_free,
)
try:
    from .thermal_solver import (
        CompactStructuredThermalOperator,
        SparsePowerVector,
        build_compact_structured_operator,
    )
except ImportError:  # pragma: no cover - optional in older builds
    CompactStructuredThermalOperator = None
    SparsePowerVector = None
    build_compact_structured_operator = None

from .workflow import (
    AreaEstimate,
    GridEstimate,
    get_process_memory_mb,
)

from .simulation_core import (
    DEFAULT_GRID_DETAIL,
    DEFAULT_GRID_MAX_CELLS,
    DEFAULT_GRID_NODE_BUDGET,
    DEFAULT_GRID_TARGET_CELLS,
    GRID_DETAIL_PRESETS,
    MAX_CUSTOM_GRID_NODES,
    SparsePadContribution,
    SparsePadContribution as _SparsePadContribution,
    auto_fast_engine_threshold,
    bbox_bounds_mm as _bbox_bounds_mm,
    build_compact_heatsink_model,
    build_dense_capacity as _build_dense_capacity,
    build_power_vector as _build_power_vector,
    build_sparse_pad_contributions as _build_sparse_pad_contributions,
    build_sparse_power_vector,
    calculate_suggested_resolution as _calculate_suggested_resolution,
    call_with_supported_kwargs,
    coarsen_grid_resolution as _coarsen_grid_resolution,
    derive_stackup_thicknesses as _derive_stackup_thicknesses,
    effective_fr4_control_volume_thicknesses as _effective_fr4_control_volume_thicknesses,
    estimate_simulation_area as _estimate_simulation_area,
    estimate_solver_cost,
    estimate_solver_grid,
    normalize_via_fill,
    pad_target_layer_index as _pad_target_layer_index,
    resolve_convection_settings,
    resolve_grid_policy as _resolve_grid_policy,
    resolve_mesh_mode,
    resolve_via_settings,
)

SCHEMA_VERSION = "1.0"
PLUGIN_VERSION = "0.5.0"

SUPPORTED_SWEEP_PARAMS = frozenset({
    "heatsink_rth_kw",
    "heatsink_mass_g",
    "heatsink_cp_j_kgk",
    "pad_k",
    "pad_th",
    "pad_contact_factor",
    "pad_sink_temp_c",
    "h_conv",
    "h_conv_bottom",
    "via_plating_um",
    "amb",
    "time",
    "power_scale",
    "copper_thickness_scale",
})

_THERMAL_PAD_RE = re.compile(r"^(EP|PAD|TAB|SLUG|TH)\d*$")
_ORIG_BUILD_STRUCTURED_OPERATOR = build_structured_operator
_ORIG_BUILD_HEATSINK_MODEL = build_heatsink_model


def _auto_fast_engine_threshold() -> int:
    return auto_fast_engine_threshold(HAS_NATIVE_MG)


def _resolve_mesh_mode(settings: Dict[str, Any], use_fast_engine: bool, node_count: int) -> str:
    return resolve_mesh_mode(settings, use_fast_engine, node_count, has_native_mg=HAS_NATIVE_MG)


def _estimate_solver_cost(nodes: int, settings: Dict[str, Any]) -> Tuple[int, int, str]:
    return estimate_solver_cost(
        nodes,
        settings,
        has_native_mg=HAS_NATIVE_MG,
        has_native_compact=HAS_NATIVE_COMPACT,
    )


def _estimate_solver_grid(
    bbox: Any,
    requested_res: float,
    settings: Dict[str, Any],
    layer_count: int,
    focus_pads: Optional[List[Any]] = None,
    area: Optional[AreaEstimate] = None,
) -> GridEstimate:
    return estimate_solver_grid(
        bbox,
        requested_res,
        settings,
        layer_count,
        focus_pads=focus_pads,
        area=area,
        has_native_mg=HAS_NATIVE_MG,
        has_native_compact=HAS_NATIVE_COMPACT,
    )


def _build_sparse_power_vector(
    pad_sources: List[Tuple[str, Any]],
    pad_contributions: List[SparsePadContribution],
    total_nodes: int,
) -> Any:
    return build_sparse_power_vector(
        pad_sources,
        pad_contributions,
        total_nodes,
        sparse_power_cls=SparsePowerVector,
    )


def _build_compact_heatsink_model(
    dx: float,
    dy: float,
    heatsink_mask: np.ndarray,
    settings: Dict[str, Any],
    amb: float,
) -> Optional[HeatsinkModel]:
    return build_compact_heatsink_model(
        dx,
        dy,
        heatsink_mask,
        settings,
        amb,
        heatsink_model_cls=HeatsinkModel,
        capacity_fn=compute_heatsink_capacity_j_k,
    )


def _json_clean(value: Any) -> Any:
    """Recursively convert values into strictly JSON-serializable Python types."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return fval
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    return str(value)


def _load_board_if_needed(board_or_path: Any) -> Tuple[Any, str]:
    """Return ``(board, board_path)`` from a file path or a loaded board object."""
    if isinstance(board_or_path, (str, Path)):
        path_str = str(board_or_path)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Board file not found: {path_str}")
        if not hasattr(pcbnew, "LoadBoard"):
            raise RuntimeError("pcbnew.LoadBoard is not available.")
        board = pcbnew.LoadBoard(path_str)
        if board is None:
            raise RuntimeError(f"pcbnew.LoadBoard returned None for '{path_str}'.")
        return board, path_str
    if board_or_path is None:
        raise ValueError("board_or_path must not be None.")
    try:
        board_path = str(board_or_path.GetFileName() or "")
    except Exception:
        board_path = ""
    return board_or_path, board_path


def _is_copper_layer(layer_id: int) -> bool:
    """Return True if ``layer_id`` is a KiCad copper layer."""
    try:
        return bool(pcbnew.IsCopperLayer(layer_id))
    except Exception:
        return 0 <= int(layer_id) < 32


def _copper_sort_key(board: Any, lid: int) -> Tuple[int, int]:
    """Order copper layer IDs from top (F.Cu) to bottom (B.Cu)."""
    try:
        return (0, int(pcbnew.CopperLayerToOrdinal(lid)))
    except Exception:
        try:
            name = str(board.GetLayerName(lid))
        except Exception:
            name = ""
        if name == "F.Cu":
            return (1, -1000)
        if name == "B.Cu":
            return (1, 1000)
        match = re.match(r"In(\d+)\.Cu", name)
        return (1, int(match.group(1)) if match else int(lid))


def _detect_copper_layers(board: Any) -> Tuple[List[int], List[str]]:
    """Detect enabled copper layers in stackup order without requiring GUI/wx."""
    copper_ids: List[int] = []
    if hasattr(board, "GetEnabledLayers"):
        enabled_layers = board.GetEnabledLayers()
        for lid in range(64):
            if _is_copper_layer(lid) and enabled_layers.Contains(lid):
                copper_ids.append(lid)
    elif hasattr(board, "_layer_names") and isinstance(board._layer_names, dict):
        for lid in board._layer_names:
            if _is_copper_layer(int(lid)):
                copper_ids.append(int(lid))
    else:
        for lid in range(32):
            try:
                lname = str(board.GetLayerName(lid))
            except Exception:
                continue
            if lname and not lname.startswith("Layer_") and _is_copper_layer(lid):
                copper_ids.append(lid)

    if not copper_ids:
        copper_ids = [int(getattr(pcbnew, "F_Cu", 0)), int(getattr(pcbnew, "B_Cu", 31))]

    copper_ids = sorted(set(copper_ids), key=lambda lid: _copper_sort_key(board, lid))
    layer_names = []
    for lid in copper_ids:
        try:
            layer_names.append(str(board.GetLayerName(lid)))
        except Exception:
            layer_names.append(f"Layer_{lid}")
    return copper_ids, layer_names


def _resolve_layers_and_stackup(board: Any) -> Tuple[List[int], List[str], Dict[str, Any]]:
    """Detect copper layers and parse saved stackup information."""
    copper_ids, layer_names = _detect_copper_layers(board)
    stack_info = parse_stackup_from_board_file(board)
    if isinstance(stack_info, dict) and stack_info.get("copper_ids"):
        copper_ids_stack = [int(x) for x in stack_info["copper_ids"]]
        detected_names = dict(zip(copper_ids, layer_names))
        valid_stack_ids = [lid for lid in copper_ids_stack if lid in detected_names]
        if len(valid_stack_ids) >= 2:
            copper_ids = valid_stack_ids
            layer_names = [detected_names[lid] for lid in copper_ids]
        elif not hasattr(board, "GetEnabledLayers") and len(copper_ids_stack) >= 2:
            copper_ids = copper_ids_stack
            layer_names = [str(board.GetLayerName(lid)) for lid in copper_ids]
    return copper_ids, layer_names, (stack_info if isinstance(stack_info, dict) else {})


def _get_board_bbox(board: Any) -> Any:
    """Return a KiCad EDA_RECT bounding box for ``board`` with mock fallback."""
    if hasattr(board, "GetBoundingBox"):
        try:
            bbox = board.GetBoundingBox()
            if bbox is not None and bbox.GetWidth() > 0 and bbox.GetHeight() > 0:
                return bbox
        except Exception:
            pass
    if hasattr(board, "ComputeBoundingBox"):
        try:
            bbox = board.ComputeBoundingBox(True)
            if bbox is not None and bbox.GetWidth() > 0 and bbox.GetHeight() > 0:
                return bbox
        except Exception:
            pass

    xs: List[int] = []
    ys: List[int] = []
    x1s: List[int] = []
    y1s: List[int] = []

    def _collect(item: Any) -> None:
        try:
            rect = item.GetBoundingBox()
            w = int(rect.GetWidth())
            h = int(rect.GetHeight())
            x = int(rect.GetX())
            y = int(rect.GetY())
            if w > 0 or h > 0:
                xs.append(x)
                ys.append(y)
                x1s.append(x + max(0, w))
                y1s.append(y + max(0, h))
        except Exception:
            pass

    try:
        fps = board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints()
        for fp in fps:
            for pad in fp.Pads():
                _collect(pad)
    except Exception:
        pass
    for getters in (("Tracks", "GetTracks"), ("Zones", "GetZones"), ("GetDrawings",)):
        for getter in getters:
            if hasattr(board, getter):
                try:
                    for item in getattr(board, getter)():
                        _collect(item)
                except Exception:
                    pass
                break

    if xs and ys and x1s and y1s:
        x0 = min(xs)
        y0 = min(ys)
        return pcbnew.EDA_RECT(x0, y0, max(1_000_000, max(x1s) - x0), max(1_000_000, max(y1s) - y0))
    return pcbnew.EDA_RECT(0, 0, 50_000_000, 50_000_000)


def _pad_net_name(pad: Any) -> str:
    """Return the net name for a KiCad pad."""
    try:
        net = pad.GetNetname()
    except Exception:
        try:
            net = pad.GetNet().GetNetname()
        except Exception:
            net = ""
    return str(net or "")


def _pad_net_code(pad: Any) -> int:
    """Return the net code for a KiCad pad."""
    try:
        return int(pad.GetNetCode())
    except Exception:
        try:
            return int(pad.GetNet().GetNetCode())
        except Exception:
            return 0


def _pad_layer_name(board: Any, pad: Any, copper_ids: Optional[List[int]] = None) -> str:
    """Return the layer name for a KiCad pad."""
    lid = get_pad_primary_layer_id(pad, copper_ids or None)
    try:
        return str(board.GetLayerName(lid))
    except Exception:
        return str(lid)


def _pad_dimensions_mm(pad: Any) -> Tuple[float, float]:
    """Return ``(width_mm, height_mm)`` for a pad."""
    if hasattr(pad, "GetSize"):
        try:
            sz = pad.GetSize()
            return abs(float(sz.x)) * 1e-6, abs(float(sz.y)) * 1e-6
        except Exception:
            pass
    try:
        bbox = pad.GetBoundingBox()
        return abs(float(bbox.GetWidth())) * 1e-6, abs(float(bbox.GetHeight())) * 1e-6
    except Exception:
        return 0.0, 0.0


def _is_thermal_name(pad_number: str) -> bool:
    """Return True if a pad number matches common exposed/thermal pad names."""
    token = str(pad_number or "").strip().upper()
    if not token:
        return False
    if token == "0" or "THERMAL" in token:
        return True
    return bool(_THERMAL_PAD_RE.match(token))


def _count_zones(board: Any) -> Tuple[int, int]:
    """Return ``(total_copper_zones, unfilled_copper_zones)``."""
    try:
        zones = list(board.Zones() if hasattr(board, "Zones") else board.GetZones())
    except Exception:
        return 0, 0
    total = 0
    unfilled = 0
    for zone in zones:
        try:
            if bool(getattr(zone, "GetIsRuleArea", lambda: False)()):
                continue
            if bool(getattr(zone, "GetIsKeepout", lambda: False)()):
                continue
            total += 1
            if hasattr(zone, "IsFilled") and not zone.IsFilled():
                unfilled += 1
        except Exception:
            continue
    return total, unfilled


def _count_eco1_shapes(board: Any) -> int:
    """Count drawings and zones present on ``User.Eco1``."""
    eco1 = getattr(pcbnew, "Eco1_User", None)
    if eco1 is None:
        return 0
    eco1_id = int(eco1)
    count = 0
    try:
        drawings = list(board.GetDrawings() if hasattr(board, "GetDrawings") else [])
        for drawing in drawings:
            try:
                if int(drawing.GetLayer()) == eco1_id:
                    count += 1
            except Exception:
                continue
    except Exception:
        pass
    try:
        zones = list(board.Zones() if hasattr(board, "Zones") else board.GetZones())
        for zone in zones:
            try:
                if hasattr(zone, "GetLayerSet") and zone.GetLayerSet().Contains(eco1_id):
                    count += 1
                elif int(zone.GetLayer()) == eco1_id:
                    count += 1
            except Exception:
                continue
    except Exception:
        pass
    return count


def _get_capabilities_dict() -> Dict[str, Any]:
    """Return headless runtime capability metadata."""
    core_version = None
    if NATIVE_CORE is not None:
        try:
            core_version = int(NATIVE_CORE.version)
        except Exception:
            core_version = None
    return {
        "has_libs": bool(HAS_NUMPY and HAS_SCIPY and HAS_MATPLOTLIB),
        "native_core": bool(HAS_NATIVE_CORE),
        "native_mg": bool(HAS_NATIVE_MG),
        "native_compact": bool(HAS_NATIVE_COMPACT),
        "core_version": core_version,
    }


def _footprint_meta(
    board: Any,
    fp: Any,
    pads: List[Any],
    copper_ids: Optional[List[int]] = None,
) -> Tuple[str, str, str, str, float, float]:
    """Extract ``(ref, value, footprint_name, layer_name, x_mm, y_mm)`` safely."""
    try:
        ref = str(fp.GetReference() or "")
    except Exception:
        ref = ""
    try:
        value = str(fp.GetValue() or "") if hasattr(fp, "GetValue") else ""
    except Exception:
        value = ""
    fp_name = ""
    if hasattr(fp, "GetFPID"):
        try:
            fpid = fp.GetFPID()
            if hasattr(fpid, "GetUniStringLibItemName"):
                fp_name = str(fpid.GetUniStringLibItemName() or "")
            elif hasattr(fpid, "GetLibItemName"):
                fp_name = str(fpid.GetLibItemName() or "")
        except Exception:
            fp_name = ""
    layer_name = ""
    if hasattr(fp, "GetLayer"):
        try:
            layer_name = str(board.GetLayerName(fp.GetLayer()))
        except Exception:
            layer_name = ""
    if not layer_name and pads:
        layer_name = _pad_layer_name(board, pads[0], copper_ids=copper_ids)
    if not layer_name:
        layer_name = "F.Cu"

    x_mm, y_mm = 0.0, 0.0
    if hasattr(fp, "GetPosition"):
        try:
            pos = fp.GetPosition()
            x_mm = float(pos.x) * 1e-6
            y_mm = float(pos.y) * 1e-6
            return ref, value, fp_name, layer_name, x_mm, y_mm
        except Exception:
            pass
    if pads:
        xs: List[float] = []
        ys: List[float] = []
        for pad in pads:
            try:
                pos = pad.GetPosition()
                xs.append(float(pos.x) * 1e-6)
                ys.append(float(pos.y) * 1e-6)
            except Exception:
                continue
        if xs and ys:
            x_mm = sum(xs) / len(xs)
            y_mm = sum(ys) / len(ys)
    return ref, value, fp_name, layer_name, x_mm, y_mm


def inspect_board(board_or_path: Any) -> Dict[str, Any]:
    """Inspect a KiCad PCB and return a JSON-serializable semantic summary.

    Parameters
    ----------
    board_or_path : str or pathlib.Path or pcbnew.BOARD
        Path to a ``.kicad_pcb`` file or an already-loaded board object.

    Returns
    -------
    dict
        Strictly JSON-serializable board metadata, stackup, footprints, pads,
        thermal candidates, nets, zones, Eco1 shapes, and capabilities.
    """
    board, board_path = _load_board_if_needed(board_or_path)
    copper_ids, layer_names, stack_info = _resolve_layers_and_stackup(board)
    bbox = _get_board_bbox(board)
    x_min_mm, y_min_mm, x_max_mm, y_max_mm = _bbox_bounds_mm(bbox)
    width_mm = max(0.0, x_max_mm - x_min_mm)
    height_mm = max(0.0, y_max_mm - y_min_mm)

    stackup_derived = _derive_stackup_thicknesses(board, copper_ids, stack_info, {})
    recommended_res_mm = _calculate_suggested_resolution(width_mm, height_mm, len(copper_ids))
    total_zones, unfilled_zones = _count_zones(board)
    eco1_count = _count_eco1_shapes(board)

    raw_fps = list(board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints())
    footprints_out: List[Dict[str, Any]] = []
    thermal_candidates: List[Dict[str, Any]] = []
    net_counts: Dict[str, int] = {}

    for fp in raw_fps:
        raw_pads = list(fp.Pads())
        ref, value, fp_name, fp_layer, fp_x, fp_y = _footprint_meta(
            board, fp, raw_pads, copper_ids=copper_ids
        )

        pad_records: List[Dict[str, Any]] = []
        max_area = 0.0
        max_idx = -1
        for idx, pad in enumerate(raw_pads):
            try:
                number = str(pad.GetNumber() or "")
            except Exception:
                number = ""
            net = _pad_net_name(pad)
            if net:
                net_counts[net] = net_counts.get(net, 0) + 1
            pad_layer = _pad_layer_name(board, pad, copper_ids=copper_ids)
            try:
                pos = pad.GetPosition()
                px, py = float(pos.x) * 1e-6, float(pos.y) * 1e-6
            except Exception:
                px, py = fp_x, fp_y
            w_mm, h_mm = _pad_dimensions_mm(pad)
            area_mm2 = w_mm * h_mm
            if area_mm2 > max_area:
                max_area = area_mm2
                max_idx = idx
            is_cand = (area_mm2 >= 4.0) or _is_thermal_name(number)
            pad_records.append({
                "number": number,
                "name": f"{ref}.{number}" if ref else number,
                "net": net,
                "layer": pad_layer,
                "x_mm": px,
                "y_mm": py,
                "width_mm": w_mm,
                "height_mm": h_mm,
                "area_mm2": area_mm2,
                "is_thermal_candidate": bool(is_cand),
            })

        if len(raw_pads) >= 8 and max_idx >= 0 and max_area >= 2.0:
            pad_records[max_idx]["is_thermal_candidate"] = True

        for prec in pad_records:
            if prec["is_thermal_candidate"]:
                thermal_candidates.append({
                    "ref": ref,
                    "pad": prec["number"],
                    "name": prec["name"],
                    "area_mm2": prec["area_mm2"],
                    "net": prec["net"],
                })

        footprints_out.append({
            "ref": ref,
            "value": value,
            "footprint": fp_name,
            "layer": fp_layer,
            "x_mm": fp_x,
            "y_mm": fp_y,
            "pad_count": len(pad_records),
            "pads": pad_records,
        })

    footprints_out.sort(key=lambda item: item["ref"])
    thermal_candidates.sort(key=lambda item: (-item["area_mm2"], item["name"]))
    nets_out = [
        {"name": name, "pad_count": count}
        for name, count in sorted(net_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    board_name = os.path.basename(board_path) if board_path else ""
    return _json_clean({
        "schema_version": SCHEMA_VERSION,
        "command": "inspect",
        "board": {
            "path": board_path,
            "name": board_name,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "x_min_mm": x_min_mm,
            "y_min_mm": y_min_mm,
        },
        "stackup": {
            "layer_count": len(copper_ids),
            "layers": layer_names,
            "copper_thickness_mm": stackup_derived["copper_thickness_mm_used"],
            "dielectric_gap_mm": stackup_derived["gap_mm_used"],
            "total_thickness_mm": stackup_derived["total_thick_mm_used"],
        },
        "recommended_res_mm": recommended_res_mm,
        "zones": {
            "total": total_zones,
            "unfilled": unfilled_zones,
        },
        "thermal_pad_eco1": {
            "present": eco1_count > 0,
            "shape_count": eco1_count,
        },
        "footprints": footprints_out,
        "thermal_candidates": thermal_candidates,
        "nets": nets_out,
        "capabilities": _get_capabilities_dict(),
    })


def _parse_ref_pad_name(raw_name: str) -> Tuple[str, str]:
    """Parse ``'U1.37'`` or ``'U1-37 [NET]'`` into ``(ref, pad)``."""
    cleaned = re.sub(r"\s*\[.*\]$", "", str(raw_name or "")).strip()
    if "." in cleaned:
        ref, pad = cleaned.split(".", 1)
        return ref.strip(), pad.strip()
    if "-" in cleaned:
        ref, pad = cleaned.rsplit("-", 1)
        return ref.strip(), pad.strip()
    return cleaned, ""


def _normalize_power_value(val: Any) -> str:
    """Normalize a numeric or PWL power specification into a clean string."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return str(float(val))
    text = str(val if val is not None else "0.0").strip()
    if len(text) >= 2 and text[-1] in ("w", "W"):
        prefix = text[:-1].strip()
        try:
            return str(float(prefix))
        except ValueError:
            pass
    try:
        return str(float(text))
    except ValueError:
        return text


def _is_pwl_string(power_str: str) -> bool:
    """Return True if ``power_str`` does not parse as a finite float."""
    try:
        float(str(power_str).strip())
        return False
    except (TypeError, ValueError):
        return True


def _normalize_power_pads_input(raw_pads: Any, fallback_power: str = "1.0") -> List[Dict[str, str]]:
    """Normalize ``power_pads`` from list or mapping into ``[{ref, pad, name, power}]``."""
    entries: List[Dict[str, str]] = []
    if isinstance(raw_pads, dict):
        for key, val in raw_pads.items():
            ref, pad = _parse_ref_pad_name(str(key))
            name = f"{ref}.{pad}" if (ref and pad) else str(key).strip()
            entries.append({
                "ref": ref,
                "pad": pad,
                "name": name,
                "power": _normalize_power_value(val),
            })
        return entries

    for item in list(raw_pads or []):
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref", "") or "").strip()
        pad = str(item.get("pad", "") or "").strip()
        raw_name = str(item.get("name", "") or "").strip()
        if not (ref and pad) and raw_name:
            parsed_ref, parsed_pad = _parse_ref_pad_name(raw_name)
            ref = ref or parsed_ref
            pad = pad or parsed_pad
        canonical_name = f"{ref}.{pad}" if (ref and pad) else raw_name

        if item.get("pwl") not in (None, ""):
            power_val = str(item["pwl"]).strip()
        elif "power_w" in item and item["power_w"] is not None:
            power_val = _normalize_power_value(item["power_w"])
        elif "power" in item and item["power"] is not None:
            power_val = _normalize_power_value(item["power"])
        elif "power_str" in item and item["power_str"] is not None:
            power_val = _normalize_power_value(item["power_str"])
        else:
            power_val = _normalize_power_value(fallback_power)

        entry: Dict[str, str] = {
            "ref": ref,
            "pad": pad,
            "name": canonical_name,
            "power": power_val,
        }
        if item.get("pad_key"):
            entry["pad_key"] = str(item["pad_key"])
        entries.append(entry)
    return entries


def _normalize_sharing_mode(raw_mode: Any, default: str = "equipotential") -> str:
    """Normalize a current-flow sharing mode to ``equipotential``, ``equal``, or ``fixed``."""
    mode = str(raw_mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if mode in {"equipotential", "equipotential_(auto)", "auto", "natural"}:
        return "equipotential"
    if mode in {"equal", "equal_split", "split"}:
        return "equal"
    if mode in {"fixed", "fixed_per_pad", "per_pad"}:
        return "fixed"
    return default


def _normalize_flow_side_entries(raw_side: Any) -> List[Dict[str, Any]]:
    """Normalize a flow side selector (mapping, string, or list) into pad spec dicts."""
    entries: List[Dict[str, Any]] = []
    if isinstance(raw_side, dict):
        for key, val in raw_side.items():
            ref, pad = _parse_ref_pad_name(str(key))
            name = f"{ref}.{pad}" if (ref and pad) else str(key).strip()
            entries.append({
                "ref": ref,
                "pad": pad,
                "name": name,
                "current_a": abs(float(val or 0.0)),
            })
        return entries

    items = [raw_side] if isinstance(raw_side, str) else list(raw_side or [])
    for item in items:
        if isinstance(item, str):
            ref, pad = _parse_ref_pad_name(item)
            name = f"{ref}.{pad}" if (ref and pad) else item.strip()
            entries.append({
                "ref": ref,
                "pad": pad,
                "name": name,
                "current_a": 0.0,
            })
        elif isinstance(item, dict):
            ref = str(item.get("ref", "") or "").strip()
            pad = str(item.get("pad", "") or "").strip()
            raw_name = str(item.get("name", "") or "").strip()
            if not (ref and pad) and raw_name:
                parsed_ref, parsed_pad = _parse_ref_pad_name(raw_name)
                ref = ref or parsed_ref
                pad = pad or parsed_pad
            canonical = raw_name or (f"{ref}.{pad}" if (ref and pad) else "")
            entry: Dict[str, Any] = {
                "ref": ref,
                "pad": pad,
                "name": canonical,
                "net_name": str(item.get("net_name", item.get("net", "")) or ""),
                "net_code": int(item.get("net_code", 0) or 0),
                "layer": str(item.get("layer", "") or ""),
                "current_a": abs(float(item.get("current_a", item.get("current", 0.0)) or 0.0)),
            }
            if item.get("pad_key"):
                entry["pad_key"] = str(item["pad_key"])
            entries.append(entry)
    return entries


def _normalize_current_flows_input(raw_flows: Any) -> List[Dict[str, Any]]:
    """Normalize ``current_flows`` from canonical GUI dicts or compact JSON job specs."""
    normalized: List[Dict[str, Any]] = []
    for idx, raw_flow in enumerate(list(raw_flows or [])):
        if not isinstance(raw_flow, dict):
            continue
        from_raw = raw_flow.get("from", raw_flow.get("from_pads", []))
        to_raw = raw_flow.get("to", raw_flow.get("to_pads", []))
        from_pads = _normalize_flow_side_entries(from_raw)
        to_pads = _normalize_flow_side_entries(to_raw)

        has_flow_total = any(
            raw_flow.get(k) is not None for k in ("total_current_a", "current_a", "total_current")
        )

        def _default_sharing(side_raw: Any, side_pads: List[Dict[str, Any]]) -> str:
            if isinstance(side_raw, dict):
                return "fixed"
            if (
                not has_flow_total
                and side_pads
                and any(float(p.get("current_a", 0.0)) > 0.0 for p in side_pads)
            ):
                return "fixed"
            return "equipotential"

        from_sharing = _normalize_sharing_mode(
            raw_flow.get("from_sharing"),
            default=_default_sharing(from_raw, from_pads),
        )
        to_sharing = _normalize_sharing_mode(
            raw_flow.get("to_sharing"),
            default=_default_sharing(to_raw, to_pads),
        )

        raw_total = raw_flow.get(
            "total_current_a",
            raw_flow.get("current_a", raw_flow.get("total_current", 0.0)),
        )
        total_current_a = abs(float(raw_total or 0.0))

        if to_sharing == "fixed" and from_sharing != "fixed":
            total_current_a = sum(float(p["current_a"]) for p in to_pads)
        elif from_sharing == "fixed" and to_sharing != "fixed":
            total_current_a = sum(float(p["current_a"]) for p in from_pads)
        elif from_sharing == "fixed" and to_sharing == "fixed":
            total_current_a = sum(float(p["current_a"]) for p in from_pads)

        def _sync_side_currents(side_pads: List[Dict[str, Any]], sharing: str) -> None:
            if not side_pads or sharing == "fixed":
                return
            if sharing == "equal":
                per_pad = total_current_a / float(len(side_pads))
                for p in side_pads:
                    p["current_a"] = per_pad
            elif sharing == "equipotential":
                for p in side_pads:
                    p["current_a"] = total_current_a if len(side_pads) == 1 else 0.0

        _sync_side_currents(from_pads, from_sharing)
        _sync_side_currents(to_pads, to_sharing)

        net_name = str(raw_flow.get("net_name", raw_flow.get("net", "")) or "")
        net_code = int(raw_flow.get("net_code", 0) or 0)
        if not net_name:
            for p in from_pads + to_pads:
                if p.get("net_name"):
                    net_name = str(p["net_name"])
                    net_code = int(p.get("net_code", net_code) or 0)
                    break

        normalized.append({
            "name": str(raw_flow.get("name") or f"Flow {idx + 1}"),
            "net_name": net_name,
            "net_code": net_code,
            "total_current_a": float(total_current_a),
            "from_sharing": from_sharing,
            "to_sharing": to_sharing,
            "from_pads": from_pads,
            "to_pads": to_pads,
        })
    return normalized


def normalize_job_config(
    job_config: Optional[Dict[str, Any]],
    suggested_res_mm: float = 0.5,
    has_eco1: bool = False,
) -> Dict[str, Any]:
    """Normalize an agent-friendly job configuration without mutating input.

    Parameters
    ----------
    job_config : dict or None
        Raw job configuration from JSON or CLI overrides.
    suggested_res_mm : float, optional
        Fallback grid resolution when ``res`` is omitted.
    has_eco1 : bool, optional
        Whether the target board has ``User.Eco1`` thermal-pad shapes.

    Returns
    -------
    dict
        Normalized, JSON-serializable configuration dictionary.
    """
    raw = dict(job_config or {})

    fallback_power = str(raw.get("power_str", "1.0") or "1.0")
    power_pads = _normalize_power_pads_input(raw.get("power_pads", []), fallback_power=fallback_power)
    uses_pwl = any(_is_pwl_string(p["power"]) for p in power_pads)

    if "steady_state" in raw and raw["steady_state"] is not None:
        steady_state = bool(raw["steady_state"])
    else:
        has_explicit_time = ("time" in raw and raw["time"] is not None)
        steady_state = (not has_explicit_time) and (not uses_pwl)

    sim_time = float(20.0 if raw.get("time") is None else raw["time"])
    amb = float(25.0 if raw.get("amb") is None else raw["amb"])
    res = float(suggested_res_mm if raw.get("res") is None else raw["res"])
    thick = float(1.6 if raw.get("thick") is None else raw["thick"])
    h_conv = float(10.0 if raw.get("h_conv") is None else raw["h_conv"])
    raw_h_conv_bottom = raw.get("h_conv_bottom")
    h_conv_bottom = h_conv if raw_h_conv_bottom is None else float(raw_h_conv_bottom)
    via_plating_um = float(25.0 if raw.get("via_plating_um") is None else raw["via_plating_um"])
    via_fill = normalize_via_fill(raw.get("via_fill"))

    grid_detail = str(raw.get("grid_detail_level") or DEFAULT_GRID_DETAIL).strip().lower()
    grid_detail = grid_detail.replace(" ", "_").replace("-", "_")
    if grid_detail not in set(GRID_DETAIL_PRESETS) | {"custom"}:
        grid_detail = DEFAULT_GRID_DETAIL

    if raw.get("grid_node_budget") is not None:
        grid_node_budget = int(raw["grid_node_budget"])
    elif grid_detail in GRID_DETAIL_PRESETS:
        grid_node_budget = int(GRID_DETAIL_PRESETS[grid_detail][0])
    else:
        grid_node_budget = DEFAULT_GRID_NODE_BUDGET

    # Spatial mesh is always uniform since v0.4.7; accept legacy keys silently.
    mesh_mode = "uniform"

    compute_engine = str(raw.get("compute_engine") or "auto").strip().lower()
    if compute_engine not in {"auto", "fast_native", "legacy"}:
        compute_engine = "auto"

    time_stepping = str(raw.get("time_stepping") or "auto").strip().lower()
    if time_stepping not in {"auto", "multi_phase", "two_phase", "uniform"}:
        time_stepping = "auto"

    raw_area_mode = raw.get("area_mode")
    if raw_area_mode is not None:
        area_mode = str(raw_area_mode).strip().lower()
        if area_mode not in {"full", "active"}:
            area_mode = "active" if bool(raw.get("limit_area", False)) else "full"
        limit_area = bool(raw.get("limit_area", area_mode == "active"))
    else:
        limit_area = bool(raw.get("limit_area", False))
        area_mode = "active" if limit_area else "full"

    pad_dist_mm = float(10.0 if raw.get("pad_dist_mm") is None else raw["pad_dist_mm"])

    has_any_hs_key = any(k.startswith("heatsink_") and raw.get(k) is not None for k in raw)
    if "pad_sink_mode" in raw and raw["pad_sink_mode"] is not None:
        pad_sink_mode = str(raw["pad_sink_mode"]).strip().lower()
        if pad_sink_mode not in {"fixed", "heatsink"}:
            pad_sink_mode = "fixed"
    elif "heatsink_rth_kw" in raw and raw["heatsink_rth_kw"] is not None:
        pad_sink_mode = "heatsink"
    else:
        pad_sink_mode = "fixed"

    if "use_heatsink" in raw and raw["use_heatsink"] is not None:
        use_heatsink = bool(raw["use_heatsink"])
    else:
        use_heatsink = (pad_sink_mode == "heatsink") or has_any_hs_key

    pad_th = float(1.0 if raw.get("pad_th") is None else raw["pad_th"])
    pad_k = float(3.0 if raw.get("pad_k") is None else raw["pad_k"])
    pad_cap_areal = float(0.0 if raw.get("pad_cap_areal") is None else raw["pad_cap_areal"])
    pad_contact_factor = float(0.2 if raw.get("pad_contact_factor") is None else raw["pad_contact_factor"])

    if "pad_sink_temp_use_ambient" in raw and raw["pad_sink_temp_use_ambient"] is not None:
        pad_sink_temp_use_ambient = bool(raw["pad_sink_temp_use_ambient"])
    elif "pad_sink_temp_c" in raw and raw["pad_sink_temp_c"] is not None:
        pad_sink_temp_use_ambient = False
    else:
        pad_sink_temp_use_ambient = True

    pad_sink_temp_c = float(25.0 if raw.get("pad_sink_temp_c") is None else raw["pad_sink_temp_c"])

    heatsink_rth_kw = float(5.0 if raw.get("heatsink_rth_kw") is None else raw["heatsink_rth_kw"])
    heatsink_material = str(raw.get("heatsink_material") or "aluminium").strip().lower()
    if heatsink_material not in set(HEATSINK_MATERIAL_CP) | {"custom"}:
        heatsink_material = "aluminium"
    heatsink_mass_g = float(50.0 if raw.get("heatsink_mass_g") is None else raw["heatsink_mass_g"])
    if heatsink_material == "custom":
        heatsink_cp_j_kgk = float(897.0 if raw.get("heatsink_cp_j_kgk") is None else raw["heatsink_cp_j_kgk"])
    else:
        heatsink_cp_j_kgk = float(HEATSINK_MATERIAL_CP[heatsink_material])

    limits_norm = None
    raw_limits = raw.get("limits")
    if isinstance(raw_limits, dict):
        comp_limits = {}
        if isinstance(raw_limits.get("components"), dict):
            for pattern, val in raw_limits["components"].items():
                if val is not None:
                    comp_limits[str(pattern)] = float(val)
        limits_norm = {
            "max_board_c": (
                float(raw_limits["max_board_c"])
                if raw_limits.get("max_board_c") is not None
                else None
            ),
            "max_heatsink_c": (
                float(raw_limits["max_heatsink_c"])
                if raw_limits.get("max_heatsink_c") is not None
                else None
            ),
            "components": comp_limits,
        }

    normalized = {
        "power_pads": power_pads,
        "steady_state": bool(steady_state),
        "time": sim_time,
        "amb": amb,
        "res": res,
        "thick": thick,
        "h_conv": h_conv,
        "h_conv_bottom": h_conv_bottom,
        "via_plating_um": via_plating_um,
        "via_fill": via_fill,
        "grid_detail_level": grid_detail,
        "grid_node_budget": grid_node_budget,
        "mesh_mode": mesh_mode,
        "compute_engine": compute_engine,
        "time_stepping": time_stepping,
        "limit_area": bool(limit_area),
        "pad_dist_mm": pad_dist_mm,
        "area_mode": area_mode,
        "area_margin_mm": pad_dist_mm,
        "ignore_traces": bool(raw.get("ignore_traces", False)),
        "ignore_polygons": bool(raw.get("ignore_polygons", False)),
        "show_all": bool(True if raw.get("show_all") is None else raw["show_all"]),
        "snapshots": bool(raw.get("snapshots", False)),
        "snap_count": max(1, min(50, int(5 if raw.get("snap_count") is None else raw["snap_count"]))),
        "use_heatsink": bool(use_heatsink),
        "pad_sink_mode": pad_sink_mode,
        "pad_th": pad_th,
        "pad_k": pad_k,
        "pad_cap_areal": pad_cap_areal,
        "pad_contact_factor": pad_contact_factor,
        "pad_sink_temp_use_ambient": bool(pad_sink_temp_use_ambient),
        "pad_sink_temp_c": pad_sink_temp_c,
        "heatsink_rth_kw": heatsink_rth_kw,
        "heatsink_material": heatsink_material,
        "heatsink_mass_g": heatsink_mass_g,
        "heatsink_cp_j_kgk": heatsink_cp_j_kgk,
        "current_enabled": bool(raw.get("current_enabled", False)),
        "electro_thermal_coupling": bool(
            True if raw.get("electro_thermal_coupling") is None else raw["electro_thermal_coupling"]
        ),
        "current_flows": _normalize_current_flows_input(raw.get("current_flows", [])),
        "current_groups": list(raw.get("current_groups") or []),
        "limits": limits_norm,
        "copper_thickness_scale": float(1.0 if raw.get("copper_thickness_scale") is None else raw["copper_thickness_scale"]),
        "power_scale": float(1.0 if raw.get("power_scale") is None else raw["power_scale"]),
        "refill_zones": bool(raw.get("refill_zones", False)),
    }
    return _json_clean(normalized)


def _build_pad_index(board: Any, copper_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """Build exact and case-insensitive lookup maps over all pads on ``board``."""
    exact_ref_pad: Dict[Tuple[str, str], Tuple[Dict[str, Any], Any]] = {}
    ci_ref_pad: Dict[Tuple[str, str], Tuple[Dict[str, Any], Any]] = {}
    by_key: Dict[str, Tuple[Dict[str, Any], Any]] = {}
    by_legacy_name: Dict[str, Tuple[Dict[str, Any], Any]] = {}
    pads_by_ref: Dict[str, List[str]] = {}
    pads_by_ref_ci: Dict[str, Tuple[str, List[str]]] = {}

    fps = list(board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints())
    for fp in fps:
        try:
            ref = str(fp.GetReference() or "")
        except Exception:
            ref = ""
        pad_nums: List[str] = []
        for pad in fp.Pads():
            try:
                num = str(pad.GetNumber() or "")
            except Exception:
                num = ""
            pad_nums.append(num)
            try:
                pos = pad.GetPosition()
                px, py = int(pos.x), int(pos.y)
            except Exception:
                px, py = 0, 0
            net_name = _pad_net_name(pad)
            net_code = _pad_net_code(pad)
            layer_name = _pad_layer_name(board, pad, copper_ids=copper_ids)
            pad_key = f"{ref}:{num}:{net_code}:{px}:{py}"
            legacy_base = f"{ref}-{num}"
            legacy_full = f"{legacy_base} [{net_name}]" if net_name else legacy_base
            desc = {
                "ref": ref,
                "pad": num,
                "name": f"{ref}.{num}" if ref else num,
                "net": net_name,
                "net_name": net_name,
                "net_code": net_code,
                "layer": layer_name,
                "pad_key": pad_key,
            }
            exact_ref_pad.setdefault((ref, num), (desc, pad))
            ci_ref_pad.setdefault((ref.upper(), num.upper()), (desc, pad))
            by_key[pad_key] = (desc, pad)
            by_legacy_name.setdefault(legacy_full, (desc, pad))
            by_legacy_name.setdefault(legacy_base, (desc, pad))

        pads_by_ref[ref] = pad_nums
        pads_by_ref_ci[ref.upper()] = (ref, pad_nums)

    return {
        "exact_ref_pad": exact_ref_pad,
        "ci_ref_pad": ci_ref_pad,
        "by_key": by_key,
        "by_legacy_name": by_legacy_name,
        "pads_by_ref": pads_by_ref,
        "pads_by_ref_ci": pads_by_ref_ci,
        "footprint_refs": sorted(pads_by_ref.keys()),
    }


def _lookup_pad(pad_index: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[Optional[Tuple[Dict[str, Any], Any]], str]:
    """Resolve a pad spec dict against ``pad_index`` and return ``(match, hint)``."""
    pad_key = spec.get("pad_key")
    if pad_key and pad_key in pad_index["by_key"]:
        return pad_index["by_key"][pad_key], ""

    ref = str(spec.get("ref", "") or "").strip()
    pad_num = str(spec.get("pad", "") or "").strip()
    raw_name = str(spec.get("name", "") or "").strip()
    if not (ref and pad_num) and raw_name:
        ref, pad_num = _parse_ref_pad_name(raw_name)

    if (ref, pad_num) in pad_index["exact_ref_pad"]:
        return pad_index["exact_ref_pad"][(ref, pad_num)], ""
    if (ref.upper(), pad_num.upper()) in pad_index["ci_ref_pad"]:
        return pad_index["ci_ref_pad"][(ref.upper(), pad_num.upper())], ""
    if raw_name in pad_index["by_legacy_name"]:
        return pad_index["by_legacy_name"][raw_name], ""

    if ref.upper() in pad_index["pads_by_ref_ci"]:
        actual_ref, nums = pad_index["pads_by_ref_ci"][ref.upper()]
        sample = ", ".join(nums[:20])
        suffix = ", ..." if len(nums) > 20 else ""
        hint = f"Available pads on {actual_ref}: {sample}{suffix}"
    else:
        refs = pad_index["footprint_refs"]
        sample = ", ".join(refs[:20])
        suffix = ", ..." if len(refs) > 20 else ""
        hint = f"Available footprints: {sample}{suffix}" if refs else "No footprints found on board."
    return None, hint


def _resolve_current_groups_entries(
    raw_groups: List[Any],
    pad_index: Dict[str, Any],
) -> Tuple[List[CurrentTerminal], List[Tuple[str, str]]]:
    """Resolve current-group pad specifications to ``CurrentTerminal`` instances."""
    terminals: List[CurrentTerminal] = []
    missing: List[Tuple[str, str]] = []

    for group in list(raw_groups or []):
        if not isinstance(group, dict):
            continue
        mode = str(group.get("mode", "per_pad") or "per_pad").strip().lower()
        raw_pads = group.get("pads", [])
        pad_items: List[Dict[str, Any]] = []
        if isinstance(raw_pads, dict):
            for key, current_val in raw_pads.items():
                ref, pnum = _parse_ref_pad_name(str(key))
                pad_items.append({"ref": ref, "pad": pnum, "name": str(key), "current_a": float(current_val)})
        else:
            for p in list(raw_pads or []):
                if isinstance(p, dict):
                    pad_items.append(dict(p))

        if mode == "total" and pad_items:
            total_i = float(group.get("total_current_a", group.get("total_current", 0.0)) or 0.0)
            per_pad = total_i / float(len(pad_items))
            for item in pad_items:
                item["current_a"] = per_pad

        for item in pad_items:
            current_a = float(item.get("current_a", item.get("current", 0.0)) or 0.0)
            if abs(current_a) <= 0.0:
                continue
            match, hint = _lookup_pad(pad_index, item)
            if not match:
                target = str(item.get("name") or f"{item.get('ref', '')}.{item.get('pad', '')}").strip(".")
                missing.append((target or "<unknown>", hint))
                continue
            desc, pad_obj = match
            terminals.append(CurrentTerminal(
                pad=pad_obj,
                name=desc["name"],
                net_name=desc["net_name"],
                net_code=int(desc["net_code"]),
                current_a=current_a,
            ))

    return terminals, missing


def _emit_flow_side_entries(
    side_label: str,
    resolved_items: List[Tuple[Dict[str, Any], Any, float]],
    sharing: str,
    total_a: float,
    sign: float,
    flow_name: str,
    flow_net_name: str,
    flow_net_code: int,
    terminals: List[CurrentTerminal],
    buses: List[Any],
) -> None:
    """Emit ``EquipotentialBus`` or ``CurrentTerminal`` entries for one side of a flow."""
    if not resolved_items:
        return
    if sharing == "equipotential":
        if abs(total_a) <= 0.0:
            return
        if len(resolved_items) >= 2 and EquipotentialBus is not None:
            buses.append(EquipotentialBus(
                name=f"{flow_name} ({side_label})",
                net_name=flow_net_name,
                net_code=int(flow_net_code),
                pads=[(pad_obj, desc["name"]) for desc, pad_obj, _ in resolved_items],
                total_current_a=float(sign * total_a),
            ))
        else:
            per_pad = (sign * total_a) / float(len(resolved_items))
            for desc, pad_obj, _ in resolved_items:
                terminals.append(CurrentTerminal(
                    pad=pad_obj,
                    name=desc["name"],
                    net_name=desc["net_name"],
                    net_code=int(desc["net_code"]),
                    current_a=float(per_pad),
                ))
    elif sharing == "equal":
        if abs(total_a) <= 0.0:
            return
        per_pad = (sign * total_a) / float(len(resolved_items))
        for desc, pad_obj, _ in resolved_items:
            terminals.append(CurrentTerminal(
                pad=pad_obj,
                name=desc["name"],
                net_name=desc["net_name"],
                net_code=int(desc["net_code"]),
                current_a=float(per_pad),
            ))
    else:  # fixed
        for desc, pad_obj, item_current in resolved_items:
            curr = float(sign * abs(float(item_current)))
            if abs(curr) <= 0.0:
                continue
            terminals.append(CurrentTerminal(
                pad=pad_obj,
                name=desc["name"],
                net_name=desc["net_name"],
                net_code=int(desc["net_code"]),
                current_a=curr,
            ))


def _resolve_current_flows_entries(
    raw_flows: List[Any],
    pad_index: Dict[str, Any],
) -> Tuple[List[CurrentTerminal], List[Any], List[Tuple[str, str]], List[Dict[str, Any]]]:
    """Resolve ``current_flows`` into ``(terminals, buses, missing, flow_errors)``."""
    terminals: List[CurrentTerminal] = []
    buses: List[Any] = []
    missing: List[Tuple[str, str]] = []
    flow_errors: List[Dict[str, Any]] = []

    for flow in list(raw_flows or []):
        if not isinstance(flow, dict):
            continue
        flow_name = str(flow.get("name") or "Flow")
        from_pads = list(flow.get("from_pads") or [])
        to_pads = list(flow.get("to_pads") or [])
        total_a = abs(float(flow.get("total_current_a", 0.0) or 0.0))
        from_sharing = _normalize_sharing_mode(flow.get("from_sharing"), default="equipotential")
        to_sharing = _normalize_sharing_mode(flow.get("to_sharing"), default="equipotential")

        if (from_pads or to_pads or total_a > 0.0) and (not from_pads or not to_pads):
            flow_errors.append({
                "code": "FLOW_INCOMPLETE",
                "target": flow_name,
                "message": f"Current flow '{flow_name}' must specify both 'from' and 'to' pads.",
            })

        def _resolve_side(side_pads: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Any, float]]:
            resolved: List[Tuple[Dict[str, Any], Any, float]] = []
            for item in side_pads:
                match, hint = _lookup_pad(pad_index, item)
                if not match:
                    target = str(item.get("name") or f"{item.get('ref', '')}.{item.get('pad', '')}").strip(".")
                    missing.append((target or "<unknown>", hint))
                    continue
                desc, pad_obj = match
                resolved.append((desc, pad_obj, float(item.get("current_a", 0.0) or 0.0)))
            return resolved

        resolved_from = _resolve_side(from_pads)
        resolved_to = _resolve_side(to_pads)

        all_resolved = resolved_from + resolved_to
        if not all_resolved:
            continue

        net_keys = {
            net_key_from_values(int(desc["net_code"]), desc["net_name"])
            for desc, _, _ in all_resolved
        }
        if len(net_keys) > 1:
            flow_errors.append({
                "code": "FLOW_NET_MISMATCH",
                "target": flow_name,
                "message": f"Current flow '{flow_name}' references pads on multiple nets; all pads in a flow must share one net.",
            })
            continue

        first_desc = all_resolved[0][0]
        flow_net_name = str(first_desc["net_name"] or flow.get("net_name", "") or "")
        flow_net_code = int(first_desc["net_code"])

        _emit_flow_side_entries(
            "From", resolved_from, from_sharing, total_a, +1.0,
            flow_name, flow_net_name, flow_net_code, terminals, buses,
        )
        _emit_flow_side_entries(
            "To", resolved_to, to_sharing, total_a, -1.0,
            flow_name, flow_net_name, flow_net_code, terminals, buses,
        )

    return terminals, buses, missing, flow_errors


def _validate_with_context(
    board: Any,
    board_path: str,
    copper_ids: List[int],
    layer_names: List[str],
    stack_info: Dict[str, Any],
    bbox: Any,
    job_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Internal validation returning both JSON payload and resolved live objects."""
    x_min_mm, y_min_mm, x_max_mm, y_max_mm = _bbox_bounds_mm(bbox)
    w_mm = max(0.0, x_max_mm - x_min_mm)
    h_mm = max(0.0, y_max_mm - y_min_mm)
    suggested_res = _calculate_suggested_resolution(w_mm, h_mm, len(copper_ids))
    eco1_count = _count_eco1_shapes(board)

    norm = normalize_job_config(job_config, suggested_res_mm=suggested_res, has_eco1=(eco1_count > 0))
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if norm["res"] <= 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "res",
            "message": f"Grid resolution must be > 0 (got {norm['res']}).",
        })
    if not norm["steady_state"] and norm["time"] <= 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "time",
            "message": f"Simulation time must be > 0 in transient mode (got {norm['time']}).",
        })
    if norm["h_conv"] < 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "h_conv",
            "message": f"Convection coefficient h_conv must be >= 0 (got {norm['h_conv']}).",
        })
    if norm["h_conv_bottom"] < 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "h_conv_bottom",
            "message": f"Bottom convection coefficient h_conv_bottom must be >= 0 (got {norm['h_conv_bottom']}).",
        })
    if not (5.0 <= norm["via_plating_um"] <= 100.0):
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "via_plating_um",
            "message": f"Via plating thickness via_plating_um must be between 5.0 and 100.0 um (got {norm['via_plating_um']}).",
        })
    if norm["copper_thickness_scale"] <= 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "copper_thickness_scale",
            "message": f"copper_thickness_scale must be > 0 (got {norm['copper_thickness_scale']}).",
        })
    if norm["heatsink_rth_kw"] <= 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "heatsink_rth_kw",
            "message": f"heatsink_rth_kw must be > 0 (got {norm['heatsink_rth_kw']}).",
        })
    if norm["thick"] is None or not math.isfinite(norm["thick"]) or norm["thick"] <= 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "thick",
            "message": f"thick must be > 0 (got {norm['thick']}).",
        })
    if norm["pad_dist_mm"] is None or not math.isfinite(norm["pad_dist_mm"]) or norm["pad_dist_mm"] < 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "pad_dist_mm",
            "message": f"pad_dist_mm must be >= 0 (got {norm['pad_dist_mm']}).",
        })
    if norm["pad_th"] is None or not math.isfinite(norm["pad_th"]) or norm["pad_th"] <= 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "pad_th",
            "message": f"pad_th must be > 0 (got {norm['pad_th']}).",
        })
    if norm["pad_k"] is None or not math.isfinite(norm["pad_k"]) or norm["pad_k"] <= 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "pad_k",
            "message": f"pad_k must be > 0 (got {norm['pad_k']}).",
        })
    if (
        norm["pad_contact_factor"] is None
        or not math.isfinite(norm["pad_contact_factor"])
        or not (0.05 <= norm["pad_contact_factor"] <= 1.0)
    ):
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "pad_contact_factor",
            "message": f"pad_contact_factor must be between 0.05 and 1.0 (got {norm['pad_contact_factor']}).",
        })
    if norm["heatsink_mass_g"] is None or not math.isfinite(norm["heatsink_mass_g"]) or norm["heatsink_mass_g"] < 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "heatsink_mass_g",
            "message": f"heatsink_mass_g must be >= 0 (got {norm['heatsink_mass_g']}).",
        })
    if (
        norm["heatsink_cp_j_kgk"] is None
        or not math.isfinite(norm["heatsink_cp_j_kgk"])
        or norm["heatsink_cp_j_kgk"] <= 0.0
    ):
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "heatsink_cp_j_kgk",
            "message": f"heatsink_cp_j_kgk must be > 0 (got {norm['heatsink_cp_j_kgk']}).",
        })
    if norm["power_scale"] is None or not math.isfinite(norm["power_scale"]) or norm["power_scale"] < 0.0:
        errors.append({
            "code": "INVALID_PARAMETER",
            "target": "power_scale",
            "message": f"power_scale must be >= 0 (got {norm['power_scale']}).",
        })

    _, unfilled_zones = _count_zones(board)
    if unfilled_zones > 0:
        if norm["refill_zones"]:
            warnings.append({
                "code": "UNFILLED_ZONES_WILL_REFILL",
                "message": f"{unfilled_zones} copper zone(s) are unfilled and will be refilled before solving.",
            })
        else:
            errors.append({
                "code": "UNFILLED_ZONES",
                "message": f"{unfilled_zones} copper zone(s) are not filled.",
                "hint": "Set refill_zones=True (or pass --refill-zones) or refill zones in KiCad.",
            })

    pad_index = _build_pad_index(board, copper_ids=copper_ids)

    resolved_power_pads: List[Dict[str, Any]] = []
    resolved_power_objs: List[Any] = []
    pad_sources: List[Tuple[str, Any]] = []
    has_nonzero_power = False
    has_pwl = False
    power_scale = float(norm["power_scale"])

    for pspec in norm["power_pads"]:
        match, hint = _lookup_pad(pad_index, pspec)
        if not match:
            target = pspec.get("name") or f"{pspec.get('ref', '')}.{pspec.get('pad', '')}"
            errors.append({
                "code": "PAD_NOT_FOUND",
                "target": target,
                "message": f"Pad '{target}' was not found on the board.",
                "hint": hint,
            })
            continue
        desc, pad_obj = match
        raw_power = str(pspec.get("power", "0.0")).strip()
        try:
            pval = float(raw_power)
            if not math.isfinite(pval):
                raise ValueError("non-finite power")
            scaled_val = pval * power_scale
            if abs(scaled_val) > 0.0:
                has_nonzero_power = True
            pad_sources.append(("const", scaled_val))
        except ValueError:
            has_pwl = True
            try:
                times_pwl, powers_pwl = parse_pwl_file(raw_power)
                pwl_warnings = validate_pwl(times_pwl, powers_pwl)
                for msg in pwl_warnings:
                    warnings.append({"code": "PWL_WARNING", "target": desc["name"], "message": msg})
                scaled_powers = np.asarray(powers_pwl, dtype=np.float64) * power_scale
                if bool(np.any(scaled_powers != 0.0)):
                    has_nonzero_power = True
                pad_sources.append(("pwl", (times_pwl, scaled_powers)))
            except Exception as exc:
                errors.append({
                    "code": "INVALID_PWL",
                    "target": desc["name"],
                    "message": f"Invalid PWL file '{raw_power}': {exc}",
                    "hint": "Provide a valid 2-column (time_s power_w) PWL file.",
                })
                continue

        resolved_power_objs.append(pad_obj)
        resolved_power_pads.append({
            "ref": desc["ref"],
            "pad": desc["pad"],
            "name": desc["name"],
            "net": desc["net"],
            "layer": desc["layer"],
            "power": raw_power,
        })

    if norm["steady_state"] and has_pwl:
        warnings.append({
            "code": "STEADY_STATE_IGNORES_PWL",
            "message": "Steady-state mode ignores PWL time variation; disable steady_state for transient PWL profiles.",
        })

    terminals: List[CurrentTerminal] = []
    buses: List[Any] = []
    has_current = False
    if norm["current_enabled"]:
        if norm["current_flows"]:
            terminals, buses, missing_curr, flow_errors = _resolve_current_flows_entries(
                norm["current_flows"], pad_index
            )
            errors.extend(flow_errors)
        else:
            terminals, missing_curr = _resolve_current_groups_entries(norm["current_groups"], pad_index)
        for target, hint in missing_curr:
            errors.append({
                "code": "PAD_NOT_FOUND",
                "target": target,
                "message": f"Current terminal pad '{target}' was not found on the board.",
                "hint": hint,
            })
        has_current = (
            any(abs(float(item.current_a)) > 0.0 for item in terminals)
            or any(abs(float(bus.total_current_a)) > 0.0 for bus in buses)
        )
        totals: Dict[str, float] = {}
        abs_totals: Dict[str, float] = {}
        for item in terminals:
            net = item.net_name or f"net:{item.net_code}"
            totals[net] = totals.get(net, 0.0) + float(item.current_a)
            abs_totals[net] = abs_totals.get(net, 0.0) + abs(float(item.current_a))
        for bus in buses:
            net = bus.net_name or f"net:{bus.net_code}"
            totals[net] = totals.get(net, 0.0) + float(bus.total_current_a)
            abs_totals[net] = abs_totals.get(net, 0.0) + abs(float(bus.total_current_a))
        for net, total in totals.items():
            tol = max(1e-9, 1e-6 * abs_totals.get(net, 0.0))
            if abs(total) > tol:
                errors.append({
                    "code": "CURRENT_UNBALANCED",
                    "target": net,
                    "message": f"Current on net '{net}' is not balanced (sum = {total:.6g} A).",
                    "hint": "Ensure source and sink currents sum to 0 A on each active net.",
                })

    if not has_nonzero_power and not has_current:
        errors.append({
            "code": "NO_HEAT_SOURCE",
            "message": "Configure at least one non-zero heat source or balanced current path.",
            "hint": "Specify power_pads with non-zero power_w or enable current_enabled.",
        })

    if norm["use_heatsink"] and eco1_count == 0:
        if norm["pad_sink_mode"] == "heatsink":
            warnings.append({
                "code": "NO_ECO1_SHAPES",
                "message": "Dynamic heatsink mode is enabled, but no User.Eco1 thermal-pad shapes exist on the board.",
                "hint": "Draw the thermal interface pad shape on User.Eco1 in KiCad so the heatsink couples to the PCB.",
            })
        else:
            warnings.append({
                "code": "NO_ECO1_SHAPES",
                "message": "Thermal pad (use_heatsink) is enabled, but no User.Eco1 shapes exist on the board.",
            })

    area_terminals = list(terminals)
    for bus in buses:
        if getattr(bus, "pads", None) and abs(float(bus.total_current_a)) > 0.0:
            area_terminals.append(CurrentTerminal(
                pad=bus.pads[0][0],
                name=bus.name,
                net_name=bus.net_name,
                net_code=int(bus.net_code),
                current_a=float(bus.total_current_a),
            ))

    area = _estimate_simulation_area(
        board,
        bbox,
        norm,
        power_pads=resolved_power_objs,
        terminals=area_terminals,
    )
    for w_msg in area.warnings:
        warnings.append({"code": "AREA_ESTIMATE", "message": w_msg})

    grid = _estimate_solver_grid(
        bbox,
        max(1e-4, float(norm["res"])),
        norm,
        len(copper_ids),
        area=area,
    )
    if grid.auto_coarsened:
        warnings.append({
            "code": "AUTO_COARSENED",
            "message": f"Grid resolution coarsened from {grid.requested_res_mm:.3f} mm to {grid.actual_res_mm:.3f} mm to fit node budget.",
        })
    if grid.nodes > 10_000_000:
        warnings.append({
            "code": "LARGE_GRID",
            "message": f"Grid contains {grid.nodes:,} equivalent uniform nodes.",
        })

    grid_dict = {
        "rows": grid.rows,
        "cols": grid.cols,
        "layer_count": grid.layer_count,
        "nodes": grid.nodes,
        "requested_res_mm": grid.requested_res_mm,
        "actual_res_mm": grid.actual_res_mm,
        "x_min_mm": grid.x_min_mm,
        "y_min_mm": grid.y_min_mm,
        "width_mm": grid.width_mm,
        "height_mm": grid.height_mm,
        "auto_coarsened": grid.auto_coarsened,
        "memory_mb_low": grid.memory_mb_low,
        "memory_mb_high": grid.memory_mb_high,
        "runtime_class": grid.runtime_class,
    }

    resolved_terminals_out = [
        {
            "name": item.name,
            "net": item.net_name,
            "layer": _pad_layer_name(board, item.pad, copper_ids=copper_ids),
            "current_a": float(item.current_a),
        }
        for item in terminals
    ]

    resolved_buses_out = [
        {
            "name": bus.name,
            "net": bus.net_name,
            "total_current_a": float(bus.total_current_a),
            "pads": [pad_name for _, pad_name in bus.pads],
        }
        for bus in buses
    ]

    valid = len(errors) == 0
    payload = _json_clean({
        "schema_version": SCHEMA_VERSION,
        "command": "validate",
        "valid": valid,
        "ready": valid,
        "errors": errors,
        "warnings": warnings,
        "grid": grid_dict,
        "resolved_power_pads": resolved_power_pads,
        "resolved_current_terminals": resolved_terminals_out,
        "resolved_equipotential_buses": resolved_buses_out,
        "normalized_config": norm,
    })

    return {
        "payload": payload,
        "board": board,
        "board_path": board_path,
        "copper_ids": copper_ids,
        "layer_names": layer_names,
        "stack_info": stack_info,
        "bbox": bbox,
        "norm": norm,
        "area": area,
        "grid": grid,
        "resolved_power_objs": resolved_power_objs,
        "resolved_power_pads": resolved_power_pads,
        "pad_sources": pad_sources,
        "terminals": terminals,
        "buses": buses,
        "unfilled_zones": unfilled_zones,
    }


def validate_job(board_or_path: Any, job_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a simulation job configuration against a KiCad board without solving.

    Parameters
    ----------
    board_or_path : str or pathlib.Path or pcbnew.BOARD
        Board file path or loaded board object.
    job_config : dict or None
        Job configuration dictionary.

    Returns
    -------
    dict
        Structured validation result with ``valid``, ``errors``, ``warnings``,
        ``grid``, resolved pads/terminals, and ``normalized_config``.
    """
    board, board_path = _load_board_if_needed(board_or_path)
    copper_ids, layer_names, stack_info = _resolve_layers_and_stackup(board)
    bbox = _get_board_bbox(board)
    ctx = _validate_with_context(
        board,
        board_path,
        copper_ids,
        layer_names,
        stack_info,
        bbox,
        job_config or {},
    )
    return ctx["payload"]


def _extract_components_temperatures(
    T: np.ndarray,
    board: Any,
    layer_names: List[str],
    x_min: float,
    y_min: float,
    res: float,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute per-component temperature stats and pad geometry for nearest-component queries."""
    if board is None:
        return {}, []

    layer_count, rows, cols = T.shape
    name_to_idx = {str(name): idx for idx, name in enumerate(layer_names)}
    lid_to_idx: Dict[int, int] = {}
    copper_ids: List[int] = []
    for idx, name in enumerate(layer_names[:layer_count]):
        lid = -1
        try:
            lid = int(board.GetLayerID(name))
        except Exception:
            lid = -1
        if lid >= 0:
            lid_to_idx[lid] = idx
        copper_ids.append(lid)
    valid_copper_ids = [lid for lid in copper_ids if lid >= 0]

    def _layer_idx_from_name(lname: str, fallback: int = 0) -> int:
        if lname in name_to_idx:
            return name_to_idx[lname]
        up = str(lname or "").upper()
        if "B." in up or "BOT" in up:
            return layer_count - 1
        return fallback

    fps = list(board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints())
    comp_entries: List[Dict[str, Any]] = []
    comp_geom: List[Dict[str, Any]] = []

    for fp in fps:
        raw_pads = list(fp.Pads())
        ref, value, fp_name, fp_layer, fp_x, fp_y = _footprint_meta(
            board, fp, raw_pads, copper_ids=valid_copper_ids or None
        )
        fp_layer_idx = _layer_idx_from_name(fp_layer, 0)
        if hasattr(fp, "GetLayer"):
            try:
                fp_layer_idx = lid_to_idx.get(int(fp.GetLayer()), fp_layer_idx)
            except Exception:
                pass

        pad_vals_list: List[np.ndarray] = []
        pad_max_vals: List[float] = []
        pad_points: List[Tuple[str, float, float]] = []

        for pad in raw_pads:
            try:
                pnum = str(pad.GetNumber() or "")
            except Exception:
                pnum = ""
            try:
                pos = pad.GetPosition()
                px, py = float(pos.x) * 1e-6, float(pos.y) * 1e-6
            except Exception:
                px, py = fp_x, fp_y
            pad_points.append((f"{ref}.{pnum}" if ref else pnum, px, py))

            p_indices = (
                get_pad_copper_layer_indices(pad, copper_ids, lid_to_idx=lid_to_idx)
                if valid_copper_ids
                else []
            )
            if not p_indices:
                try:
                    p_lid = int(pad.GetLayer())
                    p_idx = lid_to_idx.get(p_lid)
                except Exception:
                    p_idx = None
                if p_idx is None:
                    p_idx = _layer_idx_from_name(
                        _pad_layer_name(board, pad, copper_ids=valid_copper_ids or None),
                        fp_layer_idx,
                    )
                p_indices = [p_idx]

            sampled_parts: List[np.ndarray] = []
            for p_idx in p_indices:
                if not (0 <= p_idx < layer_count):
                    continue
                target_lid = (
                    copper_ids[p_idx]
                    if 0 <= p_idx < len(copper_ids) and copper_ids[p_idx] >= 0
                    else None
                )
                try:
                    rs, re_, cs, ce, local_mask = get_pad_grid_mask(
                        pad, rows, cols, x_min, y_min, res, layer_id=target_lid
                    )
                    if rs < re_ and cs < ce and np.any(local_mask):
                        vals = T[p_idx, rs:re_, cs:ce][local_mask]
                        if vals.size:
                            sampled_parts.append(vals.ravel())
                except Exception:
                    continue

            if sampled_parts:
                sampled = np.concatenate(sampled_parts)
            else:
                fallback_idx = (
                    p_indices[0]
                    if (p_indices and 0 <= p_indices[0] < layer_count)
                    else fp_layer_idx
                )
                ci = max(0, min(cols - 1, int((px - x_min) / res)))
                ri = max(0, min(rows - 1, int((py - y_min) / res)))
                sampled = np.asarray([T[fallback_idx, ri, ci]], dtype=np.float64)

            pad_vals_list.append(sampled)
            pad_max_vals.append(float(np.max(sampled)))

        if pad_vals_list:
            all_vals = np.concatenate(pad_vals_list)
            pad_max_c = float(max(pad_max_vals))
        else:
            ci = max(0, min(cols - 1, int((fp_x - x_min) / res)))
            ri = max(0, min(rows - 1, int((fp_y - y_min) / res)))
            all_vals = np.asarray([T[fp_layer_idx, ri, ci]], dtype=np.float64)
            pad_max_c = float(all_vals[0])

        comp_entries.append({
            "ref": ref,
            "value": value,
            "footprint": fp_name,
            "layer": fp_layer,
            "x_mm": fp_x,
            "y_mm": fp_y,
            "max_c": float(np.max(all_vals)),
            "mean_c": float(np.mean(all_vals)),
            "min_c": float(np.min(all_vals)),
            "pad_max_c": pad_max_c,
        })
        comp_geom.append({
            "ref": ref,
            "value": value,
            "x_mm": fp_x,
            "y_mm": fp_y,
            "pads": pad_points,
        })

    comp_entries.sort(key=lambda item: (-item["max_c"], item["ref"]))
    components_dict = {item["ref"]: item for item in comp_entries}
    return components_dict, comp_geom


def _nearest_component(
    x_mm: float,
    y_mm: float,
    comp_geom: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the closest component (and pad) to ``(x_mm, y_mm)``."""
    best = None
    best_dist_sq = float("inf")
    for comp in comp_geom:
        pads = comp["pads"]
        if pads:
            for pname, px, py in pads:
                d2 = (px - x_mm) ** 2 + (py - y_mm) ** 2
                if d2 < best_dist_sq:
                    best_dist_sq = d2
                    best = {
                        "ref": comp["ref"],
                        "value": comp["value"],
                        "distance_mm": math.sqrt(d2),
                        "nearest_pad": pname,
                    }
        else:
            d2 = (comp["x_mm"] - x_mm) ** 2 + (comp["y_mm"] - y_mm) ** 2
            if d2 < best_dist_sq:
                best_dist_sq = d2
                best = {
                    "ref": comp["ref"],
                    "value": comp["value"],
                    "distance_mm": math.sqrt(d2),
                    "nearest_pad": None,
                }
    return best


def _extract_hotspots(
    T: np.ndarray,
    amb: float,
    layer_names: List[str],
    x_min: float,
    y_min: float,
    res: float,
    comp_geom: List[Dict[str, Any]],
    max_hotspots: int = 5,
) -> List[Dict[str, Any]]:
    """Find up to ``max_hotspots`` spatially separated local temperature maxima."""
    layer_count, rows, cols = T.shape
    if rows == 0 or cols == 0 or layer_count == 0:
        return []

    work = np.max(T, axis=0).astype(np.float64, copy=True)
    rad = max(3, int(round(3.0 / max(float(res), 1e-6))))
    rad_sq = rad * rad
    hotspots: List[Dict[str, Any]] = []

    for rank in range(1, max_hotspots + 1):
        flat_idx = int(np.argmax(work))
        val = float(work.flat[flat_idx])
        if not math.isfinite(val) or val == -math.inf:
            break
        if rank > 1 and val <= amb + 1e-4:
            break
        r, c = divmod(flat_idx, cols)
        layer_idx = int(np.argmax(T[:, r, c]))
        temp_c = float(T[layer_idx, r, c])
        x_mm = float(x_min + (c + 0.5) * res)
        y_mm = float(y_min + (r + 0.5) * res)
        lname = layer_names[layer_idx] if layer_idx < len(layer_names) else f"Layer {layer_idx}"
        hotspots.append({
            "rank": rank,
            "temp_c": temp_c,
            "rise_k": temp_c - float(amb),
            "layer": lname,
            "layer_index": layer_idx,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "nearest_component": _nearest_component(x_mm, y_mm, comp_geom),
        })

        r0 = max(0, r - rad)
        r1 = min(rows, r + rad + 1)
        c0 = max(0, c - rad)
        c1 = min(cols, c + rad + 1)
        rr, cc = np.ogrid[r0:r1, c0:c1]
        mask = (rr - r) ** 2 + (cc - c) ** 2 <= rad_sq
        work[r0:r1, c0:c1][mask] = -np.inf

    return hotspots


def _evaluate_limits(
    limits_cfg: Optional[Dict[str, Any]],
    board_max_c: float,
    heatsink: Dict[str, Any],
    components: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate thermal pass/fail limits against board, heatsink, and components."""
    if not isinstance(limits_cfg, dict):
        return {"enabled": False, "passed": True, "violations": []}

    violations: List[Dict[str, Any]] = []
    enabled = False

    max_board = limits_cfg.get("max_board_c")
    if max_board is not None:
        enabled = True
        limit_f = float(max_board)
        if board_max_c > limit_f:
            violations.append({
                "rule": "max_board_c",
                "target": "board",
                "limit_c": limit_f,
                "actual_c": float(board_max_c),
                "margin_c": limit_f - float(board_max_c),
            })

    max_hs = limits_cfg.get("max_heatsink_c")
    if max_hs is not None:
        enabled = True
        if heatsink.get("enabled") and heatsink.get("temp_c") is not None:
            limit_f = float(max_hs)
            hs_temp = float(heatsink["temp_c"])
            if hs_temp > limit_f:
                violations.append({
                    "rule": "max_heatsink_c",
                    "target": "heatsink",
                    "limit_c": limit_f,
                    "actual_c": hs_temp,
                    "margin_c": limit_f - hs_temp,
                })

    comp_limits = limits_cfg.get("components")
    if isinstance(comp_limits, dict):
        for pattern, limit_val in comp_limits.items():
            if limit_val is None:
                continue
            enabled = True
            limit_f = float(limit_val)
            pat_upper = str(pattern).upper()
            for ref, comp in components.items():
                if fnmatch.fnmatchcase(ref.upper(), pat_upper):
                    actual_c = float(comp["max_c"])
                    if actual_c > limit_f:
                        violations.append({
                            "rule": "component_max_c",
                            "target": ref,
                            "pattern": str(pattern),
                            "limit_c": limit_f,
                            "actual_c": actual_c,
                            "margin_c": limit_f - actual_c,
                        })

    return {
        "enabled": enabled,
        "passed": len(violations) == 0,
        "violations": violations,
    }


def build_semantic_summary(
    T: np.ndarray,
    board: Any,
    layer_names: List[str],
    x_min: float,
    y_min: float,
    res: float,
    settings: Dict[str, Any],
    result: Any,
    grid_info: Optional[Dict[str, Any]] = None,
    electrical_summary: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    elapsed_s: float = 0.0,
    warnings: Optional[List[Any]] = None,
    board_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a semantic, JSON-serializable summary from a solved thermal field.

    Parameters
    ----------
    T : np.ndarray
        Solved temperature array of shape ``(layers, rows, cols)`` in °C.
    board : pcbnew.BOARD or None
        KiCad board object for component/pad mapping.
    layer_names : list of str
        Copper layer display names in stackup order.
    x_min, y_min, res : float
        Grid origin and spacing in millimeters.
    settings : dict
        Normalized simulation settings.
    result : SolverResult
        Result object returned by ``thermal_solver``.
    grid_info : dict, optional
        Grid/mesh metadata dictionary.
    electrical_summary : dict, optional
        Electrical Joule-heating summary dictionary, if enabled.
    artifacts : dict, optional
        Paths to generated output artifacts.
    elapsed_s : float, optional
        Total wall-clock runtime in seconds.
    warnings : list, optional
        Collected warning entries.
    board_info : dict, optional
        Pre-built board metadata dictionary.

    Returns
    -------
    dict
        Strictly JSON-serializable semantic summary.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    amb = float(settings.get("amb", 25.0))
    board_min_c = float(np.min(T_arr))
    board_max_c = float(np.max(T_arr))
    board_mean_c = float(np.mean(T_arr))

    layers_summary = []
    for idx in range(T_arr.shape[0]):
        lname = layer_names[idx] if idx < len(layer_names) else f"Layer {idx}"
        layer_slice = T_arr[idx]
        layers_summary.append({
            "index": idx,
            "name": lname,
            "min_c": float(np.min(layer_slice)),
            "max_c": float(np.max(layer_slice)),
            "mean_c": float(np.mean(layer_slice)),
        })

    k_norm = getattr(result, "k_norm_info", None) or {}
    if k_norm.get("heatsink_enabled"):
        heatsink = {
            "enabled": True,
            "mode": str(k_norm.get("heatsink_mode", "heatsink")),
            "temp_c": k_norm.get("heatsink_temp_c"),
            "temp_rise_k": k_norm.get("heatsink_temp_rise_c"),
            "rth_kw": k_norm.get("heatsink_rth_kw"),
            "capacity_j_k": k_norm.get("heatsink_capacity_j_k"),
            "tau_s": k_norm.get("heatsink_tau_s"),
            "heatsink_power_w": k_norm.get("heatsink_power_w"),
            "pcb_convection_power_w": k_norm.get("pcb_convection_power_w"),
            "heatsink_power_fraction": k_norm.get("heatsink_power_fraction"),
            "pad_flux_w": k_norm.get("heatsink_pad_flux_w"),
        }
    elif settings.get("use_heatsink"):
        inactive_reason = k_norm.get("heatsink_inactive_reason")
        mode = str(k_norm.get("heatsink_mode", settings.get("pad_sink_mode", "fixed")))
        heatsink = {
            "enabled": bool(mode == "fixed" and not inactive_reason),
            "mode": mode,
            "inactive_reason": inactive_reason,
        }
    else:
        heatsink = {"enabled": False, "mode": "none"}

    components, comp_geom = _extract_components_temperatures(
        T_arr, board, layer_names, float(x_min), float(y_min), float(res)
    )
    top_components = list(components.values())[:10]
    hotspots = _extract_hotspots(
        T_arr, amb, layer_names, float(x_min), float(y_min), float(res), comp_geom
    )
    limits_check = _evaluate_limits(
        settings.get("limits"), board_max_c, heatsink, components
    )

    if board_info is None and board is not None:
        bbox = _get_board_bbox(board)
        bx0, by0, bx1, by1 = _bbox_bounds_mm(bbox)
        try:
            bpath = str(board.GetFileName() or "")
        except Exception:
            bpath = ""
        board_info = {
            "path": bpath,
            "name": os.path.basename(bpath) if bpath else "",
            "width_mm": max(0.0, bx1 - bx0),
            "height_mm": max(0.0, by1 - by0),
            "x_min_mm": bx0,
            "y_min_mm": by0,
        }

    ginfo = grid_info or {}
    default_nodes = int(T_arr.size)
    steps_total = int(getattr(result, "step_counter", 0) or 0)
    solve_s = float(getattr(result, "total_solve_time", 0.0) or 0.0)
    solver_info = {
        "backend": k_norm.get("backend"),
        "strategy": k_norm.get("strategy"),
        "steady_state": bool(k_norm.get("steady_state", settings.get("steady_state", False))),
        "mesh_mode": ginfo.get("mesh_mode", "uniform"),
        "nodes": int(ginfo.get("equivalent_uniform_nodes", k_norm.get("N", default_nodes))),
        "adaptive_nodes": int(ginfo.get("adaptive_nodes", k_norm.get("N", default_nodes))),
        "steps_total": steps_total,
        "solve_s": solve_s,
        "total_s": float(elapsed_s),
        "pin_w": k_norm.get("pin_w"),
        "pout_w": k_norm.get("pout_final_w"),
        "steady_rel_diff": k_norm.get("steady_rel_diff"),
        "peak_rss_mb": get_process_memory_mb(peak=True),
    }

    default_artifacts = {
        "run_dir": None,
        "summary_json": None,
        "report_html": None,
        "heatmap_png": None,
        "preview_png": None,
        "joule_map_png": None,
        "current_density_png": None,
    }
    if artifacts:
        default_artifacts.update(artifacts)

    status = (
        "limit_violation"
        if (limits_check["enabled"] and not limits_check["passed"])
        else "ok"
    )

    return _json_clean({
        "schema_version": SCHEMA_VERSION,
        "plugin_version": PLUGIN_VERSION,
        "command": "run",
        "status": status,
        "board": board_info or {},
        "temperatures": {
            "ambient_c": amb,
            "min_c": board_min_c,
            "max_c": board_max_c,
            "mean_c": board_mean_c,
            "max_rise_k": board_max_c - amb,
            "layers": layers_summary,
        },
        "heatsink": heatsink,
        "hotspots": hotspots,
        "components": components,
        "top_components": top_components,
        "limits_check": limits_check,
        "electrical": electrical_summary,
        "solver": solver_info,
        "warnings": list(warnings or []),
        "errors": [],
        "artifacts": default_artifacts,
    })


def _build_geometry_context_key(grid: GridEstimate, settings: Dict[str, Any]) -> Tuple[Any, ...]:
    via_plating_um, via_fill = resolve_via_settings(settings)
    return (
        int(grid.rows),
        int(grid.cols),
        round(float(grid.x_min_mm), 6),
        round(float(grid.y_min_mm), 6),
        round(float(grid.actual_res_mm), 6),
        bool(settings.get("ignore_traces", False)),
        bool(settings.get("ignore_polygons", False)),
        bool(settings.get("limit_area", False)),
        str(settings.get("area_mode", "full")),
        round(float(settings.get("pad_dist_mm", 0.0) or 0.0), 6),
        bool(settings.get("use_heatsink", False)),
        round(float(via_plating_um), 4),
        via_fill,
    )


def run_headless_simulation(
    board_or_path: Any,
    job_config: Optional[Dict[str, Any]],
    out_dir: Optional[Union[str, Path]] = None,
    generate_plots: bool = True,
    generate_html: bool = True,
    refill_zones: bool = False,
    cached_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate and execute a thermal simulation headlessly, returning semantic JSON.

    Parameters
    ----------
    board_or_path : str or pathlib.Path or pcbnew.BOARD
        Board file path or loaded board object.
    job_config : dict or None
        Job configuration dictionary.
    out_dir : str or pathlib.Path, optional
        Output directory for ``summary.json`` and optional plots/HTML report.
    generate_plots : bool, optional
        Generate heatmap, preview, and Joule-map PNGs when ``out_dir`` is set.
    generate_html : bool, optional
        Generate ``thermal_report.html`` when ``out_dir`` is set.
    refill_zones : bool, optional
        Refill unfilled copper zones before building geometry.
    cached_context : dict, optional
        Internal preloaded board/geometry context reused across parameter sweeps.

    Returns
    -------
    dict
        Strictly JSON-serializable run summary or validation error dictionary.
    """
    run_started_at = time.perf_counter()
    effective_cfg = dict(job_config or {})
    if refill_zones:
        effective_cfg["refill_zones"] = True

    if cached_context is not None and "board" in cached_context:
        board = cached_context["board"]
        board_path = cached_context.get("board_path", "")
        copper_ids = cached_context["copper_ids"]
        layer_names = cached_context["layer_names"]
        stack_info = cached_context["stack_info"]
        bbox = cached_context["bbox"]
    else:
        board, board_path = _load_board_if_needed(board_or_path)
        copper_ids, layer_names, stack_info = _resolve_layers_and_stackup(board)
        bbox = _get_board_bbox(board)

    vctx = _validate_with_context(
        board,
        board_path,
        copper_ids,
        layer_names,
        stack_info,
        bbox,
        effective_cfg,
    )
    validation = vctx["payload"]
    if not validation["valid"]:
        return _json_clean({
            "schema_version": SCHEMA_VERSION,
            "command": "run",
            "status": "validation_error",
            "valid": False,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        })

    settings = vctx["norm"]
    area = vctx["area"]
    grid = vctx["grid"]
    warnings = list(validation["warnings"])
    unfilled_zones = int(vctx["unfilled_zones"])
    power_pads = vctx["resolved_power_objs"]
    pad_sources = vctx["pad_sources"]
    terminals = vctx["terminals"]
    buses = vctx["buses"]

    if settings["refill_zones"] and unfilled_zones > 0:
        try:
            pcbnew.ZONE_FILLER(board).Fill(board.Zones())
            if cached_context is not None:
                cached_context.pop("geometry_state", None)
                cached_context.pop("geometry_key", None)
        except Exception as exc:
            warnings.append({"code": "ZONE_REFILL_WARNING", "message": f"Zone refill warning: {exc}"})

    stackup_derived = _derive_stackup_thicknesses(board, copper_ids, stack_info, settings)
    total_thick_mm = stackup_derived["total_thick_mm_used"]
    res = grid.actual_res_mm
    x_min = grid.x_min_mm
    y_min = grid.y_min_mm
    rows = grid.rows
    cols = grid.cols
    layer_count = len(copper_ids)

    grid_info = {
        "grid_requested_res_mm": grid.requested_res_mm,
        "grid_res_mm": float(res),
        "grid_auto_coarsened": grid.auto_coarsened,
        "grid_expert_limits": grid.expert_limits,
        "grid_max_cells": grid.max_cells,
        "grid_target_cells": grid.target_cells,
        "grid_rows": int(rows),
        "grid_cols": int(cols),
        "grid_x_min_mm": float(x_min),
        "grid_y_min_mm": float(y_min),
        "grid_width_mm": float(cols * res),
        "grid_height_mm": float(rows * res),
        "grid_cell_area_mm2": float(res * res),
        "grid_detail_level": grid.detail_level,
        "grid_node_budget": grid.node_budget,
        "grid_memory_mb_low": grid.memory_mb_low,
        "grid_memory_mb_high": grid.memory_mb_high,
        "grid_runtime_class": grid.runtime_class,
        "area_mode": area.mode,
        "area_fraction": area.area_fraction,
        "area_margin_mm": area.margin_mm,
        "area_active_nets": list(area.active_net_names),
    }

    k_cu = 390.0
    k_fr4 = 0.3
    rho_cu, cp_cu = 8960.0, 385.0
    rho_fr4, cp_fr4 = 1850.0, 1100.0
    via_plating_um, via_fill = resolve_via_settings(settings)
    h_top, h_bottom = resolve_convection_settings(settings)

    cu_thick_m = [max(1e-9, th * 1e-3) for th in stackup_derived["copper_thickness_mm_used"]]
    gap_m = [max(1e-9, g * 1e-3) for g in stackup_derived["gap_mm_used"]]

    bus_pads = [pad_obj for bus in buses for pad_obj, _ in getattr(bus, "pads", [])]
    focus_pads: List[Any] = []
    _seen_focus = set()
    for p in list(power_pads) + [item.pad for item in terminals] + bus_pads:
        if id(p) not in _seen_focus:
            focus_pads.append(p)
            _seen_focus.add(id(p))
    geom_key = _build_geometry_context_key(grid, settings)
    geometry_state = None
    if (
        cached_context is not None
        and cached_context.get("geometry_state") is not None
        and cached_context.get("geometry_key") == geom_key
    ):
        geometry_state = cached_context["geometry_state"]
    if geometry_state is None:
        geometry_state = call_with_supported_kwargs(
            build_geometry_state,
            board=board,
            copper_ids=copper_ids,
            rows=rows,
            cols=cols,
            x_min=x_min,
            y_min=y_min,
            res=res,
            settings=settings,
            via_factor=k_cu / k_fr4,
            pads_list=focus_pads,
            optional_kwargs={
                "via_plating_um": via_plating_um,
                "via_fill": via_fill,
                "k_cu": k_cu,
                "k_fr4": k_fr4,
            },
        )
        if cached_context is not None:
            cached_context["geometry_state"] = geometry_state
            cached_context["geometry_key"] = geom_key

    copper_mask = geometry_state.copper_mask
    via_map_3d = getattr(geometry_state, "via_map_3d", None)
    has_via_map_3d = via_map_3d is not None
    V_map = via_map_3d if has_via_map_3d else geometry_state.via_map
    H_map = geometry_state.heatsink_mask

    dx = res * 1e-3
    dy = dx
    sim_time = float(settings["time"])
    steps_target = max(1, min(600, max(80, int(120 * (sim_time ** 0.35)))))
    dt = sim_time / steps_target
    amb = float(settings["amb"])
    pixel_area = dx * dy
    t_cu = np.asarray(cu_thick_m, dtype=np.float64)
    t_fr4_eff = _effective_fr4_control_volume_thicknesses(gap_m, total_thick_mm * 1e-3, layer_count)
    t_fr4_eff_mm = (t_fr4_eff * 1e3).tolist()

    engine_setting = str(settings.get("compute_engine", "auto") or "auto").strip().lower()
    use_fast_engine = (
        engine_setting == "fast_native"
        or (engine_setting == "auto" and grid.nodes >= _auto_fast_engine_threshold())
    )
    resolved_engine = "fast_native" if use_fast_engine else "legacy"

    use_compact_uniform = (
        use_fast_engine
        and not has_via_map_3d
        and build_compact_structured_operator is not None
        and build_structured_operator is _ORIG_BUILD_STRUCTURED_OPERATOR
    )
    use_compact_path = use_compact_uniform

    pad_cap_areal = float(settings.get("pad_cap_areal", 0.0) or 0.0)
    C = None
    if not use_compact_path:
        C = _build_dense_capacity(
            layer_count,
            rows,
            cols,
            copper_mask,
            t_cu,
            t_fr4_eff,
            pixel_area,
            rho_cu,
            cp_cu,
            rho_fr4,
            cp_fr4,
            pad_cap_areal,
            H_map,
        )

    RC = rows * cols
    N = RC * layer_count
    pad_contributions = _build_sparse_pad_contributions(
        board=board,
        copper_ids=copper_ids,
        pads_list=power_pads,
        rows=rows,
        cols=cols,
        x_min=x_min,
        y_min=y_min,
        res=res,
        area_mask=getattr(geometry_state, "area_mask", None),
    )
    has_pwl_sources = any(stype == "pwl" for stype, _ in pad_sources)
    Q = None
    Q_func = None
    Q_sparse = None
    if not use_compact_path or has_pwl_sources or SparsePowerVector is None:
        Q, Q_func = _build_power_vector(pad_sources, pad_contributions, N)

    def _serialize_net_summary(item: Any) -> Dict[str, Any]:
        data = asdict(item)
        data.pop("current_density_a_mm2", None)
        data["net"] = data.get("net_name", "")
        return data

    def _supports_kwarg(fn: Any, kw_name: str) -> bool:
        try:
            params = inspect.signature(fn).parameters
            return kw_name in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            return False

    electrical_summary = None
    electrical_result = None
    electrical_config = None
    q_joule = None
    q_base = Q
    base_q_func = Q_func
    electro_thermal_passes = 0
    initial_loss_w = 0.0
    coupled_loss_w = 0.0

    def _apply_joule_heating(q_j: np.ndarray) -> None:
        nonlocal Q, Q_func, q_base, base_q_func
        if q_base is None:
            q_base, base_q_func = _build_power_vector(pad_sources, pad_contributions, N)
        Q = q_base + q_j
        if base_q_func is not None:
            def q_func_with_joule(t, _base=base_q_func, _q_joule=q_j):
                return _base(t) + _q_joule
            Q_func = q_func_with_joule

    if settings.get("current_enabled"):
        if terminals or buses:
            electrical_config = call_with_supported_kwargs(
                ElectricalConfig,
                copper_ids=list(copper_ids),
                rows=rows,
                cols=cols,
                x_min=x_min,
                y_min=y_min,
                res=res,
                t_cu=t_cu,
                layer_names={lid: layer_names[idx] for idx, lid in enumerate(copper_ids) if idx < len(layer_names)},
                optional_kwargs={
                    "via_plating_um": via_plating_um,
                    "via_fill": via_fill,
                    "ambient_c": amb,
                    "gap_m": gap_m,
                    "equipotential_pads": True,
                },
            )
            electrical_result = call_with_supported_kwargs(
                solve_electrical_heating,
                board,
                terminals,
                electrical_config,
                optional_kwargs={"equipotential_buses": buses or None},
            )
            if not electrical_result.valid:
                return _json_clean({
                    "schema_version": SCHEMA_VERSION,
                    "command": "run",
                    "status": "validation_error",
                    "valid": False,
                    "errors": [
                        {"code": "ELECTRICAL_VALIDATION_ERROR", "message": msg}
                        for msg in electrical_result.errors
                    ],
                    "warnings": warnings,
                })
            q_joule = electrical_result.q_joule
            electro_thermal_passes = 1
            initial_loss_w = float(electrical_result.total_loss_w)
            coupled_loss_w = initial_loss_w
            if np.any(q_joule):
                _apply_joule_heating(q_joule)
            electrical_summary = {
                "total_loss_w": float(electrical_result.total_loss_w),
                "electro_thermal_passes": electro_thermal_passes,
                "initial_loss_w": initial_loss_w,
                "coupled_loss_w": coupled_loss_w,
                "warnings": list(electrical_result.warnings),
                "nets": [_serialize_net_summary(item) for item in electrical_result.net_summaries],
            }
        else:
            electrical_summary = {
                "total_loss_w": 0.0,
                "warnings": ["Current simulation enabled but no non-zero pad currents were configured."],
                "nets": [],
            }

    if Q is None and Q_func is None:
        if use_compact_path and SparsePowerVector is not None:
            Q_sparse = _build_sparse_power_vector(pad_sources, pad_contributions, N)
        else:
            Q, Q_func = _build_power_vector(pad_sources, pad_contributions, N)

    K_matrix = None
    b = None
    hA = None
    try:
        if use_compact_uniform:
            K_matrix = build_compact_structured_operator(
                layer_count,
                rows,
                cols,
                copper_mask,
                t_cu,
                t_fr4_eff,
                k_cu,
                k_fr4,
                dx,
                dy,
                V_map,
                gap_m,
                H_map,
                settings,
                amb,
                rho_cu=rho_cu,
                cp_cu=cp_cu,
                rho_fr4=rho_fr4,
                cp_fr4=cp_fr4,
                pad_cap_areal=pad_cap_areal,
            )
        else:
            H_map_float = H_map.astype(np.float64, copy=False)
            builder = build_structured_operator if use_fast_engine else build_stiffness_matrix
            K_matrix, b, hA, _ = builder(
                layer_count,
                rows,
                cols,
                copper_mask,
                t_cu,
                t_fr4_eff,
                k_cu,
                k_fr4,
                dx,
                dy,
                V_map,
                gap_m,
                H_map_float,
                settings,
                amb,
            )

        grid_info.update({
            "mesh_mode": "uniform",
            "equivalent_uniform_nodes": int(N),
        })

        if build_heatsink_model is not _ORIG_BUILD_HEATSINK_MODEL or not use_compact_path:
            H_map_float = H_map.astype(np.float64, copy=False)
            heatsink_model = build_heatsink_model(
                layer_count, rows, cols, dx, dy, H_map_float, settings, amb
            )
        elif use_compact_uniform:
            heatsink_model = K_matrix.build_heatsink_model(settings, amb)
        else:
            heatsink_model = _build_compact_heatsink_model(dx, dy, H_map, settings, amb)

        solver_heatsink_model = heatsink_model

        steady_state = bool(settings.get("steady_state", False)) and Q_func is None
        snap_times: List[float] = []
        if settings.get("snapshots") and not steady_state:
            snap_count = max(1, min(50, int(settings.get("snap_count", 5))))
            snap_times = sorted({
                sim_time * (k / (snap_count + 1))
                for k in range(1, snap_count + 1)
                if 0.0 < sim_time * (k / (snap_count + 1)) < sim_time
            })

        run_dir_str = str(out_dir) if out_dir is not None else None
        if run_dir_str is not None:
            os.makedirs(run_dir_str, exist_ok=True)

        def snapshot_callback(T_view, t_elapsed, snap_idx):
            if run_dir_str is None or not generate_plots:
                return None
            from .visualization import save_snapshot
            return save_snapshot(T_view, H_map, amb, layer_names, snap_idx, t_elapsed, out_dir=run_dir_str)

        configured_time_stepping = settings.get("time_stepping", "auto")

        config = SolverConfig(
            sim_time=sim_time,
            amb=amb,
            dt_base=dt,
            steps_target=steps_target,
            use_multi_phase=True,
            time_stepping=configured_time_stepping,
            snapshots_enabled=bool(settings.get("snapshots", False)) and not steady_state and generate_plots and run_dir_str is not None,
            snap_times=snap_times,
            steady_state=steady_state,
        )

        def _solve_thermal_once():
            try:
                if use_fast_engine:
                    solver_power = Q_sparse if Q_sparse is not None else Q
                    mf_kwargs = {}
                    if solver_heatsink_model is not None:
                        mf_kwargs["heatsink_model"] = solver_heatsink_model
                    return run_simulation_matrix_free(
                        config,
                        K_matrix,
                        C,
                        solver_power,
                        b,
                        hA,
                        None,
                        snapshot_callback,
                        Q_func=Q_func,
                        **mf_kwargs,
                    )
                legacy_kwargs = {}
                if heatsink_model is not None:
                    legacy_kwargs["heatsink_model"] = heatsink_model
                return run_simulation(
                    config,
                    K_matrix,
                    C,
                    Q,
                    b,
                    hA,
                    layer_count,
                    rows,
                    cols,
                    None,
                    snapshot_callback,
                    Q_func=Q_func,
                    **legacy_kwargs,
                )
            except Exception:
                if not (use_fast_engine and N <= 1_000_000):
                    raise
                fast_tb = traceback.format_exc()
                H_map_float = H_map.astype(np.float64, copy=False)
                fallback_C = (
                    C
                    if C is not None
                    else _build_dense_capacity(
                        layer_count,
                        rows,
                        cols,
                        copper_mask,
                        t_cu,
                        t_fr4_eff,
                        pixel_area,
                        rho_cu,
                        cp_cu,
                        rho_fr4,
                        cp_fr4,
                        pad_cap_areal,
                        H_map,
                    )
                )
                if Q is not None:
                    fallback_Q, fallback_Q_func = Q, Q_func
                elif Q_sparse is not None:
                    fallback_Q, fallback_Q_func = Q_sparse.to_dense(), None
                else:
                    fallback_Q, fallback_Q_func = _build_power_vector(pad_sources, pad_contributions, N)
                fb_K, fb_b, fb_hA, _ = build_stiffness_matrix(
                    layer_count,
                    rows,
                    cols,
                    copper_mask,
                    t_cu,
                    t_fr4_eff,
                    k_cu,
                    k_fr4,
                    dx,
                    dy,
                    V_map,
                    gap_m,
                    H_map_float,
                    settings,
                    amb,
                )
                fb_config = SolverConfig(
                    sim_time=sim_time,
                    amb=amb,
                    dt_base=dt,
                    steps_target=steps_target,
                    use_multi_phase=True,
                    time_stepping=settings.get("time_stepping", "auto"),
                    snapshots_enabled=False,
                    snap_times=[],
                    steady_state=steady_state,
                )
                fb_hs = (
                    heatsink_model
                    if (heatsink_model is not None and not getattr(heatsink_model, "bottom_layer_only", False))
                    else build_heatsink_model(layer_count, rows, cols, dx, dy, H_map_float, settings, amb)
                )
                fb_kwargs = {"heatsink_model": fb_hs} if fb_hs is not None else {}
                fb_res = run_simulation(
                    fb_config,
                    fb_K,
                    fallback_C,
                    fallback_Q,
                    fb_b,
                    fb_hA,
                    layer_count,
                    rows,
                    cols,
                    None,
                    None,
                    Q_func=fallback_Q_func,
                    **fb_kwargs,
                )
                fb_res.k_norm_info.update({
                    "compute_engine": "legacy_fallback",
                    "fast_engine_error": fast_tb,
                })
                grid_info.update({
                    "mesh_mode": "uniform",
                    "equivalent_uniform_nodes": int(N),
                })
                return fb_res

        result = _solve_thermal_once()

        # 2-pass (up to 3-pass) electro-thermal coupling: update rho_cu(T)
        if (
            settings.get("current_enabled")
            and settings.get("electro_thermal_coupling", True)
            and electrical_result is not None
            and electrical_result.valid
            and initial_loss_w > 0.0
            and q_joule is not None
            and np.any(q_joule)
            and not getattr(result, "aborted", False)
            and (float(np.max(result.T)) - amb) > 5.0
            and _supports_kwarg(solve_electrical_heating, "temperature_field")
        ):
            did_not_converge = False
            for pass_idx in (2, 3):
                next_electrical = call_with_supported_kwargs(
                    solve_electrical_heating,
                    board,
                    terminals,
                    electrical_config,
                    optional_kwargs={
                        "equipotential_buses": buses or None,
                        "temperature_field": result.T,
                    },
                )
                if not next_electrical.valid:
                    break
                loss_prev = coupled_loss_w
                loss_new = float(next_electrical.total_loss_w)
                electrical_result = next_electrical
                coupled_loss_w = loss_new
                electro_thermal_passes = pass_idx
                q_joule = electrical_result.q_joule
                _apply_joule_heating(q_joule)
                result = _solve_thermal_once()
                if getattr(result, "aborted", False):
                    break
                rel_change = abs(loss_new - loss_prev) / max(loss_prev, 1e-9)
                if rel_change <= 0.02:
                    break
                if pass_idx == 3:
                    did_not_converge = True

            et_warnings = list(electrical_result.warnings)
            if initial_loss_w > 0.0 and coupled_loss_w > 1.5 * initial_loss_w:
                msg = (
                    f"Electro-thermal coupling increased copper Joule loss by more than 50% "
                    f"({initial_loss_w:.4f} W -> {coupled_loss_w:.4f} W): thermal runaway risk."
                )
                et_warnings.append(msg)
                warnings.append({"code": "ELECTRO_THERMAL_RUNAWAY_RISK", "message": msg})
            if did_not_converge:
                msg = (
                    "Electro-thermal coupling did not converge within 3 passes "
                    "(thermal runaway risk)."
                )
                et_warnings.append(msg)
                warnings.append({"code": "ELECTRO_THERMAL_NONCONVERGENCE", "message": msg})

            electrical_summary = {
                "total_loss_w": float(electrical_result.total_loss_w),
                "electro_thermal_passes": int(electro_thermal_passes),
                "initial_loss_w": float(initial_loss_w),
                "coupled_loss_w": float(coupled_loss_w),
                "warnings": et_warnings,
                "nets": [_serialize_net_summary(item) for item in electrical_result.net_summaries],
            }
    finally:
        if K_matrix is not None and hasattr(K_matrix, "clear_caches"):
            K_matrix.clear_caches()
        clear_native_scratch()

    result.k_norm_info.update({
        **grid_info,
        "t_fr4_eff_min": float(np.min(t_fr4_eff)),
        "t_fr4_eff_max": float(np.max(t_fr4_eff)),
        "t_fr4_eff_per_plane_mm": t_fr4_eff_mm,
        "pad_cap_input_areal": pad_cap_areal,
        "h_top": h_top,
        "h_air_bottom": h_bottom,
        "via_plating_um": via_plating_um,
        "via_fill": via_fill,
        "compute_engine": result.k_norm_info.get("compute_engine", resolved_engine),
        "electrical_summary": electrical_summary,
    })
    if settings.get("use_heatsink"):
        pad_sink_mode = str(settings.get("pad_sink_mode", "fixed") or "fixed").strip().lower()
        if pad_sink_mode == "fixed":
            result.k_norm_info.update({
                "heatsink_mode": "fixed",
                "pad_contact_factor": float(settings.get("pad_contact_factor", 0.2)),
            })
        elif pad_sink_mode == "heatsink" and not np.any(H_map):
            result.k_norm_info.update({
                "heatsink_enabled": False,
                "heatsink_mode": "heatsink",
                "heatsink_inactive_reason": "No User.Eco1 thermal pad shapes found on board",
            })

    artifacts = {
        "run_dir": str(out_dir) if out_dir is not None else None,
        "summary_json": None,
        "report_html": None,
        "heatmap_png": None,
        "preview_png": None,
        "joule_map_png": None,
        "current_density_png": None,
    }

    if out_dir is not None:
        out_dir_str = str(out_dir)
        os.makedirs(out_dir_str, exist_ok=True)
        heatmap_path = None
        preview_path = None
        joule_map_path = None
        current_density_path = None

        if generate_plots:
            from .visualization import (
                save_current_density_map,
                save_joule_loss_map,
                save_preview_image,
                show_results_all_layers,
                show_results_top_bot,
            )

            if settings["show_all"]:
                heatmap_path = show_results_all_layers(
                    result.T, H_map, amb, layer_names,
                    open_file=False, t_elapsed=sim_time, out_dir=out_dir_str,
                )
            else:
                heatmap_path = show_results_top_bot(
                    result.T, H_map, amb,
                    open_file=False, t_elapsed=sim_time, out_dir=out_dir_str,
                )

            if q_joule is not None and np.any(q_joule):
                joule_map_path = save_joule_loss_map(
                    q_joule,
                    layer_count=layer_count,
                    rows=rows,
                    cols=cols,
                    layer_names=layer_names,
                    x_min_mm=x_min,
                    y_min_mm=y_min,
                    res_mm=res,
                    electrical_summary=electrical_summary,
                    out_dir=out_dir_str,
                )
            current_density_map = getattr(electrical_result, "current_density_map", None) if electrical_result is not None else None
            if current_density_map is not None and np.any(current_density_map):
                current_density_path = save_current_density_map(
                    current_density_map,
                    layer_names=layer_names,
                    out_dir=out_dir_str,
                )

            def _derive_wrapper(b, cids, sinfo, st):
                return _derive_stackup_thicknesses(b, cids, sinfo, st)

            preview_path = save_preview_image(
                board,
                copper_ids,
                bbox,
                focus_pads,
                settings,
                layer_names,
                stack_info,
                get_pad_pixels,
                None,
                _derive_wrapper,
                open_file=False,
                out_dir=out_dir_str,
                geometry_state=geometry_state,
                grid_spec=grid,
            )
            artifacts["heatmap_png"] = heatmap_path
            artifacts["preview_png"] = preview_path
            artifacts["joule_map_png"] = joule_map_path
            artifacts["current_density_png"] = current_density_path

        if generate_html:
            from .thermal_report import write_html_report
            from .visualization import build_interactive_heatmap_payload

            interactive_heatmap = build_interactive_heatmap_payload(
                result.T,
                amb=amb,
                layer_names=layer_names,
                res_mm=res,
                x_min_mm=x_min,
                y_min_mm=y_min,
                show_all=bool(settings.get("show_all", True)),
            )
            pad_power_rows = []
            for p_item in vctx["resolved_power_pads"]:
                pad_power_rows.append((p_item["name"], p_item["power"]))
            snapshot_debug = {
                "snapshots_enabled": settings.get("snapshots"),
                "snap_count": settings.get("snap_count"),
                "dt_base": dt,
                "steps_target": steps_target,
                "steps_total": result.step_counter,
                "snap_times": snap_times,
                "base_output_dir": out_dir_str,
                "run_dir": out_dir_str,
                "solver_backend": result.k_norm_info.get("backend"),
                "avg_solve_s": result.total_solve_time / max(result.step_counter, 1),
                "factorizations": result.factor_count,
                "factorization_s": result.total_factor_time,
                "phase_metrics": json.dumps(result.phase_metrics),
            }
            report_path = write_html_report(
                settings=settings,
                stack_info=stack_info,
                stackup_derived=stackup_derived,
                pad_power=pad_power_rows,
                layer_names=layer_names,
                preview_path=preview_path,
                heatmap_path=heatmap_path,
                k_norm_info=result.k_norm_info,
                out_dir=out_dir_str,
                snapshot_debug=snapshot_debug,
                snapshot_files=result.snapshot_files,
                interactive_heatmap=interactive_heatmap,
                electrical_summary=electrical_summary,
                joule_map_path=joule_map_path,
                current_density_path=current_density_path,
            )
            artifacts["report_html"] = report_path

        artifacts["summary_json"] = os.path.join(out_dir_str, "summary.json")

    elapsed_s = time.perf_counter() - run_started_at
    bx0, by0, bx1, by1 = _bbox_bounds_mm(bbox)
    board_info = {
        "path": board_path,
        "name": os.path.basename(board_path) if board_path else "",
        "width_mm": max(0.0, bx1 - bx0),
        "height_mm": max(0.0, by1 - by0),
        "x_min_mm": bx0,
        "y_min_mm": by0,
    }

    summary = build_semantic_summary(
        T=result.T,
        board=board,
        layer_names=layer_names,
        x_min=x_min,
        y_min=y_min,
        res=res,
        settings=settings,
        result=result,
        grid_info=grid_info,
        electrical_summary=electrical_summary,
        artifacts=artifacts,
        elapsed_s=elapsed_s,
        warnings=warnings,
        board_info=board_info,
    )

    if artifacts["summary_json"] is not None:
        with open(artifacts["summary_json"], "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)

    return summary


def run_parameter_sweep(
    board_or_path: Any,
    base_config: Optional[Dict[str, Any]],
    param: str,
    values: List[Any],
    out_dir: Optional[Union[str, Path]] = None,
    generate_plots: bool = False,
    refill_zones: bool = False,
) -> Dict[str, Any]:
    """Sweep a single parameter across ``values``, reusing board & geometry state.

    Parameters
    ----------
    board_or_path : str or pathlib.Path or pcbnew.BOARD
        Board file path or loaded board object.
    base_config : dict or None
        Baseline job configuration.
    param : str
        Parameter name from ``SUPPORTED_SWEEP_PARAMS``.
    values : list
        Parameter values to evaluate.
    out_dir : str or pathlib.Path, optional
        Output directory for per-run subdirectories and ``sweep_summary.json``.
    generate_plots : bool, optional
        Whether to generate PNG plots for each run (default False).
    refill_zones : bool, optional
        Whether to refill unfilled copper zones once before sweeping.

    Returns
    -------
    dict
        Strictly JSON-serializable sweep summary.
    """
    sweep_start = time.perf_counter()
    if param not in SUPPORTED_SWEEP_PARAMS:
        return _json_clean({
            "schema_version": SCHEMA_VERSION,
            "command": "sweep",
            "status": "validation_error",
            "valid": False,
            "errors": [{
                "code": "INVALID_SWEEP_PARAM",
                "target": str(param),
                "message": f"Unsupported sweep parameter '{param}'.",
                "hint": f"Supported parameters: {', '.join(sorted(SUPPORTED_SWEEP_PARAMS))}",
            }],
            "warnings": [],
        })

    if not values:
        return _json_clean({
            "schema_version": SCHEMA_VERSION,
            "command": "sweep",
            "status": "validation_error",
            "valid": False,
            "errors": [{
                "code": "EMPTY_SWEEP_VALUES",
                "target": str(param),
                "message": "Sweep values list must not be empty.",
            }],
            "warnings": [],
        })

    board, board_path = _load_board_if_needed(board_or_path)
    copper_ids, layer_names, stack_info = _resolve_layers_and_stackup(board)
    bbox = _get_board_bbox(board)

    cached_context: Dict[str, Any] = {
        "board": board,
        "board_path": board_path,
        "copper_ids": copper_ids,
        "layer_names": layer_names,
        "stack_info": stack_info,
        "bbox": bbox,
    }

    runs_out: List[Dict[str, Any]] = []
    out_root = str(out_dir) if out_dir is not None else None
    if out_root is not None:
        os.makedirs(out_root, exist_ok=True)

    for idx, raw_val in enumerate(values):
        val = float(raw_val)
        point_cfg = dict(base_config or {})
        point_cfg[param] = val
        if param == "time" and "steady_state" not in point_cfg:
            point_cfg["steady_state"] = False
        if refill_zones:
            point_cfg["refill_zones"] = True

        point_dir = os.path.join(out_root, f"run_{idx:03d}") if out_root is not None else None
        run_res = run_headless_simulation(
            board,
            point_cfg,
            out_dir=point_dir,
            generate_plots=generate_plots,
            generate_html=False,
            refill_zones=(refill_zones and idx == 0),
            cached_context=cached_context,
        )
        if run_res.get("status") == "validation_error":
            return _json_clean({
                "schema_version": SCHEMA_VERSION,
                "command": "sweep",
                "status": "validation_error",
                "valid": False,
                "errors": run_res.get("errors", []),
                "warnings": run_res.get("warnings", []),
            })

        top_comps = run_res.get("top_components") or []
        top_comp = (
            {"ref": top_comps[0]["ref"], "max_c": top_comps[0]["max_c"]}
            if top_comps
            else None
        )
        runs_out.append({
            "value": val,
            "status": run_res["status"],
            "max_temp_c": run_res["temperatures"]["max_c"],
            "heatsink_temp_c": run_res["heatsink"].get("temp_c"),
            "passed": run_res["limits_check"]["passed"],
            "violations_count": len(run_res["limits_check"]["violations"]),
            "top_component": top_comp,
            "solve_s": run_res["solver"]["solve_s"],
            "summary_json": run_res["artifacts"].get("summary_json"),
        })

    best_run = min(runs_out, key=lambda r: r["max_temp_c"])
    first_passing = next((r["value"] for r in runs_out if r["passed"]), None)

    sweep_summary = _json_clean({
        "schema_version": SCHEMA_VERSION,
        "command": "sweep",
        "status": "ok",
        "param": param,
        "values": [float(v) for v in values],
        "runs": runs_out,
        "best_value": best_run["value"],
        "min_max_temp_c": best_run["max_temp_c"],
        "first_passing_value": first_passing,
        "total_s": time.perf_counter() - sweep_start,
    })

    if out_root is not None:
        sweep_path = os.path.join(out_root, "sweep_summary.json")
        with open(sweep_path, "w", encoding="utf-8") as handle:
            json.dump(sweep_summary, handle, indent=2, allow_nan=False)

    return sweep_summary


def _load_run_summary(run_ref: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
    """Load a run summary from a dict, ``summary.json`` path, or run directory."""
    if isinstance(run_ref, dict):
        return run_ref
    path = Path(run_ref)
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Summary JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_runs(
    run_a: Union[Dict[str, Any], str, Path],
    run_b: Union[Dict[str, Any], str, Path],
) -> Dict[str, Any]:
    """Compare two simulation runs (``delta = B - A``) and return structured deltas.

    Parameters
    ----------
    run_a, run_b : dict or str or pathlib.Path
        Run summary dict, path to ``summary.json``, or directory containing ``summary.json``.

    Returns
    -------
    dict
        Strictly JSON-serializable comparison summary.
    """
    a = _load_run_summary(run_a)
    b = _load_run_summary(run_b)

    ta = a.get("temperatures") or {}
    tb = b.get("temperatures") or {}

    delta_max = float(tb.get("max_c", 0.0)) - float(ta.get("max_c", 0.0))
    delta_min = float(tb.get("min_c", 0.0)) - float(ta.get("min_c", 0.0))
    delta_mean = float(tb.get("mean_c", 0.0)) - float(ta.get("mean_c", 0.0))

    hs_a = (a.get("heatsink") or {}).get("temp_c")
    hs_b = (b.get("heatsink") or {}).get("temp_c")
    delta_heatsink = (
        float(hs_b) - float(hs_a)
        if (hs_a is not None and hs_b is not None)
        else None
    )

    layers_a = {
        str(item.get("name", item.get("index"))): item
        for item in (ta.get("layers") or [])
    }
    layers_b = {
        str(item.get("name", item.get("index"))): item
        for item in (tb.get("layers") or [])
    }
    layer_deltas = []
    seen_layers = []
    for item in (ta.get("layers") or []) + (tb.get("layers") or []):
        lname = str(item.get("name", item.get("index")))
        if lname not in seen_layers:
            seen_layers.append(lname)
    for lname in seen_layers:
        la = layers_a.get(lname)
        lb = layers_b.get(lname)
        if la is not None and lb is not None:
            layer_deltas.append({
                "name": lname,
                "a_max_c": float(la["max_c"]),
                "b_max_c": float(lb["max_c"]),
                "delta_max_c": float(lb["max_c"]) - float(la["max_c"]),
                "a_mean_c": float(la["mean_c"]),
                "b_mean_c": float(lb["mean_c"]),
                "delta_mean_c": float(lb["mean_c"]) - float(la["mean_c"]),
            })

    comps_a = a.get("components") or {}
    comps_b = b.get("components") or {}
    comp_deltas = []
    for ref in set(comps_a.keys()) & set(comps_b.keys()):
        ca = comps_a[ref]
        cb = comps_b[ref]
        d_max = float(cb["max_c"]) - float(ca["max_c"])
        comp_deltas.append({
            "ref": ref,
            "a_max_c": float(ca["max_c"]),
            "b_max_c": float(cb["max_c"]),
            "delta_max_c": d_max,
        })
    comp_deltas.sort(key=lambda item: (-abs(item["delta_max_c"]), item["ref"]))
    comp_deltas = comp_deltas[:20]

    lim_a = a.get("limits_check") or {}
    lim_b = b.get("limits_check") or {}
    viols_a = {
        (v.get("rule"), v.get("target")): v
        for v in (lim_a.get("violations") or [])
    }
    viols_b = {
        (v.get("rule"), v.get("target")): v
        for v in (lim_b.get("violations") or [])
    }
    resolved_violations = [viols_a[k] for k in viols_a if k not in viols_b]
    new_violations = [viols_b[k] for k in viols_b if k not in viols_a]

    return _json_clean({
        "schema_version": SCHEMA_VERSION,
        "command": "compare",
        "status": "ok",
        "delta_max_temp_c": delta_max,
        "delta_min_temp_c": delta_min,
        "delta_mean_temp_c": delta_mean,
        "delta_heatsink_temp_c": delta_heatsink,
        "layer_deltas": layer_deltas,
        "component_deltas": comp_deltas,
        "limits_comparison": {
            "a_passed": bool(lim_a.get("passed", True)),
            "b_passed": bool(lim_b.get("passed", True)),
            "resolved_violations": resolved_violations,
            "new_violations": new_violations,
        },
    })
