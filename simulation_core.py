"""Shared zero-GUI simulation setup, grid, cost, stackup, and power helpers.

This module is the single source of truth for board-geometry estimation,
grid policy/coarsening, calibrated solver cost estimation, stackup thickness
derivation, capacity assembly, and sparse pad power distribution shared by
``thermal_plugin.py`` and ``headless_api.py``.

No GUI or ``wx`` modules may be imported here.
"""

import inspect
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pcbnew

from .capabilities import HAS_NATIVE_COMPACT, HAS_NATIVE_MG
from .electrical_solver import (
    CurrentTerminal,
    net_key_from_obj,
    net_key_from_values,
)
from .geometry_mapper import (
    get_pad_copper_layer_indices,
    get_pad_power_indices,
)
from .thermal_solver import (
    HeatsinkModel,
    SparsePowerVector,
    compute_heatsink_capacity_j_k,
)
from .workflow import AreaEstimate, GridEstimate

GRID_DETAIL_PRESETS: Dict[str, Tuple[int, int]] = {
    "fast": (1_000_000, 850_000),
    "balanced": (3_000_000, 2_550_000),
    "detailed": (8_000_000, 6_800_000),
    "very_detailed": (16_000_000, 13_600_000),
}
DEFAULT_GRID_DETAIL = "balanced"
DEFAULT_GRID_NODE_BUDGET = GRID_DETAIL_PRESETS[DEFAULT_GRID_DETAIL][0]
DEFAULT_GRID_MAX_CELLS = 200_000
DEFAULT_GRID_TARGET_CELLS = 100_000
MAX_CUSTOM_GRID_NODES = 100_000_000


@dataclass
class SparsePadContribution:
    """Sparse per-pad unit power distribution on the simulation grid."""

    indices: np.ndarray
    weights: np.ndarray


def auto_fast_engine_threshold(has_native_mg: bool) -> int:
    """Return the node threshold for switching Auto compute_engine to matrix-free."""
    return 50_000 if has_native_mg else 250_000


def resolve_mesh_mode(
    settings: Dict[str, Any],
    use_fast_engine: bool,
    node_count: int,
    *,
    has_native_mg: Optional[bool] = None,
) -> str:
    """Return the active spatial mesh mode (always ``'uniform'`` since v0.4.7)."""
    _ = (settings, use_fast_engine, node_count, has_native_mg)
    return "uniform"


def normalize_via_fill(value: Any) -> str:
    """Normalize a via-fill setting into ``'copper'`` or ``'plated'``."""
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"copper", "copper_filled"}:
        return "copper"
    return "plated"


def resolve_via_settings(settings: Optional[Dict[str, Any]]) -> Tuple[float, str]:
    """Return clamped ``(via_plating_um, via_fill)`` from simulation settings."""
    cfg = settings or {}
    raw_plating = cfg.get("via_plating_um")
    via_plating_um = 25.0 if raw_plating is None else float(raw_plating)
    via_plating_um = max(5.0, min(100.0, via_plating_um))
    via_fill = normalize_via_fill(cfg.get("via_fill"))
    return via_plating_um, via_fill


def resolve_convection_settings(settings: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """Return ``(h_conv_top, h_conv_bottom)`` in W/(m^2*K), >= 0.0."""
    cfg = settings or {}
    raw_top = cfg.get("h_conv")
    h_top = max(0.0, float(10.0 if raw_top is None else raw_top))
    raw_bot = cfg.get("h_conv_bottom")
    h_bot = h_top if raw_bot is None else max(0.0, float(raw_bot))
    return h_top, h_bot


def call_with_supported_kwargs(
    fn: Callable[..., Any],
    *args: Any,
    optional_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Invoke ``fn`` passing ``optional_kwargs`` only if accepted by its signature."""
    call_kwargs = dict(kwargs)
    if optional_kwargs:
        try:
            params = inspect.signature(fn).parameters
            accepts_var_kw = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
            )
            for key, value in optional_kwargs.items():
                if value is None and key in {"progress_callback", "cancel_check"}:
                    continue
                if accepts_var_kw or key in params:
                    call_kwargs[key] = value
        except (TypeError, ValueError):
            pass
    return fn(*args, **call_kwargs)


def resolve_grid_limits(settings: Dict[str, Any]) -> Tuple[bool, int, int]:
    """Resolve legacy automatic grid coarsening limits from settings."""
    expert_enabled = bool(settings.get("grid_expert_limits", False))
    if not expert_enabled:
        return False, DEFAULT_GRID_MAX_CELLS, DEFAULT_GRID_TARGET_CELLS

    try:
        max_cells = int(settings.get("grid_max_cells", DEFAULT_GRID_MAX_CELLS))
        target_cells = int(settings.get("grid_target_cells", DEFAULT_GRID_TARGET_CELLS))
    except Exception:
        return False, DEFAULT_GRID_MAX_CELLS, DEFAULT_GRID_TARGET_CELLS

    if max_cells < 1000 or target_cells < 1000 or target_cells > max_cells:
        return False, DEFAULT_GRID_MAX_CELLS, DEFAULT_GRID_TARGET_CELLS
    return True, max_cells, target_cells


def resolve_grid_policy(
    settings: Dict[str, Any],
    layer_count: int,
) -> Tuple[str, bool, int, int, int]:
    """Resolve a layer-aware detail preset to 2D cell limits and total node budget."""
    layers = max(1, int(layer_count or 1))
    if "grid_detail_level" not in settings:
        expert, max_cells, target_cells = resolve_grid_limits(settings)
        detail = "custom" if expert else "legacy"
        return detail, expert, max_cells, target_cells, max_cells * layers

    detail = str(settings.get("grid_detail_level") or DEFAULT_GRID_DETAIL).strip().lower()
    detail = detail.replace(" ", "_").replace("-", "_")
    if detail not in set(GRID_DETAIL_PRESETS) | {"custom"}:
        detail = DEFAULT_GRID_DETAIL

    if detail == "custom":
        try:
            max_nodes = int(settings.get("grid_node_budget", DEFAULT_GRID_NODE_BUDGET))
        except Exception:
            max_nodes = DEFAULT_GRID_NODE_BUDGET
        max_nodes = min(MAX_CUSTOM_GRID_NODES, max(50_000, max_nodes))
        target_nodes = max(25_000, int(max_nodes * 0.85))
        expert = True
    else:
        max_nodes, target_nodes = GRID_DETAIL_PRESETS[detail]
        expert = False

    max_cells = max(1000, max_nodes // layers)
    target_cells = max(1000, min(max_cells, target_nodes // layers))
    return detail, expert, max_cells, target_cells, max_nodes


def coarsen_grid_resolution(
    width_mm: float,
    height_mm: float,
    requested_res: float,
    settings: Dict[str, Any],
    layer_count: int = 1,
) -> Tuple[float, bool, bool, int, int]:
    """Apply automatic grid coarsening when a board exceeds the active cell budget."""
    _, expert_enabled, max_cells, target_cells, _ = resolve_grid_policy(
        settings, layer_count
    )
    area = float(width_mm) * float(height_mm)
    res = float(requested_res)
    auto_coarsened = False
    if res > 0.0 and area > 0.0 and (float(width_mm) / res) * (float(height_mm) / res) > max_cells:
        res = math.sqrt(area / float(target_cells))
        auto_coarsened = True
    return res, auto_coarsened, expert_enabled, max_cells, target_cells


def estimate_solver_cost(
    nodes: int,
    settings: Dict[str, Any],
    *,
    has_native_mg: Optional[bool] = None,
    has_native_compact: Optional[bool] = None,
) -> Tuple[int, int, str]:
    """Return calibrated memory bounds (MB) and a relative runtime class."""
    nodes = max(0, int(nodes))
    mg_enabled = HAS_NATIVE_MG if has_native_mg is None else bool(has_native_mg)
    compact_enabled = (
        HAS_NATIVE_COMPACT if has_native_compact is None else bool(has_native_compact)
    )
    compute_engine = str(
        settings.get("compute_engine", "auto") or "auto"
    ).strip().lower()
    fast_engine = (
        compute_engine == "fast_native"
        or (compute_engine == "auto" and nodes >= auto_fast_engine_threshold(mg_enabled))
    )
    def _ceil_mb(mb_per_million: int) -> int:
        return (nodes * mb_per_million + 999_999) // 1_000_000

    if fast_engine:
        if compact_enabled and mg_enabled:
            # Calibrated against real-world measurements (6M -> ~0.5-0.75 GB,
            # 25.6M -> ~2.0 GB solver RSS on Windows/Linux).
            memory_low = _ceil_mb(80)
            memory_high = _ceil_mb(125)
            thresholds = (2_000_000, 8_000_000, 20_000_000)
        elif mg_enabled:
            memory_low = _ceil_mb(130)
            memory_high = _ceil_mb(220)
            thresholds = (800_000, 3_000_000, 8_000_000)
        else:
            memory_low = _ceil_mb(110)
            memory_high = _ceil_mb(190)
            thresholds = (500_000, 2_000_000, 7_000_000)
    else:
        memory_low = _ceil_mb(350)
        memory_high = _ceil_mb(1200)
        thresholds = (150_000, 500_000, 1_000_000)

    if nodes < thresholds[0]:
        runtime = "Fast"
    elif nodes < thresholds[1]:
        runtime = "Moderate"
    elif nodes < thresholds[2]:
        runtime = "Slow"
    else:
        runtime = "Very slow"
    return memory_low, memory_high, runtime


def bbox_bounds_mm(bbox: Any) -> Tuple[float, float, float, float]:
    """Return a KiCad rectangle as ``(x_min, y_min, x_max, y_max)`` in mm."""
    x_min = float(bbox.GetX()) * 1e-6
    y_min = float(bbox.GetY()) * 1e-6
    return (
        x_min,
        y_min,
        x_min + float(bbox.GetWidth()) * 1e-6,
        y_min + float(bbox.GetHeight()) * 1e-6,
    )


def calculate_suggested_resolution(w_mm: float, h_mm: float, layer_count: int) -> float:
    """Calculate suggested grid resolution in mm based on board size and layer count."""
    target_nodes = 25_000 if layer_count <= 2 else 15_000
    area = max(0.0, float(w_mm)) * max(0.0, float(h_mm))
    if area > 0.0:
        suggested_res = round(math.sqrt(area / target_nodes), 2)
        if suggested_res < 0.2:
            suggested_res = 0.2
        return float(suggested_res)
    return 0.5


def estimate_simulation_area(
    board: Any,
    bbox: Any,
    settings: Dict[str, Any],
    power_pads: Optional[List[Any]] = None,
    terminals: Optional[List[CurrentTerminal]] = None,
) -> AreaEstimate:
    """Build a safe rectangular domain around heat sources and current nets."""
    board_x0, board_y0, board_x1, board_y1 = bbox_bounds_mm(bbox)
    board_w = max(0.0, board_x1 - board_x0)
    board_h = max(0.0, board_y1 - board_y0)
    legacy_limited = bool(settings.get("limit_area", False))
    mode = str(settings.get("area_mode") or ("active" if legacy_limited else "full"))
    mode = mode.strip().lower()
    if mode not in ("full", "active"):
        mode = "active" if legacy_limited else "full"
    margin = max(
        0.0,
        float(settings.get("area_margin_mm", settings.get("pad_dist_mm", 0.0)) or 0.0),
    )

    active_terms = [
        item for item in (terminals or []) if abs(float(getattr(item, "current_a", 0.0))) > 0.0
    ]
    active_keys = {
        net_key_from_values(item.net_code, item.net_name) for item in active_terms
    }
    active_names = tuple(sorted({
        str(item.net_name or net_key_from_values(item.net_code, item.net_name))
        for item in active_terms
    }))

    if mode == "full":
        return AreaEstimate(
            mode="full",
            x_min_mm=board_x0,
            y_min_mm=board_y0,
            width_mm=board_w,
            height_mm=board_h,
            board_width_mm=board_w,
            board_height_mm=board_h,
            margin_mm=margin,
            heat_source_count=len(power_pads or []),
            active_net_names=active_names,
        )

    bounds: List[Tuple[float, float, float, float]] = []
    warnings: List[str] = []
    collection_failed = False

    def add_bbox(item: Any) -> None:
        try:
            item_bbox = item.GetBoundingBox() if hasattr(item, "GetBoundingBox") else item
            x0, y0, x1, y1 = bbox_bounds_mm(item_bbox)
            if x1 > x0 and y1 > y0:
                bounds.append((x0, y0, x1, y1))
        except Exception:
            return

    for pad in power_pads or []:
        add_bbox(pad)

    if active_keys:
        try:
            footprints = list(
                board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints()
            )
            for footprint in footprints:
                for pad in footprint.Pads():
                    if net_key_from_obj(pad)[0] in active_keys:
                        add_bbox(pad)
        except Exception:
            collection_failed = True

        try:
            tracks = list(board.Tracks() if hasattr(board, "Tracks") else board.GetTracks())
            for track in tracks:
                if net_key_from_obj(track)[0] in active_keys:
                    add_bbox(track)
        except Exception:
            collection_failed = True

        try:
            zones = list(board.Zones() if hasattr(board, "Zones") else board.GetZones())
            for zone in zones:
                if net_key_from_obj(zone)[0] not in active_keys:
                    continue
                if hasattr(zone, "IsFilled") and not zone.IsFilled():
                    continue
                add_bbox(zone)
        except Exception:
            collection_failed = True

    if settings.get("use_heatsink"):
        try:
            for drawing in list(board.GetDrawings() if hasattr(board, "GetDrawings") else []):
                if drawing.GetLayer() == pcbnew.Eco1_User:
                    add_bbox(drawing)
        except Exception:
            warnings.append("Thermal-pad geometry could not be included in the area estimate.")

    if collection_failed and active_keys:
        warnings.append("Current-net geometry could not be inspected safely; the full board is used.")
        return AreaEstimate(
            mode="full",
            x_min_mm=board_x0,
            y_min_mm=board_y0,
            width_mm=board_w,
            height_mm=board_h,
            board_width_mm=board_w,
            board_height_mm=board_h,
            margin_mm=margin,
            heat_source_count=len(power_pads or []),
            active_net_names=active_names,
            fallback_to_full=True,
            warnings=tuple(warnings),
        )

    if not bounds:
        warnings.append("No active source geometry was found; the full board is used.")
        return AreaEstimate(
            mode="full",
            x_min_mm=board_x0,
            y_min_mm=board_y0,
            width_mm=board_w,
            height_mm=board_h,
            board_width_mm=board_w,
            board_height_mm=board_h,
            margin_mm=margin,
            heat_source_count=0,
            active_net_names=active_names,
            fallback_to_full=True,
            warnings=tuple(warnings),
        )

    safety = max(0.0, float(settings.get("res", 0.0) or 0.0))
    x0 = max(board_x0, min(item[0] for item in bounds) - margin - safety)
    y0 = max(board_y0, min(item[1] for item in bounds) - margin - safety)
    x1 = min(board_x1, max(item[2] for item in bounds) + margin + safety)
    y1 = min(board_y1, max(item[3] for item in bounds) + margin + safety)
    area = AreaEstimate(
        mode="active",
        x_min_mm=x0,
        y_min_mm=y0,
        width_mm=max(0.0, x1 - x0),
        height_mm=max(0.0, y1 - y0),
        board_width_mm=board_w,
        board_height_mm=board_h,
        margin_mm=margin,
        heat_source_count=len(power_pads or []),
        active_net_names=active_names,
        warnings=tuple(warnings),
    )
    if active_keys and area.area_fraction >= 0.95:
        warnings.append("Active current geometry covers almost the full board; area limiting saves little work.")
        area = AreaEstimate(
            mode=area.mode,
            x_min_mm=area.x_min_mm,
            y_min_mm=area.y_min_mm,
            width_mm=area.width_mm,
            height_mm=area.height_mm,
            board_width_mm=area.board_width_mm,
            board_height_mm=area.board_height_mm,
            margin_mm=area.margin_mm,
            heat_source_count=area.heat_source_count,
            active_net_names=area.active_net_names,
            warnings=tuple(warnings),
        )
    return area


def estimate_solver_grid(
    bbox: Any,
    requested_res: float,
    settings: Dict[str, Any],
    layer_count: int,
    focus_pads: Optional[List[Any]] = None,
    area: Optional[AreaEstimate] = None,
    *,
    has_native_mg: Optional[bool] = None,
    has_native_compact: Optional[bool] = None,
) -> GridEstimate:
    """Compute the final solver grid after area limiting and coarsening."""
    original_x_min, original_y_min, original_x_max, original_y_max = bbox_bounds_mm(bbox)
    x_min, y_min = original_x_min, original_y_min
    x_max, y_max = original_x_max, original_y_max

    if area is not None:
        x_min = float(area.x_min_mm)
        y_min = float(area.y_min_mm)
        x_max = x_min + float(area.width_mm)
        y_max = y_min + float(area.height_mm)
    elif settings.get("limit_area") and settings.get("pad_dist_mm", 0.0) > 0:
        radius = float(settings["pad_dist_mm"])
        pad_xs = []
        pad_ys = []
        for pad in focus_pads or []:
            try:
                pos = pad.GetPosition()
                pad_xs.append(pos.x * 1e-6)
                pad_ys.append(pos.y * 1e-6)
            except Exception:
                continue
        if pad_xs and pad_ys:
            x_min = max(original_x_min, min(pad_xs) - radius)
            y_min = max(original_y_min, min(pad_ys) - radius)
            x_max = min(original_x_max, max(pad_xs) + radius)
            y_max = min(original_y_max, max(pad_ys) + radius)

    req_res = max(1e-4, float(requested_res))
    width = max(req_res, x_max - x_min)
    height = max(req_res, y_max - y_min)
    res, coarsened, expert, max_cells, target_cells = coarsen_grid_resolution(
        width, height, req_res, settings, layer_count
    )
    rows = int(height / res) + 4
    cols = int(width / res) + 4
    nodes = rows * cols * int(layer_count)
    detail, _, _, _, node_budget = resolve_grid_policy(settings, layer_count)
    memory_low, memory_high, runtime = estimate_solver_cost(
        nodes,
        settings,
        has_native_mg=has_native_mg,
        has_native_compact=has_native_compact,
    )
    return GridEstimate(
        requested_res_mm=float(req_res),
        actual_res_mm=float(res),
        x_min_mm=float(x_min),
        y_min_mm=float(y_min),
        width_mm=float(width),
        height_mm=float(height),
        rows=rows,
        cols=cols,
        layer_count=int(layer_count),
        auto_coarsened=bool(coarsened),
        expert_limits=bool(expert),
        max_cells=int(max_cells),
        target_cells=int(target_cells),
        detail_level=detail,
        node_budget=int(node_budget),
        memory_mb_low=memory_low,
        memory_mb_high=memory_high,
        runtime_class=runtime,
    )


def derive_stackup_thicknesses(
    board: Any,
    copper_ids: List[int],
    stack_info: Dict[str, Any],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive copper thicknesses and dielectric gaps from stackup or defaults."""
    stack_copper = stack_info.get("copper", []) if isinstance(stack_info, dict) else []
    stack_gaps = stack_info.get("dielectric_gaps_mm", []) if isinstance(stack_info, dict) else []
    stack_board_thick = stack_info.get("board_thickness_mm") if isinstance(stack_info, dict) else None

    copper_thickness_by_id: Dict[int, float] = {}
    copper_thickness_by_name: Dict[str, float] = {}
    for c in stack_copper:
        th = c.get("thickness_mm")
        if isinstance(th, (int, float)) and th > 0:
            lid = c.get("layer_id")
            name = c.get("name")
            if isinstance(lid, int):
                copper_thickness_by_id[lid] = float(th)
            if isinstance(name, str):
                copper_thickness_by_name[name] = float(th)

    cu_scale = float(settings.get("copper_thickness_scale", 1.0) or 1.0)
    copper_thickness_mm_used: List[float] = []
    for lid in copper_ids:
        th = copper_thickness_by_id.get(lid)
        if th is None:
            try:
                lname = str(board.GetLayerName(lid))
            except Exception:
                lname = ""
            th = copper_thickness_by_name.get(lname)
        if not isinstance(th, (int, float)) or th <= 0:
            th = 0.035
        copper_thickness_mm_used.append(float(th) * cu_scale)

    total_thick_mm_used = float(settings.get("thick", 1.6) or 1.6)
    if isinstance(stack_board_thick, (int, float)) and stack_board_thick > 0:
        total_thick_mm_used = float(stack_board_thick)

    fallback_gap_mm = total_thick_mm_used / max(1, len(copper_ids) - 1)
    use_uniform_gap = len(stack_gaps) != max(0, len(copper_ids) - 1)
    if not use_uniform_gap:
        for g in stack_gaps:
            if not isinstance(g, (int, float)) or g <= 0:
                use_uniform_gap = True
                break
    if use_uniform_gap:
        gap_mm_used = [fallback_gap_mm] * max(0, len(copper_ids) - 1)
    else:
        gap_mm_used = [float(g) for g in stack_gaps]

    return {
        "total_thick_mm_used": float(total_thick_mm_used),
        "stack_board_thick_mm": (
            float(stack_board_thick)
            if isinstance(stack_board_thick, (int, float))
            else None
        ),
        "copper_thickness_mm_used": copper_thickness_mm_used,
        "gap_mm_used": gap_mm_used,
        "gap_fallback_used": bool(use_uniform_gap),
    }


def effective_fr4_control_volume_thicknesses(
    gap_m: Sequence[float],
    total_thickness_m: float,
    layer_count: int,
) -> np.ndarray:
    """Return the FR4 thickness assigned to each copper-plane control volume."""
    count = max(1, int(layer_count))
    total = max(float(total_thickness_m), 1e-5)
    if count == 1:
        return np.asarray([total], dtype=np.float64)

    gaps = np.asarray(
        list(gap_m) if gap_m is not None else [],
        dtype=np.float64,
    )
    if (
        gaps.size != count - 1
        or not np.all(np.isfinite(gaps))
        or np.any(gaps <= 0.0)
    ):
        gaps = np.full(count - 1, total / float(count - 1), dtype=np.float64)

    thicknesses = np.empty(count, dtype=np.float64)
    thicknesses[0] = 0.5 * gaps[0]
    thicknesses[-1] = 0.5 * gaps[-1]
    if count > 2:
        thicknesses[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
    return np.clip(thicknesses, 1e-6, 5e-3)


def build_dense_capacity(
    layer_count: int,
    rows: int,
    cols: int,
    copper_mask: np.ndarray,
    t_cu: Sequence[float],
    t_fr4_eff: Sequence[float],
    pixel_area: float,
    rho_cu: float,
    cp_cu: float,
    rho_fr4: float,
    cp_fr4: float,
    pad_cap_areal: float,
    heatsink_mask: np.ndarray,
) -> np.ndarray:
    """Materialize the 1D float64 heat-capacity vector for legacy/dense paths."""
    C_layers = np.empty((layer_count, rows, cols), dtype=np.float64)
    for layer_idx in range(layer_count):
        V_cu = pixel_area * t_cu[layer_idx]
        V_fr4 = pixel_area * t_fr4_eff[layer_idx]
        mask = copper_mask[layer_idx]
        C_layer = np.where(mask, rho_cu * cp_cu * V_cu, rho_fr4 * cp_fr4 * V_fr4)
        C_layer += mask * (rho_fr4 * cp_fr4 * V_fr4)
        C_layers[layer_idx] = C_layer
    if pad_cap_areal > 0.0 and np.any(heatsink_mask):
        pad_cap_per_cell = pad_cap_areal * pixel_area
        C_layers[-1] += pad_cap_per_cell * np.asarray(heatsink_mask, dtype=np.float64)
    return C_layers.reshape(-1)


def build_compact_heatsink_model(
    dx: float,
    dy: float,
    heatsink_mask: np.ndarray,
    settings: Dict[str, Any],
    amb: float,
    *,
    heatsink_model_cls: Optional[Any] = None,
    capacity_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
) -> Optional[HeatsinkModel]:
    """Build a 2D bottom-layer HeatsinkModel without allocating a full 3D vector."""
    if not bool(settings.get("use_heatsink", True)):
        return None
    sink_mode = str(settings.get("pad_sink_mode", "fixed") or "fixed").strip().lower()
    if sink_mode != "heatsink":
        return None
    hs_bool = np.asarray(heatsink_mask, dtype=bool)
    if not np.any(hs_bool):
        return None
    pad_thick_m = max(0.0001, float(settings.get("pad_th", 1.0)) * 1e-3)
    pad_k = float(settings.get("pad_k", 3.0))
    contact_factor = max(1e-6, float(settings.get("pad_contact_factor", 0.2)))
    h_sink = (pad_k / pad_thick_m) * contact_factor
    pad_conductance = hs_bool.astype(np.float64).reshape(-1) * (h_sink * float(dx) * float(dy))
    rth_val = settings.get("heatsink_rth_kw", 5.0)
    rth_kw = max(1e-6, float(5.0 if rth_val is None else rth_val))
    model_cls = HeatsinkModel if heatsink_model_cls is None else heatsink_model_cls
    cap_fn = compute_heatsink_capacity_j_k if capacity_fn is None else capacity_fn
    return model_cls(
        pad_conductance=pad_conductance,
        rth_kw=rth_kw,
        capacity_j_k=cap_fn(settings),
        initial_temp_c=float(amb),
        bottom_layer_only=True,
    )


def pad_target_layer_index(
    board: Any,
    copper_ids: List[int],
    pad: Any,
    lid_to_idx: Dict[int, int],
) -> int:
    """Resolve the solver copper-layer index for a pad."""
    copper_indices = get_pad_copper_layer_indices(pad, copper_ids, lid_to_idx=lid_to_idx)
    if len(copper_indices) == 1:
        return copper_indices[0]
    if len(copper_indices) > 1:
        parent = None
        if hasattr(pad, "GetParent"):
            try:
                parent = pad.GetParent()
            except Exception:
                parent = None
        if parent is not None and hasattr(parent, "GetLayer"):
            try:
                parent_idx = lid_to_idx.get(int(parent.GetLayer()))
                if parent_idx in copper_indices:
                    return parent_idx
            except Exception:
                pass
        return copper_indices[0]

    pad_lid = pad.GetLayer()
    target_idx = lid_to_idx.get(pad_lid)
    if target_idx is not None:
        return target_idx

    try:
        lname = str(board.GetLayerName(pad_lid)).upper()
    except Exception:
        lname = ""
    return len(copper_ids) - 1 if ("B." in lname or "BOT" in lname) else 0


def build_sparse_pad_contributions(
    board: Any,
    copper_ids: List[int],
    pads_list: List[Any],
    rows: int,
    cols: int,
    x_min: float,
    y_min: float,
    res: float,
    area_mask: Optional[np.ndarray] = None,
) -> List[SparsePadContribution]:
    """Build sparse per-pad unit power distributions normalized to 1 W per pad."""
    lid_to_idx = {lid: idx for idx, lid in enumerate(copper_ids)}
    contributions: List[SparsePadContribution] = []

    for pad in pads_list:
        target_idx = pad_target_layer_index(board, copper_ids, pad, lid_to_idx)
        target_lid = copper_ids[target_idx] if 0 <= target_idx < len(copper_ids) else None
        indices = get_pad_power_indices(
            pad,
            target_idx=target_idx,
            rows=rows,
            cols=cols,
            x_min=x_min,
            y_min=y_min,
            res=res,
            layer_id=target_lid,
            area_mask=area_mask,
        )
        if indices.size:
            weights = np.full(indices.shape, 1.0 / float(indices.size), dtype=np.float64)
        else:
            weights = np.empty(0, dtype=np.float64)
        contributions.append(SparsePadContribution(indices=indices, weights=weights))

    return contributions


def build_power_vector(
    pad_sources: List[Tuple[str, Any]],
    pad_contributions: List[SparsePadContribution],
    total_nodes: int,
) -> Tuple[np.ndarray, Optional[Callable[[float], np.ndarray]]]:
    """Build the initial dense power vector and optional PWL callback."""
    q_const = np.zeros(total_nodes, dtype=np.float64)
    pwl_terms = []

    for idx, (source_type, source_value) in enumerate(pad_sources):
        if idx >= len(pad_contributions):
            break
        contribution = pad_contributions[idx]
        if contribution.indices.size == 0:
            continue
        if source_type == "const":
            q_const[contribution.indices] += float(source_value) * contribution.weights
        else:
            times, powers = source_value
            pwl_terms.append((
                np.asarray(times, dtype=np.float64),
                np.asarray(powers, dtype=np.float64),
                contribution.indices,
                contribution.weights,
            ))

    if not pwl_terms:
        return q_const, None

    q_initial = q_const.copy()
    for times, powers, indices, weights in pwl_terms:
        q_initial[indices] += float(np.interp(0.0, times, powers)) * weights

    q_workspace = np.empty_like(q_const)

    def q_func(t, _q_const=q_const, _pwl_terms=tuple(pwl_terms), _workspace=q_workspace):
        np.copyto(_workspace, _q_const)
        for times, powers, indices, weights in _pwl_terms:
            _workspace[indices] += float(np.interp(t, times, powers)) * weights
        return _workspace

    return q_initial, q_func


def build_sparse_power_vector(
    pad_sources: List[Tuple[str, Any]],
    pad_contributions: List[SparsePadContribution],
    total_nodes: int,
    *,
    sparse_power_cls: Optional[Any] = None,
) -> Any:
    """Build a SparsePowerVector for constant pad sources without dense allocation."""
    idx_parts = []
    val_parts = []
    for idx, (source_type, source_value) in enumerate(pad_sources):
        if idx >= len(pad_contributions):
            break
        contribution = pad_contributions[idx]
        if contribution.indices.size == 0 or source_type != "const":
            continue
        val = float(source_value)
        if val == 0.0:
            continue
        idx_parts.append(contribution.indices)
        val_parts.append(val * contribution.weights)
    if idx_parts:
        indices = np.concatenate(idx_parts)
        values = np.concatenate(val_parts)
    else:
        indices = np.empty(0, dtype=np.int64)
        values = np.empty(0, dtype=np.float64)
    spv_cls = SparsePowerVector if sparse_power_cls is None else sparse_power_cls
    return spv_cls(indices=indices, values=values, size=total_nodes)
