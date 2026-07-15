"""
ThermalSim - KiCad PCB thermal simulation plugin.

This is the main controller module that orchestrates the thermal simulation
workflow using the specialized sub-modules.
"""

import os
import re
import time
import math
import json
import tempfile
import threading
import traceback
from dataclasses import dataclass, asdict

import pcbnew
import numpy as np
import wx

from .capabilities import HAS_LIBS, HAS_PARDISO, get_pypardiso_optional_dependency
from .stackup_parser import parse_stackup_from_board_file, format_stackup_report_um
from .gui_dialogs import (
    DEFAULT_GRID_MAX_CELLS,
    DEFAULT_GRID_TARGET_CELLS,
    SettingsDialog,
    prepare_current_groups,
    prepare_power_pads,
)
from .electrical_solver import (
    CurrentTerminal,
    ElectricalConfig,
    net_key_from_obj,
    net_key_from_values,
    solve_electrical_heating,
)
from .geometry_mapper import build_geometry_state, create_multilayer_maps, get_pad_pixels
from .thermal_solver import SolverConfig, build_stiffness_matrix, run_simulation
from .pwl_parser import parse_pwl_file
from .visualization import (
    save_snapshot, show_results_top_bot, show_results_all_layers, save_preview_image,
    build_interactive_heatmap_payload, save_joule_loss_map
)
from .thermal_report import write_html_report
from .workflow import (
    AreaEstimate,
    BoardSnapshot,
    CancellationToken,
    GeometryCache,
    ThermalFactorizationCache,
    ThermalOperatorCache,
    GridEstimate,
    PreflightResult,
    SimulationArtifacts,
    geometry_cache_key,
    stable_fingerprint,
)


GRID_DETAIL_PRESETS = {
    "fast": (300_000, 150_000),
    "balanced": (800_000, 400_000),
    "detailed": (1_600_000, 800_000),
    "very_detailed": (3_000_000, 1_500_000),
}
DEFAULT_GRID_DETAIL = "balanced"
DEFAULT_GRID_NODE_BUDGET = GRID_DETAIL_PRESETS[DEFAULT_GRID_DETAIL][0]


@dataclass
class SparsePadContribution:
    """Sparse per-pad power distribution on the simulation grid."""

    indices: np.ndarray
    weights: np.ndarray


def _electrical_net_summary_dict(summary):
    """Convert electrical net diagnostics to report-friendly dictionaries."""
    data = asdict(summary)
    data["net"] = data.get("net_name", "")
    return data


def _bbox_to_power_indices(bbox, target_idx, rows, cols, x_min, y_min, res, rc):
    """
    Convert a pad bounding box to flattened node indices on one copper layer.

    Parameters
    ----------
    bbox : pcbnew.EDA_RECT
        Pad bounding box in KiCad internal units.
    target_idx : int
        Target copper layer index.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    x_min : float
        Grid origin x coordinate in millimeters.
    y_min : float
        Grid origin y coordinate in millimeters.
    res : float
        Grid resolution in millimeters.
    rc : int
        Number of nodes per layer.

    Returns
    -------
    np.ndarray
        Flattened node indices for the rectangular pad extent.
    """
    x0 = bbox.GetX() * 1e-6
    y0 = bbox.GetY() * 1e-6
    w = bbox.GetWidth() * 1e-6
    h = bbox.GetHeight() * 1e-6
    cs = max(0, int((x0 - x_min) / res))
    rs = max(0, int((y0 - y_min) / res))
    ce = min(cols, int((x0 + w - x_min) / res) + 1)
    re = min(rows, int((y0 + h - y_min) / res) + 1)
    if cs >= ce or rs >= re:
        return np.empty(0, dtype=np.int64)

    row_offsets = np.arange(rs, re, dtype=np.int64) * cols
    col_offsets = np.arange(cs, ce, dtype=np.int64)
    return target_idx * rc + (row_offsets[:, None] + col_offsets[None, :]).ravel(order="C")


def _pad_target_layer_index(board, copper_ids, pad, lid_to_idx):
    """
    Resolve the solver layer index for a pad.

    Parameters
    ----------
    board : pcbnew.BOARD
        The active board object.
    copper_ids : list of int
        Copper layer IDs in stackup order.
    pad : pcbnew.PAD
        Pad to place on the thermal grid.
    lid_to_idx : dict
        Mapping from KiCad layer IDs to solver indices.

    Returns
    -------
    int
        Target copper layer index for the pad.
    """
    pad_lid = pad.GetLayer()
    target_idx = lid_to_idx.get(pad_lid)
    if target_idx is not None:
        return target_idx

    try:
        lname = board.GetLayerName(pad_lid).upper()
    except Exception:
        lname = ""
    return len(copper_ids) - 1 if ("B." in lname or "BOT" in lname) else 0


def _build_sparse_pad_contributions(board, copper_ids, pads_list, rows, cols, x_min, y_min, res):
    """
    Build sparse per-pad unit power distributions.

    Parameters
    ----------
    board : pcbnew.BOARD
        The active board object.
    copper_ids : list of int
        Copper layer IDs in stackup order.
    pads_list : list
        Selected pads used as heat sources.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    x_min : float
        Grid origin x coordinate in millimeters.
    y_min : float
        Grid origin y coordinate in millimeters.
    res : float
        Grid resolution in millimeters.

    Returns
    -------
    list of SparsePadContribution
        Sparse power distributions normalized to 1 W per pad.
    """
    rc = rows * cols
    lid_to_idx = {lid: idx for idx, lid in enumerate(copper_ids)}
    contributions = []

    for pad in pads_list:
        target_idx = _pad_target_layer_index(board, copper_ids, pad, lid_to_idx)
        indices = _bbox_to_power_indices(
            pad.GetBoundingBox(),
            target_idx=target_idx,
            rows=rows,
            cols=cols,
            x_min=x_min,
            y_min=y_min,
            res=res,
            rc=rc,
        )
        if indices.size:
            weights = np.full(indices.shape, 1.0 / float(indices.size), dtype=np.float64)
        else:
            weights = np.empty(0, dtype=np.float64)
        contributions.append(SparsePadContribution(indices=indices, weights=weights))

    return contributions


def _build_power_vector(pad_sources, pad_contributions, total_nodes):
    """
    Build the constant power vector and optional time-varying callback.

    Parameters
    ----------
    pad_sources : list
        Parsed pad source descriptors: ('const', value) or ('pwl', (times, powers)).
    pad_contributions : list of SparsePadContribution
        Sparse per-pad unit distributions.
    total_nodes : int
        Total number of thermal nodes.

    Returns
    -------
    tuple
        (Q, Q_func) where Q is the initial dense power vector and Q_func is an
        optional callback for time-varying inputs.
    """
    q_const = np.zeros(total_nodes, dtype=np.float64)
    pwl_terms = []

    for idx, (source_type, source_value) in enumerate(pad_sources):
        if idx >= len(pad_contributions):
            break
        contribution = pad_contributions[idx]
        if contribution.indices.size == 0:
            continue
        if source_type == 'const':
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


def _format_timing_summary(timings):
    """Format initialization timings for console output."""
    parts = []
    for key in (
        "zone_refill_s",
        "geometry_maps_s",
        "capacity_build_s",
        "power_vector_build_s",
        "electrical_solve_s",
        "stiffness_matrix_s",
    ):
        if key in timings:
            parts.append(f"{key}={float(timings[key]):.4f}s")
    return ", ".join(parts)


def _write_run_manifest(run_dir, status, **details):
    """Persist the lifecycle state of a result directory."""
    payload = {"schema_version": 1, "status": str(status), "updated_at": time.time()}
    payload.update(details)
    try:
        with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    except Exception:
        pass


def _resolve_grid_limits(settings):
    """
    Resolve automatic grid coarsening limits from settings.

    Returns
    -------
    tuple
        ``(expert_enabled, max_cells, target_cells)`` with safe defaults.
    """
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


def _resolve_grid_policy(settings, layer_count):
    """Resolve a layer-aware detail preset to legacy 2D cell limits."""
    layers = max(1, int(layer_count or 1))
    if "grid_detail_level" not in settings:
        expert, max_cells, target_cells = _resolve_grid_limits(settings)
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
        max_nodes = min(10_000_000, max(50_000, max_nodes))
        target_nodes = max(25_000, max_nodes // 2)
        expert = True
    else:
        max_nodes, target_nodes = GRID_DETAIL_PRESETS[detail]
        expert = False

    max_cells = max(1000, max_nodes // layers)
    target_cells = max(1000, min(max_cells, target_nodes // layers))
    return detail, expert, max_cells, target_cells, max_nodes


def _coarsen_grid_resolution(w_mm, h_mm, requested_res, settings, layer_count=1):
    """
    Apply automatic grid coarsening for large boards.

    Parameters
    ----------
    w_mm, h_mm : float
        Board width and height in millimeters.
    requested_res : float
        User-requested grid resolution in millimeters.
    settings : dict
        Simulation settings.

    Returns
    -------
    tuple
        ``(res, auto_coarsened, expert_enabled, max_cells, target_cells)``.
    """
    _, expert_enabled, max_cells, target_cells, _ = _resolve_grid_policy(
        settings, layer_count
    )
    area = w_mm * h_mm
    res = float(requested_res)
    auto_coarsened = False
    if res > 0.0 and area > 0.0 and (w_mm / res) * (h_mm / res) > max_cells:
        res = math.sqrt(area / float(target_cells))
        auto_coarsened = True
    return res, auto_coarsened, expert_enabled, max_cells, target_cells


def _bbox_bounds_mm(bbox):
    """Return a KiCad rectangle as absolute millimetre bounds."""
    x_min = float(bbox.GetX()) * 1e-6
    y_min = float(bbox.GetY()) * 1e-6
    return (
        x_min,
        y_min,
        x_min + float(bbox.GetWidth()) * 1e-6,
        y_min + float(bbox.GetHeight()) * 1e-6,
    )


def _estimate_simulation_area(board, bbox, settings, power_pads=None, terminals=None):
    """Build a safe rectangular domain around heat sources and current nets."""
    board_x0, board_y0, board_x1, board_y1 = _bbox_bounds_mm(bbox)
    board_w = max(0.0, board_x1 - board_x0)
    board_h = max(0.0, board_y1 - board_y0)
    legacy_limited = bool(settings.get("limit_area", False))
    mode = str(settings.get("area_mode") or ("active" if legacy_limited else "full"))
    mode = mode.strip().lower()
    if mode not in ("full", "active"):
        mode = "active" if legacy_limited else "full"
    margin = max(0.0, float(settings.get("area_margin_mm", settings.get("pad_dist_mm", 0.0)) or 0.0))

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
            mode="full", x_min_mm=board_x0, y_min_mm=board_y0,
            width_mm=board_w, height_mm=board_h,
            board_width_mm=board_w, board_height_mm=board_h,
            margin_mm=margin, heat_source_count=len(power_pads or []),
            active_net_names=active_names,
        )

    bounds = []
    warnings = []
    collection_failed = False

    def add_bbox(item):
        try:
            item_bbox = item.GetBoundingBox() if hasattr(item, "GetBoundingBox") else item
            x0, y0, x1, y1 = _bbox_bounds_mm(item_bbox)
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
            mode="full", x_min_mm=board_x0, y_min_mm=board_y0,
            width_mm=board_w, height_mm=board_h,
            board_width_mm=board_w, board_height_mm=board_h,
            margin_mm=margin, heat_source_count=len(power_pads or []),
            active_net_names=active_names, fallback_to_full=True,
            warnings=tuple(warnings),
        )

    if not bounds:
        warnings.append("No active source geometry was found; the full board is used.")
        return AreaEstimate(
            mode="full", x_min_mm=board_x0, y_min_mm=board_y0,
            width_mm=board_w, height_mm=board_h,
            board_width_mm=board_w, board_height_mm=board_h,
            margin_mm=margin, heat_source_count=0,
            active_net_names=active_names, fallback_to_full=True,
            warnings=tuple(warnings),
        )

    safety = max(0.0, float(settings.get("res", 0.0) or 0.0))
    x0 = max(board_x0, min(item[0] for item in bounds) - margin - safety)
    y0 = max(board_y0, min(item[1] for item in bounds) - margin - safety)
    x1 = min(board_x1, max(item[2] for item in bounds) + margin + safety)
    y1 = min(board_y1, max(item[3] for item in bounds) + margin + safety)
    area = AreaEstimate(
        mode="active", x_min_mm=x0, y_min_mm=y0,
        width_mm=max(0.0, x1 - x0), height_mm=max(0.0, y1 - y0),
        board_width_mm=board_w, board_height_mm=board_h,
        margin_mm=margin, heat_source_count=len(power_pads or []),
        active_net_names=active_names, warnings=tuple(warnings),
    )
    if active_keys and area.area_fraction >= 0.95:
        warnings.append("Active current geometry covers almost the full board; area limiting saves little work.")
        area = AreaEstimate(
            mode=area.mode, x_min_mm=area.x_min_mm, y_min_mm=area.y_min_mm,
            width_mm=area.width_mm, height_mm=area.height_mm,
            board_width_mm=area.board_width_mm, board_height_mm=area.board_height_mm,
            margin_mm=area.margin_mm, heat_source_count=area.heat_source_count,
            active_net_names=area.active_net_names, warnings=tuple(warnings),
        )
    return area


def _estimate_solver_cost(nodes, settings):
    """Return conservative memory bounds and a relative runtime class."""
    nodes = max(0, int(nodes))
    memory_low = int(math.ceil(nodes * 0.00035))
    memory_high = int(math.ceil(nodes * 0.00120))
    backend = str(settings.get("solver_backend", "auto") or "auto").lower()
    thresholds = (150_000, 500_000, 1_000_000)
    if backend == "pardiso" or (backend == "auto" and HAS_PARDISO):
        thresholds = (250_000, 800_000, 1_600_000)
    if nodes < thresholds[0]:
        runtime = "Fast"
    elif nodes < thresholds[1]:
        runtime = "Moderate"
    elif nodes < thresholds[2]:
        runtime = "Slow"
    else:
        runtime = "Very slow"
    return memory_low, memory_high, runtime


def _estimate_solver_grid(
    bbox, requested_res, settings, layer_count, focus_pads=None, area=None
):
    """Compute the final solver grid after area limiting and coarsening."""
    original_x_min, original_y_min, original_x_max, original_y_max = _bbox_bounds_mm(bbox)
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

    width = max(float(requested_res), x_max - x_min)
    height = max(float(requested_res), y_max - y_min)
    res, coarsened, expert, max_cells, target_cells = _coarsen_grid_resolution(
        width, height, requested_res, settings, layer_count
    )
    rows = int(height / res) + 4
    cols = int(width / res) + 4
    nodes = rows * cols * int(layer_count)
    detail, _, _, _, node_budget = _resolve_grid_policy(settings, layer_count)
    memory_low, memory_high, runtime = _estimate_solver_cost(nodes, settings)
    return GridEstimate(
        requested_res_mm=float(requested_res),
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


class ThermalPlugin(pcbnew.ActionPlugin):
    """
    KiCad Action Plugin for 2.5D transient thermal simulation.

    This plugin simulates heat spreading across multilayer PCBs using
    finite volume methods with BDF2 time integration.
    """

    def defaults(self):
        """Set plugin metadata and initialize state."""
        self.name = "2.5D Thermal Sim"
        self.category = "Simulation"
        self.description = "Crash-safe Multilayer Sim"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "ThermalSim_icon.png")

        # Store references for preview
        self.board = None
        self.copper_ids = []
        self.bbox = None
        self.pads_list = []
        self.stack_info = None
        self.last_zone_refill_s = 0.0
        self.settings_dialog = None
        self.geometry_cache = GeometryCache()
        self.operator_cache = ThermalOperatorCache()
        self.electrical_cache = ThermalOperatorCache()
        self.factorization_cache = ThermalFactorizationCache()
        self.cancel_token = None
        self.last_artifacts = None

    def _capture_board_snapshot(self, board, copper_ids, bbox):
        """Capture a cheap deterministic identity for geometry cache safety."""
        tracks = list(board.Tracks() if hasattr(board, "Tracks") else board.GetTracks())
        footprints = list(board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints())
        zones = list(board.Zones() if hasattr(board, "Zones") else board.GetZones())
        primitive_rows = []
        for item in tracks + zones:
            try:
                rect = item.GetBoundingBox()
                primitive_rows.append((
                    type(item).__name__, int(item.GetLayer()), rect.GetX(), rect.GetY(),
                    rect.GetWidth(), rect.GetHeight(),
                ))
            except Exception:
                continue
        for fp in footprints:
            try:
                reference = fp.GetReference()
            except Exception:
                reference = ""
            for pad in fp.Pads():
                try:
                    pos = pad.GetPosition()
                    primitive_rows.append(("pad", reference, pad.GetNumber(), pos.x, pos.y, pad.GetLayer()))
                except Exception:
                    continue
        bbox_mm = (
            bbox.GetX() * 1e-6,
            bbox.GetY() * 1e-6,
            bbox.GetWidth() * 1e-6,
            bbox.GetHeight() * 1e-6,
        )
        return BoardSnapshot(
            filename=str(board.GetFileName() or ""),
            fingerprint=stable_fingerprint((bbox_mm, tuple(primitive_rows))),
            bbox_mm=bbox_mm,
            copper_layers=tuple(int(x) for x in copper_ids),
            track_count=len(tracks),
            footprint_count=len(footprints),
            zone_count=len(zones),
        )

    def _geometry_key(self, board, copper_ids, bbox, grid, settings, pads):
        snapshot = self._capture_board_snapshot(board, copper_ids, bbox)
        pad_keys = []
        for pad in pads or []:
            try:
                pos = pad.GetPosition()
                pad_keys.append((pad.GetNumber(), pad.GetLayer(), pos.x, pos.y))
            except Exception:
                pad_keys.append(id(pad))
        return geometry_cache_key(snapshot, grid, settings, pad_keys)

    def _settings_path(self):
        """Return path to settings persistence file."""
        try:
            base_dir = wx.StandardPaths.Get().GetUserConfigDir()
        except Exception:
            base_dir = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
        return os.path.join(base_dir, "ThermalSim", "thermal_sim_last_settings.json")

    def _load_settings(self, path=None):
        """Load settings from a JSON file."""
        settings_path = path or self._settings_path()
        if path is None and not os.path.isfile(settings_path):
            legacy_path = os.path.join(os.path.dirname(__file__), "thermal_sim_last_settings.json")
            if os.path.isfile(legacy_path):
                try:
                    with open(legacy_path, "r", encoding="utf-8") as f:
                        legacy = json.load(f)
                    if isinstance(legacy, dict):
                        legacy["schema_version"] = 2
                        self._save_settings(legacy)
                except Exception:
                    pass
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_settings(self, settings, path=None):
        """Save settings to a JSON file."""
        settings_path = path or self._settings_path()
        try:
            parent_dir = os.path.dirname(settings_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            payload = dict(settings)
            payload["schema_version"] = 2
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            return True
        except Exception:
            return False

    def Run(self):
        """Plugin entry point with error handling."""
        try:
            self.RunSafe()
        except Exception:
            wx.MessageBox(traceback.format_exc(), "Thermal Sim Error")

    def RunSafe(self):
        """Main plugin execution logic."""
        if not HAS_LIBS:
            from .capabilities import get_missing_packages
            from .dependency_installer import DependencyInstallDialog
            missing = get_missing_packages()
            if missing:
                dlg = DependencyInstallDialog(None, missing, [get_pypardiso_optional_dependency()])
                dlg.ShowModal()
                dlg.Destroy()
            else:
                wx.MessageBox(
                    "All packages appear installed. Please restart KiCad.",
                    "ThermalSim"
                )
            return

        board = pcbnew.GetBoard()

        # Keep zone fills up-to-date
        zone_refill_start = time.perf_counter()
        try:
            pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        except Exception:
            pass
        zone_refill_s = time.perf_counter() - zone_refill_start

        # --- 1. Layer Detection ---
        copper_ids, layer_names = self._detect_copper_layers(board)
        stack_info = parse_stackup_from_board_file(board)

        # Use stackup order if available
        copper_ids_stack = stack_info.get("copper_ids") if isinstance(stack_info, dict) else None
        if copper_ids_stack and len(copper_ids_stack) >= 2:
            copper_ids = copper_ids_stack
            layer_names = [board.GetLayerName(lid) for lid in copper_ids]

        # --- 2. Auto-Resolution ---
        try:
            bbox = board.GetBoundingBox()
        except:
            bbox = board.ComputeBoundingBox(True)
        w_mm = bbox.GetWidth() * 1e-6
        h_mm = bbox.GetHeight() * 1e-6
        suggested_res = self._calculate_suggested_resolution(w_mm, h_mm, len(copper_ids))

        # --- 3. Initial Pad Selection ---
        selected_pads = self._get_selected_pads(board)
        pads_list = [p[1] for p in selected_pads]

        # Store for preview
        self.board = board
        self.copper_ids = copper_ids
        self.bbox = bbox
        self.pads_list = pads_list
        self.stack_info = stack_info
        self.last_zone_refill_s = zone_refill_s

        # --- 4. Show Dialog ---
        stackup_details = format_stackup_report_um(stack_info) if stack_info else ""
        pad_names = self._format_pad_names(selected_pads)
        initial_power_pads = self._get_selected_pad_descriptors(board)
        board_path = str(board.GetFileName() or "")
        board_dir = os.path.dirname(board_path) if board_path else ""
        default_output_dir = (
            os.path.join(board_dir, "ThermalSim_results")
            if board_dir else os.path.join(os.path.expanduser("~"), "Documents", "ThermalSim_results")
        )
        last_settings = self._load_settings()
        if last_settings.get("output_dir") and os.path.isdir(last_settings.get("output_dir")):
            default_output_dir = last_settings.get("output_dir")

        if self.settings_dialog is not None:
            try:
                self.settings_dialog.Raise()
                return
            except Exception:
                self.settings_dialog = None

        def selection_provider():
            return self._get_selected_pad_descriptors(board)

        def run_callback(settings):
            self._save_settings(settings)
            power_pads = self._resolve_power_pad_objects(board, settings, legacy_pads=pads_list)
            current_pads = self._resolve_current_pad_objects(board, settings)
            focus_pads = self._unique_pads(power_pads + current_pads)
            try:
                self._run_simulation(
                    board, copper_ids, layer_names, bbox, pads_list,
                    settings, stack_info, pad_names, zone_refill_s=zone_refill_s,
                    focus_pads=focus_pads
                )
            except Exception:
                error_text = traceback.format_exc()
                if self.settings_dialog is not None:
                    try:
                        self.settings_dialog.set_run_state(
                            "failed",
                            "The simulation failed. See the error dialog for details.",
                        )
                    except Exception:
                        pass
                wx.MessageBox(error_text, "Thermal Sim Error")

        def close_callback():
            self.settings_dialog = None

        def preflight_callback(settings):
            return self._preflight(board, copper_ids, bbox, settings)

        dlg = SettingsDialog(
            None, len(pads_list), suggested_res, layer_names,
            preview_callback=self.generate_preview,
            selection_provider=selection_provider,
            run_callback=run_callback,
            close_callback=close_callback,
            preflight_callback=preflight_callback,
            load_settings_callback=self._load_settings,
            save_settings_callback=self._save_settings,
            stackup_details=stackup_details,
            pad_names=pad_names,
            initial_power_pads=initial_power_pads,
            default_output_dir=default_output_dir,
            defaults=last_settings,
            board_name=os.path.basename(board_path) if board_path else "Unsaved board",
            board_size_mm=(w_mm, h_mm),
        )
        self.settings_dialog = dlg
        try:
            dlg.Show()
        except Exception:
            try:
                if dlg.ShowModal() == wx.ID_OK:
                    settings = dlg.get_values()
                    if settings:
                        self._save_settings(settings)
                        power_pads = self._resolve_power_pad_objects(board, settings, legacy_pads=pads_list)
                        current_pads = self._resolve_current_pad_objects(board, settings)
                        focus_pads = self._unique_pads(power_pads + current_pads)
                        self._run_simulation(
                            board, copper_ids, layer_names, bbox, pads_list,
                            settings, stack_info, pad_names, zone_refill_s=zone_refill_s,
                            focus_pads=focus_pads
                        )
            finally:
                self.settings_dialog = None
                dlg.Destroy()

    def _detect_copper_layers(self, board):
        """Detect enabled copper layers in stackup order."""
        copper_ids = []
        layer_names = []
        enabled_layers = board.GetEnabledLayers()

        for lid in range(64):
            try:
                is_copper = pcbnew.IsCopperLayer(lid)
            except:
                is_copper = (lid < 32)

            if enabled_layers.Contains(lid) and is_copper:
                copper_ids.append(lid)
                layer_names.append(board.GetLayerName(lid))

        # Sort by copper ordinal
        try:
            copper_ids = sorted(copper_ids, key=lambda lid: int(pcbnew.CopperLayerToOrdinal(lid)))
        except Exception:
            def _copper_key(lid):
                nm = board.GetLayerName(lid)
                if nm == "F.Cu":
                    return -1000
                if nm == "B.Cu":
                    return 1000
                m = re.match(r"In(\d+)\.Cu", nm)
                return int(m.group(1)) if m else 0
            copper_ids = sorted(copper_ids, key=_copper_key)

        layer_names = [board.GetLayerName(lid) for lid in copper_ids]
        return copper_ids, layer_names

    def _calculate_suggested_resolution(self, w_mm, h_mm, layer_count):
        """Calculate suggested grid resolution based on board size."""
        target_nodes = 25000 if layer_count <= 2 else 15000
        area = w_mm * h_mm
        if area > 0:
            suggested_res = round(math.sqrt(area / target_nodes), 2)
            if suggested_res < 0.2:
                suggested_res = 0.2
        else:
            suggested_res = 0.5
        return suggested_res

    def _get_selected_pads(self, board):
        """Get list of selected pads."""
        selected_pads = []
        try:
            footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
            for fp in footprints:
                for pad in fp.Pads():
                    if pad.IsSelected():
                        name = f"{fp.GetReference()}-{pad.GetNumber()}"
                        selected_pads.append((name, pad))
        except Exception as e:
            wx.MessageBox(f"Error reading pads: {e}", "Error")
            return []
        selected_pads.sort(key=lambda x: x[0])
        return selected_pads

    def _pad_key(self, fp_ref, pad):
        """Build a stable key for a pad in settings."""
        try:
            pos = pad.GetPosition()
            px, py = int(pos.x), int(pos.y)
        except Exception:
            px, py = 0, 0
        try:
            net_code = int(pad.GetNetCode())
        except Exception:
            net_code = 0
        try:
            number = pad.GetNumber()
        except Exception:
            number = ""
        return f"{fp_ref}:{number}:{net_code}:{px}:{py}"

    def _pad_net_name(self, pad):
        """Return the display net name for a pad."""
        try:
            net = pad.GetNetname()
        except Exception:
            try:
                net = pad.GetNet().GetNetname()
            except Exception:
                net = ""
        return net or ""

    def _pad_net_code(self, pad):
        """Return the KiCad net code for a pad."""
        try:
            return int(pad.GetNetCode())
        except Exception:
            try:
                return int(pad.GetNet().GetNetCode())
            except Exception:
                return 0

    def _pad_layer_name(self, board, pad):
        """Return the KiCad layer name for a pad."""
        try:
            return board.GetLayerName(pad.GetLayer())
        except Exception:
            return str(pad.GetLayer())

    def _pad_descriptor(self, board, fp_ref, pad):
        """Return a serializable pad descriptor for current groups."""
        try:
            number = pad.GetNumber()
        except Exception:
            number = ""
        name = f"{fp_ref}-{number}"
        net_name = self._pad_net_name(pad)
        return {
            "pad_key": self._pad_key(fp_ref, pad),
            "name": f"{name} [{net_name}]" if net_name else name,
            "net_name": net_name,
            "net_code": self._pad_net_code(pad),
            "layer": self._pad_layer_name(board, pad),
            "current_a": 0.0,
        }

    def _get_selected_pad_descriptors(self, board):
        """Return serializable descriptors for currently selected KiCad pads."""
        descriptors = []
        try:
            footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
            for fp in footprints:
                fp_ref = fp.GetReference()
                for pad in fp.Pads():
                    if pad.IsSelected():
                        descriptors.append(self._pad_descriptor(board, fp_ref, pad))
        except Exception:
            return []
        descriptors.sort(key=lambda item: item.get("name", ""))
        return descriptors

    def _build_pad_lookup(self, board):
        """Build lookup maps from persisted pad descriptors to live pad objects."""
        by_key = {}
        by_name = {}
        try:
            footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
        except Exception:
            footprints = []
        for fp in footprints:
            fp_ref = fp.GetReference()
            for pad in fp.Pads():
                descriptor = self._pad_descriptor(board, fp_ref, pad)
                by_key[descriptor["pad_key"]] = (descriptor, pad)
                by_name[descriptor["name"]] = (descriptor, pad)
        return by_key, by_name

    def _resolve_power_pad_entries(self, board, settings):
        """Resolve manual power-pad settings to live pad objects and power strings."""
        raw_power_pads = prepare_power_pads(
            settings.get("power_pads", []),
            settings.get("power_str", "1.0"),
        )
        if not raw_power_pads:
            return [], []
        by_key, by_name = self._build_pad_lookup(board)
        entries = []
        missing = []
        for pad_info in raw_power_pads:
            match = by_key.get(pad_info.get("pad_key")) or by_name.get(pad_info.get("name"))
            if not match:
                missing.append(pad_info.get("name", pad_info.get("pad_key", "<unknown>")))
                continue
            descriptor, pad = match
            merged = dict(descriptor)
            merged["power"] = str(pad_info.get("power", "0.0")).strip()
            entries.append((merged, pad, merged["power"]))
        return entries, missing

    def _resolve_power_pad_objects(self, board, settings, legacy_pads=None):
        """Resolve manual power pads to live pad objects for preview/focus area."""
        if "power_pads" not in (settings or {}):
            return list(legacy_pads or [])
        entries, _ = self._resolve_power_pad_entries(board, settings)
        return self._unique_pads([pad for _, pad, _ in entries])

    def _resolve_current_pad_objects(self, board, settings):
        """Resolve current-group pad descriptors to live pad objects."""
        if not settings.get("current_enabled"):
            return []
        by_key, by_name = self._build_pad_lookup(board)
        pads = []
        seen = set()
        for group in prepare_current_groups(settings.get("current_groups", [])):
            for pad_info in group.get("pads", []):
                match = by_key.get(pad_info.get("pad_key")) or by_name.get(pad_info.get("name"))
                if not match:
                    continue
                _, pad = match
                if id(pad) not in seen:
                    pads.append(pad)
                    seen.add(id(pad))
        return pads

    def _resolve_current_terminals(self, board, settings):
        """Resolve current-group settings into CurrentTerminal objects."""
        terminals = []
        missing = []
        if not settings.get("current_enabled"):
            return terminals, missing
        by_key, by_name = self._build_pad_lookup(board)
        for group in prepare_current_groups(settings.get("current_groups", [])):
            for pad_info in group.get("pads", []):
                current = float(pad_info.get("current_a", 0.0) or 0.0)
                if abs(current) <= 0.0:
                    continue
                match = by_key.get(pad_info.get("pad_key")) or by_name.get(pad_info.get("name"))
                if not match:
                    missing.append(pad_info.get("name", pad_info.get("pad_key", "<unknown>")))
                    continue
                descriptor, pad = match
                terminals.append(CurrentTerminal(
                    pad=pad,
                    name=descriptor.get("name", pad_info.get("name", "")),
                    net_name=descriptor.get("net_name", pad_info.get("net_name", "")),
                    net_code=int(descriptor.get("net_code", pad_info.get("net_code", 0)) or 0),
                    current_a=current,
                ))
        return terminals, missing

    def _unique_pads(self, pads):
        """Return pads without duplicate object identities."""
        result = []
        seen = set()
        for pad in pads or []:
            ident = id(pad)
            if ident in seen:
                continue
            result.append(pad)
            seen.add(ident)
        return result

    def _format_pad_names(self, selected_pads):
        """Format pad names with net info for display."""
        pad_names = []
        for nm, pad in selected_pads:
            net = ""
            try:
                net = pad.GetNetname()
            except Exception:
                try:
                    net = pad.GetNet().GetNetname()
                except Exception:
                    net = ""
            pad_names.append(f"{nm} [{net}]" if net else nm)
        return pad_names

    def _preflight(self, board, copper_ids, bbox, settings):
        """Validate settings and estimate the exact grid before execution."""
        result = PreflightResult()
        power_entries, missing_power = self._resolve_power_pad_entries(board, settings)
        power_pads = [pad for _, pad, _ in power_entries]
        terminals = []
        missing_current = []
        if settings.get("current_enabled"):
            terminals, missing_current = self._resolve_current_terminals(board, settings)
        current_pads = [item.pad for item in terminals]
        focus_pads = self._unique_pads(power_pads + current_pads)
        result.area = _estimate_simulation_area(
            board, bbox, settings, power_pads=power_pads, terminals=terminals
        )
        result.grid = _estimate_solver_grid(
            bbox, float(settings.get("res", 0.5)), settings, len(copper_ids),
            focus_pads, area=result.area,
        )
        result.warnings.extend(result.area.warnings)

        if missing_power:
            result.errors.append(f"{len(missing_power)} heat-source pad(s) are missing from the board.")
        has_power = False
        for _, _, entry in power_entries:
            value = str(entry or "").strip()
            if not value:
                continue
            try:
                has_power = has_power or abs(float(value)) > 0.0
            except ValueError:
                try:
                    _, pwl_values = parse_pwl_file(value)
                    has_power = has_power or bool(np.any(np.asarray(pwl_values) != 0.0))
                except Exception as exc:
                    result.errors.append(f"Invalid PWL source '{value}': {exc}")

        has_current = False
        if settings.get("current_enabled"):
            if missing_current:
                result.errors.append(f"{len(missing_current)} current-terminal pad(s) are missing from the board.")
            has_current = any(abs(float(item.current_a)) > 0.0 for item in terminals)
            totals = {}
            for item in terminals:
                net = item.net_name or f"net:{item.net_code}"
                totals[net] = totals.get(net, 0.0) + float(item.current_a)
            unbalanced = [name for name, total in totals.items() if abs(total) > 1e-9]
            if unbalanced:
                result.errors.append("Current is not balanced for: " + ", ".join(unbalanced[:3]))

        if not has_power and not has_current:
            result.errors.append("Configure at least one non-zero heat source or balanced current path.")

        output_dir = str(settings.get("output_dir", "") or "").strip()
        writable_parent = output_dir
        while writable_parent and not os.path.exists(writable_parent):
            parent = os.path.dirname(writable_parent)
            if parent == writable_parent:
                break
            writable_parent = parent
        if not output_dir or not writable_parent or not os.access(writable_parent, os.W_OK):
            result.errors.append("The output folder is not writable.")

        if result.grid.auto_coarsened:
            result.warnings.append(
                f"Resolution will be coarsened from {result.grid.requested_res_mm:.3f} "
                f"to {result.grid.actual_res_mm:.3f} mm."
            )
        if settings.get("solver_backend") == "pardiso" and not HAS_PARDISO:
            result.errors.append("PyPardiso was selected but is not available.")
        return result

    def _derive_stackup_thicknesses(self, board, copper_ids, stack_info, settings):
        """Derive thickness values from stackup or defaults."""
        stack_copper = stack_info.get("copper", []) if isinstance(stack_info, dict) else []
        stack_gaps = stack_info.get("dielectric_gaps_mm", []) if isinstance(stack_info, dict) else []
        stack_board_thick = stack_info.get("board_thickness_mm") if isinstance(stack_info, dict) else None

        copper_thickness_by_id = {}
        copper_thickness_by_name = {}
        for c in stack_copper:
            th = c.get("thickness_mm")
            if isinstance(th, (int, float)):
                lid = c.get("layer_id")
                name = c.get("name")
                if isinstance(lid, int):
                    copper_thickness_by_id[lid] = th
                if isinstance(name, str):
                    copper_thickness_by_name[name] = th

        copper_thickness_mm_used = []
        for lid in copper_ids:
            th = copper_thickness_by_id.get(lid)
            if th is None:
                lname = board.GetLayerName(lid)
                th = copper_thickness_by_name.get(lname)
            if not isinstance(th, (int, float)) or th <= 0:
                th = 0.035
            copper_thickness_mm_used.append(th)

        total_thick_mm_used = settings['thick']
        if isinstance(stack_board_thick, (int, float)) and stack_board_thick > 0:
            total_thick_mm_used = stack_board_thick

        fallback_gap_mm = total_thick_mm_used / max(1, len(copper_ids) - 1)
        gap_mm_used = []
        use_uniform_gap = False
        if len(stack_gaps) != max(0, len(copper_ids) - 1):
            use_uniform_gap = True
        else:
            for g in stack_gaps:
                if not isinstance(g, (int, float)) or g <= 0:
                    use_uniform_gap = True
                    break
        if use_uniform_gap:
            gap_mm_used = [fallback_gap_mm] * max(0, len(copper_ids) - 1)
        else:
            gap_mm_used = [float(g) for g in stack_gaps]

        return {
            "total_thick_mm_used": total_thick_mm_used,
            "stack_board_thick_mm": stack_board_thick,
            "copper_thickness_mm_used": copper_thickness_mm_used,
            "gap_mm_used": gap_mm_used,
            "gap_fallback_used": use_uniform_gap,
        }

    def generate_preview(self, settings, layer_names):
        """Generate geometry preview image."""
        power_pads = self._resolve_power_pad_objects(self.board, settings, legacy_pads=self.pads_list)
        terminals, _ = self._resolve_current_terminals(self.board, settings)
        current_pads = [item.pad for item in terminals]
        preview_pads = self._unique_pads(power_pads + current_pads)
        effective_settings = dict(settings)
        area = _estimate_simulation_area(
            self.board, self.bbox, effective_settings,
            power_pads=power_pads, terminals=terminals,
        )
        effective_settings["_preview_area_summary"] = (
            f"{area.width_mm:.1f} x {area.height_mm:.1f} mm / "
            f"{area.area_fraction * 100.0:.0f}% of board"
        )
        effective_settings["_preview_area_limited"] = area.limited
        grid = _estimate_solver_grid(
            self.bbox, float(effective_settings["res"]), effective_settings,
            len(self.copper_ids), preview_pads, area=area,
        )
        cache_key = self._geometry_key(
            self.board, self.copper_ids, self.bbox, grid, effective_settings, preview_pads
        )
        geometry_state = self.geometry_cache.get(cache_key)
        if geometry_state is None:
            geometry_state = build_geometry_state(
                board=self.board,
                copper_ids=self.copper_ids,
                rows=grid.rows,
                cols=grid.cols,
                x_min=grid.x_min_mm,
                y_min=grid.y_min_mm,
                res=grid.actual_res_mm,
                settings=effective_settings,
                via_factor=390.0 / 0.3,
                pads_list=preview_pads,
            )
            self.geometry_cache.put(cache_key, geometry_state)
        output_file = save_preview_image(
            self.board, self.copper_ids, self.bbox, preview_pads,
            effective_settings, layer_names,
            self.stack_info if self.stack_info is not None else parse_stackup_from_board_file(self.board),
            get_pad_pixels,
            create_multilayer_maps,
            self._derive_stackup_thicknesses,
            open_file=True,
            geometry_state=geometry_state,
            grid_spec=grid,
        )
        if not output_file:
            wx.MessageBox("Board data missing for preview", "Error")
        return output_file

    def _run_simulation(self, board, copper_ids, layer_names, bbox, pads_list,
                        settings, stack_info, pad_names, zone_refill_s=0.0,
                        focus_pads=None):
        """Execute the thermal simulation."""
        run_started_at = time.perf_counter()

        def set_dialog_state(status, message=""):
            if self.settings_dialog is not None:
                try:
                    wx.CallAfter(self.settings_dialog.set_run_state, status, message)
                except Exception:
                    pass

        focus_pads = focus_pads if focus_pads is not None else pads_list
        area_power_pads = self._resolve_power_pad_objects(
            board, settings, legacy_pads=pads_list
        )
        area_terminals = []
        if settings.get("current_enabled"):
            area_terminals, _ = self._resolve_current_terminals(board, settings)
        area = _estimate_simulation_area(
            board, bbox, settings,
            power_pads=area_power_pads,
            terminals=area_terminals,
        )
        # Derive thicknesses
        stackup_derived = self._derive_stackup_thicknesses(board, copper_ids, stack_info, settings)
        total_thick_mm = stackup_derived["total_thick_mm_used"]
        init_timings = {"zone_refill_s": float(zone_refill_s)}

        # Output folder setup
        base_output_dir = settings.get('output_dir') or os.path.dirname(__file__)
        try:
            os.makedirs(base_output_dir, exist_ok=True)
            run_dir = os.path.join(base_output_dir, time.strftime("Thermalsim_%Y%m%d_%H%M%S"))
            os.makedirs(run_dir, exist_ok=True)
            test_path = os.path.join(run_dir, ".write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
        except Exception:
            run_dir = tempfile.mkdtemp(prefix="ThermalSim_")
        _write_run_manifest(run_dir, "running", settings=settings)

        requested_res = float(settings['res'])
        layer_count = len(copper_ids)
        grid = _estimate_solver_grid(
            bbox, requested_res, settings, layer_count, focus_pads, area=area
        )
        res = grid.actual_res_mm
        x_min = grid.x_min_mm
        y_min = grid.y_min_mm
        rows = grid.rows
        cols = grid.cols
        grid_info = {
            "grid_requested_res_mm": requested_res,
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

        # Physical parameters
        k_fr4_rel = 1.0
        via_factor = 390.0 / 0.3
        k_cu = 390.0
        k_fr4 = 0.3
        rho_cu, cp_cu = 8960.0, 385.0
        rho_fr4, cp_fr4 = 1850.0, 1100.0

        cu_thick_mm_used = stackup_derived["copper_thickness_mm_used"]
        gap_mm_used = stackup_derived["gap_mm_used"]
        cu_thick_m = [max(1e-9, th * 1e-3) for th in cu_thick_mm_used]
        gap_m = [max(1e-9, g * 1e-3) for g in gap_mm_used]

        # Build internal geometry state
        geometry_start = time.perf_counter()
        cache_key = self._geometry_key(board, copper_ids, bbox, grid, settings, focus_pads)
        geometry_state = self.geometry_cache.get(cache_key)
        geometry_cache_hit = geometry_state is not None
        try:
            if geometry_state is None:
                geometry_state = build_geometry_state(
                board=board,
                copper_ids=copper_ids,
                rows=rows,
                cols=cols,
                x_min=x_min,
                y_min=y_min,
                res=res,
                settings=settings,
                via_factor=via_factor,
                pads_list=focus_pads,
                )
                self.geometry_cache.put(cache_key, geometry_state)
        except Exception as e:
            wx.MessageBox(f"Error mapping geometry: {e}", "Error")
            set_dialog_state("failed", "Board geometry could not be mapped.")
            return
        init_timings["geometry_maps_s"] = time.perf_counter() - geometry_start
        init_timings["geometry_cache_hit"] = geometry_cache_hit
        copper_mask = geometry_state.copper_mask
        V_map = geometry_state.via_map
        H_map = geometry_state.heatsink_mask.astype(np.float64, copy=False)

        # Time step calculation
        dx = res * 1e-3
        dy = dx
        sim_time = settings['time']
        steps_target = max(1, min(600, max(80, int(120 * (sim_time ** 0.35)))))
        dt = sim_time / steps_target

        # Build capacity array
        amb = settings['amb']
        pixel_area = dx * dy
        copper_threshold_rel = k_fr4_rel * 1.5
        t_cu = np.array(cu_thick_m)

        # Effective FR4 thickness per layer
        if layer_count > 1 and gap_m:
            t_fr4_eff = []
            for i in range(layer_count):
                if i == 0:
                    gap = gap_m[0]
                elif i == layer_count - 1:
                    gap = gap_m[-1]
                else:
                    gap = 0.5 * (gap_m[i - 1] + gap_m[i])
                t_fr4_eff.append(gap)
        else:
            t_fr4_eff = [max(total_thick_mm * 1e-3, 1e-5)] * layer_count
        t_fr4_eff = np.clip(np.array(t_fr4_eff), 1e-6, 5e-3)
        t_fr4_eff_mm = (t_fr4_eff * 1e3).tolist()

        operator_key = stable_fingerprint({
            "geometry": cache_key,
            "copper_thickness_m": [float(value) for value in t_cu],
            "dielectric_gap_m": [float(value) for value in gap_m],
            "fr4_effective_m": [float(value) for value in t_fr4_eff],
            "cell_size_m": float(dx),
            "ambient_c": float(amb),
            "h_conv": float(settings.get("h_conv", 10.0)),
            "pad_th": float(settings.get("pad_th", 1.0)),
            "pad_k": float(settings.get("pad_k", 3.0)),
            "pad_cap_areal": float(settings.get("pad_cap_areal", 0.0) or 0.0),
        })
        cached_operator = self.operator_cache.get(operator_key)
        operator_cache_hit = cached_operator is not None

        # Heat capacity
        capacity_start = time.perf_counter()
        pad_cap_areal = float(settings.get('pad_cap_areal', 0.0) or 0.0)
        if operator_cache_hit:
            C, K_matrix, b, hA = cached_operator
        else:
            C_layers = np.empty((layer_count, rows, cols), dtype=np.float64)
            for l in range(layer_count):
                V_cu = pixel_area * t_cu[l]
                V_fr4 = pixel_area * t_fr4_eff[l]
                mask = copper_mask[l]
                C_layer = np.where(mask, rho_cu * cp_cu * V_cu, rho_fr4 * cp_fr4 * V_fr4)
                C_layer += mask * (rho_fr4 * cp_fr4 * V_fr4)
                C_layers[l] = C_layer
            if pad_cap_areal > 0.0 and np.any(H_map):
                pad_cap_per_cell = pad_cap_areal * pixel_area
                C_layers[-1] += pad_cap_per_cell * H_map
            C = C_layers.reshape(-1)
        init_timings["capacity_build_s"] = time.perf_counter() - capacity_start
        init_timings["operator_cache_hit"] = operator_cache_hit

        # Power injection (supports constant values and PWL file paths)
        RC = rows * cols
        N = RC * layer_count
        power_start = time.perf_counter()

        has_power_pad_settings = "power_pads" in settings
        missing_power_pads = []
        if has_power_pad_settings:
            power_pad_entries, missing_power_pads = self._resolve_power_pad_entries(board, settings)
            power_pads = [pad for _, pad, _ in power_pad_entries]
            power_pad_names = [descriptor.get("name", "") for descriptor, _, _ in power_pad_entries]
            entries = [power for _, _, power in power_pad_entries if str(power).strip()]
        else:
            power_pads = list(pads_list or [])
            power_pad_names = list(pad_names or [])
            entries = [x.strip() for x in settings.get('power_str', '').split(',') if x.strip()]
            if len(entries) == 1:
                entries = entries * len(power_pads)

        if missing_power_pads:
            wx.MessageBox(
                "Manual power references pads that were not found on the board:\n"
                + "\n".join(str(name) for name in missing_power_pads),
                "Power Pad Error"
            )
            set_dialog_state("failed", "One or more heat-source pads were not found.")
            return

        # Parse each entry as constant float or PWL file path
        pad_sources = []  # ('const', float) or ('pwl', (times, powers))
        for entry in entries:
            try:
                pad_sources.append(('const', float(entry)))
            except ValueError:
                try:
                    times_pwl, powers_pwl = parse_pwl_file(entry)
                    pad_sources.append(('pwl', (times_pwl, powers_pwl)))
                except (FileNotFoundError, ValueError) as e:
                    wx.MessageBox(
                        f"Error reading PWL file:\n{entry}\n\n{e}",
                        "PWL Error"
                    )
                    set_dialog_state("failed", "A PWL power profile could not be read.")
                    return

        if len(pad_sources) == 1 and len(power_pads) > 1:
            pad_sources = pad_sources * len(power_pads)
            entries = entries * len(power_pads)

        if len(pad_sources) != len(power_pads):
            wx.MessageBox(
                f"Number of power entries ({len(pad_sources)}) does not match "
                f"number of power pads ({len(power_pads)}).",
                "Warning"
            )

        pad_contributions = _build_sparse_pad_contributions(
            board=board,
            copper_ids=copper_ids,
            pads_list=power_pads,
            rows=rows,
            cols=cols,
            x_min=x_min,
            y_min=y_min,
            res=res,
        )
        Q, Q_func = _build_power_vector(pad_sources, pad_contributions, N)
        init_timings["power_vector_build_s"] = time.perf_counter() - power_start

        electrical_summary = None
        q_joule = None
        if settings.get("current_enabled"):
            electrical_start = time.perf_counter()
            electrical_cache_hit = False
            terminals, missing_pads = self._resolve_current_terminals(board, settings)
            if missing_pads:
                wx.MessageBox(
                    "Current simulation references pads that were not found on the board:\n"
                    + "\n".join(str(name) for name in missing_pads),
                    "Current Path Error"
                )
                set_dialog_state("failed", "One or more current-terminal pads were not found.")
                return
            if terminals:
                electrical_config = ElectricalConfig(
                    copper_ids=list(copper_ids),
                    rows=rows,
                    cols=cols,
                    x_min=x_min,
                    y_min=y_min,
                    res=res,
                    t_cu=np.asarray(t_cu, dtype=np.float64),
                    layer_names={lid: layer_names[idx] for idx, lid in enumerate(copper_ids) if idx < len(layer_names)},
                )
                electrical_key = stable_fingerprint({
                    "geometry": cache_key,
                    "copper_thickness_m": [float(value) for value in t_cu],
                    "terminals": [
                        (item.name, item.net_name, int(item.net_code), float(item.current_a))
                        for item in terminals
                    ],
                })
                electrical_result = self.electrical_cache.get(electrical_key)
                electrical_cache_hit = electrical_result is not None
                if electrical_result is None:
                    electrical_result = solve_electrical_heating(board, terminals, electrical_config)
                    if electrical_result.valid:
                        self.electrical_cache.put(electrical_key, electrical_result)
                if not electrical_result.valid:
                    wx.MessageBox(
                        "Current simulation validation failed:\n\n"
                        + "\n".join(electrical_result.errors),
                        "Current Path Error"
                    )
                    set_dialog_state("failed", "Current-path validation failed.")
                    return
                q_joule = electrical_result.q_joule
                if np.any(q_joule):
                    Q = Q + q_joule
                    if Q_func is not None:
                        base_q_func = Q_func

                        def q_func_with_joule(t, _base=base_q_func, _q_joule=q_joule):
                            return _base(t) + _q_joule

                        Q_func = q_func_with_joule
                electrical_summary = {
                    "total_loss_w": electrical_result.total_loss_w,
                    "warnings": electrical_result.warnings,
                    "nets": [_electrical_net_summary_dict(item) for item in electrical_result.net_summaries],
                }
            else:
                electrical_summary = {
                    "total_loss_w": 0.0,
                    "warnings": ["Current simulation enabled but no non-zero pad currents were configured."],
                    "nets": [],
                }
            init_timings["electrical_solve_s"] = time.perf_counter() - electrical_start
            init_timings["electrical_cache_hit"] = electrical_cache_hit

        # Build pad_power for reporting
        pad_power = []
        for i, name in enumerate(power_pad_names):
            if i < len(pad_sources):
                stype, sval = pad_sources[i]
                if stype == 'const':
                    pad_power.append((name, sval))
                else:
                    entry_label = entries[i] if i < len(entries) else ""
                    pad_power.append((name, f"PWL:{entry_label}"))
            else:
                pad_power.append((name, None))

        # Build stiffness matrix
        stiffness_start = time.perf_counter()
        if not operator_cache_hit:
            K_matrix, b, hA, _ = build_stiffness_matrix(
                layer_count, rows, cols, copper_mask, t_cu, t_fr4_eff,
                k_cu, k_fr4, dx, dy, V_map, gap_m, H_map, settings, amb
            )
            self.operator_cache.put(operator_key, (C, K_matrix, b, hA))
        init_timings["stiffness_matrix_s"] = time.perf_counter() - stiffness_start
        print(f"[ThermalSim] init timings: {_format_timing_summary(init_timings)}")

        # Snapshot configuration
        snap_times = []
        if settings.get('snapshots'):
            snap_count = max(1, min(50, int(settings.get('snap_count', 5))))
            snap_times = [sim_time * (k / (snap_count + 1)) for k in range(1, snap_count + 1)]
        snap_times = sorted({t for t in snap_times if 0.0 < t < sim_time})

        # Progress dialog
        pd = wx.ProgressDialog(
            "ThermalSim", "Preparing thermal model...", 100,
            style=wx.PD_CAN_ABORT | wx.PD_APP_MODAL | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE
        )

        progress_state = {"current": 0, "total": 1}
        cancel_token = CancellationToken()
        self.cancel_token = cancel_token

        def progress_callback(current, total):
            progress_state["current"] = int(current)
            progress_state["total"] = max(1, int(total))
            return not cancel_token.cancelled

        def snapshot_callback(T_view, t_elapsed, snap_idx):
            return save_snapshot(T_view, H_map, amb, layer_names, snap_idx, t_elapsed, out_dir=run_dir)

        # Run solver
        config = SolverConfig(
            sim_time=sim_time,
            amb=amb,
            dt_base=dt,
            steps_target=steps_target,
            use_pardiso=HAS_PARDISO and settings.get("solver_backend", "auto") != "scipy",
            use_multi_phase=True,
            time_stepping=settings.get("time_stepping", "auto"),
            snapshots_enabled=settings.get('snapshots', False),
            snap_times=snap_times
        )
        factorization_cache_key = stable_fingerprint({
            "operator": operator_key,
            "backend": "pardiso" if config.use_pardiso else "scipy",
            "time_stepping": config.time_stepping,
            "sim_time": float(config.sim_time),
            "dt_base": float(config.dt_base),
            "steps_target": int(config.steps_target),
            "use_multi_phase": bool(config.use_multi_phase),
        })

        worker_result = {}

        def solver_worker():
            try:
                worker_result["result"] = run_simulation(
                    config, K_matrix, C, Q, b, hA,
                    layer_count, rows, cols,
                    progress_callback, snapshot_callback,
                    Q_func=Q_func,
                    cancel_check=lambda: cancel_token.cancelled,
                    factorization_cache=self.factorization_cache,
                    factorization_cache_key=factorization_cache_key,
                )
            except Exception:
                worker_result["traceback"] = traceback.format_exc()

        worker = threading.Thread(target=solver_worker, name="ThermalSimSolver", daemon=True)
        worker.start()
        while worker.is_alive():
            current = progress_state["current"]
            total = progress_state["total"]
            percent = int((current / total) * 100) if total else 0
            try:
                update_result = pd.Update(percent, f"Solving thermal model - step {current}/{total}")
                keep_going = update_result[0] if isinstance(update_result, tuple) else update_result
                if not keep_going or (hasattr(pd, "WasCancelled") and pd.WasCancelled()):
                    cancel_token.cancel()
                app = wx.GetApp()
                if app is not None:
                    app.Yield(True)
            except Exception:
                cancel_token.cancel()
            worker.join(0.05)
        worker.join()
        self.cancel_token = None

        try:
            pd.Update(100, "Done")
        except Exception:
            pass
        pd.Hide()
        pd.Destroy()
        try:
            app = wx.GetApp()
            if app is not None:
                app.Yield()
        except Exception:
            pass

        if "traceback" in worker_result:
            error_text = worker_result["traceback"]
            _write_run_manifest(run_dir, "error", traceback=error_text)
            wx.MessageBox(f"Solver failed:\n{error_text}", "Solver Error")
            set_dialog_state("failed", "The thermal solver failed. See the error dialog for details.")
            return
        result = worker_result["result"]

        if result.aborted:
            _write_run_manifest(run_dir, "cancelled")
            set_dialog_state("cancelled")
            return

        # Add extra info to k_norm_info
        result.k_norm_info.update({
            **grid_info,
            "copper_threshold_rel": copper_threshold_rel,
            "t_fr4_eff_min": float(np.min(t_fr4_eff)),
            "t_fr4_eff_max": float(np.max(t_fr4_eff)),
            "t_fr4_eff_per_plane_mm": t_fr4_eff_mm,
            "pad_cap_input_areal": pad_cap_areal,
            "h_top": float(settings.get('h_conv', 10.0)),
            "h_air_bottom": float(settings.get('h_conv', 10.0)),
            "init_zone_refill_s": init_timings.get("zone_refill_s"),
            "init_geometry_maps_s": init_timings.get("geometry_maps_s"),
            "init_capacity_build_s": init_timings.get("capacity_build_s"),
            "init_power_vector_build_s": init_timings.get("power_vector_build_s"),
            "init_electrical_solve_s": init_timings.get("electrical_solve_s"),
            "init_electrical_cache_hit": init_timings.get("electrical_cache_hit"),
            "init_stiffness_matrix_s": init_timings.get("stiffness_matrix_s"),
            "init_operator_cache_hit": init_timings.get("operator_cache_hit"),
            "factorization_cache_hit": result.k_norm_info.get("factorization_cache_hit"),
            "electrical_summary": electrical_summary,
        })

        # Save results
        if settings['show_all']:
            heatmap_path = show_results_all_layers(
                result.T, H_map, amb, layer_names,
                open_file=False, t_elapsed=sim_time, out_dir=run_dir
            )
        else:
            heatmap_path = show_results_top_bot(
                result.T, H_map, amb,
                open_file=False, t_elapsed=sim_time, out_dir=run_dir
            )

        joule_map_path = None
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
                out_dir=run_dir,
            )

        preview_path = save_preview_image(
            board, copper_ids, bbox, focus_pads,
            settings, layer_names, stack_info,
            get_pad_pixels, create_multilayer_maps,
            self._derive_stackup_thicknesses,
            open_file=False, out_dir=run_dir,
            geometry_state=geometry_state,
            grid_spec=grid,
        )

        interactive_heatmap = build_interactive_heatmap_payload(
            result.T,
            amb=amb,
            layer_names=layer_names,
            res_mm=res,
            x_min_mm=x_min,
            y_min_mm=y_min,
            show_all=bool(settings.get('show_all', True))
        )

        snapshot_debug = {
            "snapshots_enabled": settings.get('snapshots'),
            "snap_count": settings.get('snap_count'),
            "dt_base": dt,
            "steps_target": steps_target,
            "steps_total": result.step_counter,
            "snap_times": snap_times,
            "base_output_dir": base_output_dir,
            "run_dir": run_dir,
            "solver_backend": result.k_norm_info.get("backend"),
            "avg_solve_s": result.total_solve_time / max(result.step_counter, 1),
            "factorizations": result.factor_count,
            "factorization_s": result.total_factor_time,
            "phase_metrics": json.dumps(result.phase_metrics),
            "init_zone_refill_s": init_timings.get("zone_refill_s"),
            "init_geometry_maps_s": init_timings.get("geometry_maps_s"),
            "init_capacity_build_s": init_timings.get("capacity_build_s"),
            "init_power_vector_build_s": init_timings.get("power_vector_build_s"),
            "init_electrical_solve_s": init_timings.get("electrical_solve_s"),
            "init_electrical_cache_hit": init_timings.get("electrical_cache_hit"),
            "init_stiffness_matrix_s": init_timings.get("stiffness_matrix_s"),
            "init_operator_cache_hit": init_timings.get("operator_cache_hit"),
            "factorization_cache_hit": result.k_norm_info.get("factorization_cache_hit"),
        }

        report_path = write_html_report(
            settings=settings,
            stack_info=stack_info,
            stackup_derived=stackup_derived,
            pad_power=pad_power,
            layer_names=layer_names,
            preview_path=preview_path,
            heatmap_path=heatmap_path,
            k_norm_info=result.k_norm_info,
            out_dir=run_dir,
            snapshot_debug=snapshot_debug,
            snapshot_files=result.snapshot_files,
            interactive_heatmap=interactive_heatmap,
            electrical_summary=electrical_summary,
            joule_map_path=joule_map_path
        )
        _write_run_manifest(
            run_dir,
            "success",
            report_path=report_path,
            heatmap_path=heatmap_path,
            preview_path=preview_path,
        )
        elapsed_s = time.perf_counter() - run_started_at
        max_temp_c = float(np.max(result.T))
        self.last_artifacts = SimulationArtifacts(
            report_path=report_path,
            preview_path=preview_path,
            heatmap_path=heatmap_path,
            run_dir=run_dir,
            status="success",
            elapsed_s=elapsed_s,
            max_temp_c=max_temp_c,
        )
        if self.settings_dialog is not None:
            try:
                wx.CallAfter(
                    self.settings_dialog.set_artifacts,
                    report_path,
                    run_dir,
                    elapsed_s,
                    max_temp_c,
                )
            except Exception:
                pass

        # Open outputs
        if report_path:
            def _open_outputs():
                try:
                    import webbrowser
                    webbrowser.open("file://" + os.path.abspath(report_path))
                except Exception:
                    pass
            wx.CallAfter(_open_outputs)
