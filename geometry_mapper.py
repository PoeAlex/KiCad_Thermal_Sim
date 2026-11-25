"""
PCB geometry to thermal grid mapping.

This module converts KiCad PCB geometry (copper, vias, zones) into
discretized conductivity arrays for thermal simulation.
"""

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pcbnew


class GeometryCancelledError(RuntimeError):
    """Raised when geometry mapping or rasterization is cancelled."""


class _ProgressEmitter:
    """Throttled progress reporting and cancellation check helper."""

    def __init__(
        self,
        progress_callback: Optional[Callable[[int, int, str], Optional[bool]]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        interval_s: float = 0.05,
    ):
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.interval_s = float(interval_s)
        self._last_emit = 0.0
        self._last_check = 0.0
        self.active = (progress_callback is not None) or (cancel_check is not None)

    def check_cancel(self, force: bool = False) -> None:
        """Raise GeometryCancelledError if cancellation was requested."""
        if self.cancel_check is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_check) < self.interval_s:
            return
        self._last_check = now
        if self.cancel_check():
            raise GeometryCancelledError("Geometry mapping cancelled.")

    def emit(self, done: int, total: int, message: str, force: bool = False) -> None:
        """Emit throttled progress and check cancellation."""
        if not self.active:
            return
        now = time.monotonic()
        if self.cancel_check is not None:
            if force or (now - self._last_check) >= self.interval_s:
                self._last_check = now
                if self.cancel_check():
                    raise GeometryCancelledError("Geometry mapping cancelled.")
        if self.progress_callback is not None:
            if force or (now - self._last_emit) >= self.interval_s:
                self._last_emit = now
                if self.progress_callback(int(done), int(total), message) is False:
                    raise GeometryCancelledError("Geometry mapping cancelled.")


@dataclass
class FillContext:
    """
    Context for geometry fill operations.

    This dataclass holds all the arrays and parameters needed by the
    fill functions, replacing nested closures with explicit state.

    Attributes
    ----------
    K : np.ndarray
        Thermal conductivity map, shape (layers, rows, cols).
    V : np.ndarray
        Via enhancement map, shape (rows, cols).
    H : np.ndarray
        Heatsink/thermal-pad mask, shape (rows, cols).
    area_mask : np.ndarray or None
        Boolean mask limiting the simulation area.
    x_min : float
        X coordinate of grid origin in mm.
    y_min : float
        Y coordinate of grid origin in mm.
    res : float
        Grid resolution in mm.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    """
    K: np.ndarray
    V: np.ndarray
    H: np.ndarray
    area_mask: Optional[np.ndarray]
    x_min: float
    y_min: float
    res: float
    rows: int
    cols: int


@dataclass
class GeometryState:
    """
    Internal geometry representation for fast map construction.

    Attributes
    ----------
    copper_mask : np.ndarray
        Boolean copper occupancy mask, shape (layers, rows, cols).
    via_map : np.ndarray
        Via enhancement map, shape (rows, cols).
    heatsink_mask : np.ndarray
        Boolean heatsink/thermal-pad mask, shape (rows, cols).
    area_mask : np.ndarray or None
        Boolean mask limiting the simulation area.
    x_min : float
        X coordinate of grid origin in mm.
    y_min : float
        Y coordinate of grid origin in mm.
    res : float
        Grid resolution in mm.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    x_centers_iu : np.ndarray
        Precomputed x cell centers in KiCad internal units.
    y_centers_iu : np.ndarray
        Precomputed y cell centers in KiCad internal units.
    via_map_3d : np.ndarray or None
        Optional per-gap via enhancement map of shape
        ``(layer_count - 1, rows, cols)`` when blind/buried vias exist on a
        multilayer board (``layer_count > 2``); ``None`` when all vias are
        through-hole.
    """
    copper_mask: np.ndarray
    via_map: np.ndarray
    heatsink_mask: np.ndarray
    area_mask: Optional[np.ndarray]
    x_min: float
    y_min: float
    res: float
    rows: int
    cols: int
    x_centers_iu: np.ndarray
    y_centers_iu: np.ndarray
    via_map_3d: Optional[np.ndarray] = None


def normalize_via_fill(value: Any) -> str:
    """Normalize a via fill specification to ``'copper'`` or ``'plated'``."""
    if value is None:
        return "plated"
    norm = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if norm in {"copper", "copper_filled"}:
        return "copper"
    return "plated"


def _via_areas_m2(
    drill_x_m: float,
    drill_y_m: Optional[float] = None,
    *,
    plating_um: float = 25.0,
) -> Tuple[float, float]:
    """Return ``(A_barrel, A_core)`` in square meters for a round or oval drill."""
    dx = max(0.0, float(drill_x_m))
    dy = dx if drill_y_m is None else max(0.0, float(drill_y_m))
    if dx <= 0.0 or dy <= 0.0:
        return 0.0, 0.0
    t_wall = max(0.0, float(plating_um)) * 1e-6
    if drill_y_m is None or abs(dx - dy) <= 1e-12:
        r_i = 0.5 * dx
        r_o = r_i + t_wall
        a_barrel = math.pi * (r_o * r_o - r_i * r_i)
        a_core = math.pi * r_i * r_i
    else:
        a = max(dx, dy)
        b = min(dx, dy)
        p_hole = 2.0 * (a - b) + math.pi * b
        a_barrel = p_hole * t_wall + math.pi * t_wall * t_wall
        a_core = b * (a - b) + 0.25 * math.pi * b * b
    return a_barrel, a_core


def compute_via_copper_area_m2(
    drill_x_m: float,
    drill_y_m: Optional[float] = None,
    *,
    plating_um: float = 25.0,
    via_fill: str = "plated",
) -> float:
    """Return effective vertical copper cross-sectional area (m^2) of a via/PTH."""
    a_barrel, a_core = _via_areas_m2(drill_x_m, drill_y_m, plating_um=plating_um)
    if normalize_via_fill(via_fill) == "copper":
        return a_barrel + a_core
    return a_barrel


def compute_via_excess_ka(
    drill_x_m: float,
    drill_y_m: Optional[float] = None,
    *,
    plating_um: float = 25.0,
    via_fill: str = "plated",
    k_cu: float = 390.0,
    k_fr4: float = 0.3,
) -> float:
    """Compute excess vertical thermal conductance-times-thickness ``delta(kA)`` (W*m/K)."""
    a_barrel, a_core = _via_areas_m2(drill_x_m, drill_y_m, plating_um=plating_um)
    if a_barrel <= 0.0 and a_core <= 0.0:
        return 0.0
    fill_mode = normalize_via_fill(via_fill)
    k_fill = float(k_cu) if fill_mode == "copper" else 0.026
    return max(
        0.0,
        (float(k_cu) - float(k_fr4)) * a_barrel
        + (k_fill - float(k_fr4)) * a_core,
    )


def _bbox_to_grid_indices(bbox, ctx):
    """
    Convert a KiCad bounding box to grid indices.

    Parameters
    ----------
    bbox : pcbnew.EDA_RECT
        Bounding box in internal units (nm).
    ctx : FillContext
        Grid context with origin and resolution.

    Returns
    -------
    tuple
        (rs, re, cs, ce) - row start, row end, col start, col end.
        Returns valid slicing indices clamped to grid bounds.
    """
    x0, y0 = bbox.GetX() * 1e-6, bbox.GetY() * 1e-6
    w, h = bbox.GetWidth() * 1e-6, bbox.GetHeight() * 1e-6
    cs = max(0, int((x0 - ctx.x_min) / ctx.res))
    rs = max(0, int((y0 - ctx.y_min) / ctx.res))
    ce = min(ctx.cols, int((x0 + w - ctx.x_min) / ctx.res) + 1)
    re = min(ctx.rows, int((y0 + h - ctx.y_min) / ctx.res) + 1)
    return rs, re, cs, ce


def _grid_centers_to_iu(count, origin_mm, res_mm):
    """
    Convert grid cell centers to KiCad internal units.

    Parameters
    ----------
    count : int
        Number of cells along one axis.
    origin_mm : float
        Grid origin in millimeters.
    res_mm : float
        Grid spacing in millimeters.

    Returns
    -------
    np.ndarray
        Cell centers in KiCad internal units.
    """
    centers_mm = origin_mm + (np.arange(count, dtype=np.float64) + 0.5) * res_mm
    return np.asarray(centers_mm * 1e6, dtype=np.int64)


def _fill_box(ctx, l_idx, bbox, val):
    """
    Fill a rectangular region in the conductivity map.

    Parameters
    ----------
    ctx : FillContext
        Grid context.
    l_idx : int
        Layer index.
    bbox : pcbnew.EDA_RECT
        Bounding box to fill.
    val : float
        Conductivity value to set (uses max with existing).
    """
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, ctx)
    if cs < ce and rs < re and rs < ctx.rows and cs < ctx.cols:
        if ctx.area_mask is None:
            ctx.K[l_idx, rs:re, cs:ce] = np.maximum(ctx.K[l_idx, rs:re, cs:ce], val)
        else:
            region_mask = ctx.area_mask[rs:re, cs:ce]
            if np.any(region_mask):
                K_slice = ctx.K[l_idx, rs:re, cs:ce]
                np.maximum(K_slice, val, out=K_slice, where=region_mask)


def _fill_via(ctx, bbox, val):
    """
    Fill a via region in the vertical conductivity map.

    Parameters
    ----------
    ctx : FillContext
        Grid context.
    bbox : pcbnew.EDA_RECT
        Bounding box of the via.
    val : float
        Via enhancement factor.
    """
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, ctx)
    if cs < ce and rs < re and rs < ctx.rows and cs < ctx.cols:
        if ctx.area_mask is None:
            ctx.V[rs:re, cs:ce] = np.maximum(ctx.V[rs:re, cs:ce], val)
        else:
            region_mask = ctx.area_mask[rs:re, cs:ce]
            if np.any(region_mask):
                V_slice = ctx.V[rs:re, cs:ce]
                np.maximum(V_slice, val, out=V_slice, where=region_mask)


def _fill_heatsink(ctx, bbox):
    """
    Mark a region as heatsink/thermal-pad area.

    Parameters
    ----------
    ctx : FillContext
        Grid context.
    bbox : pcbnew.EDA_RECT
        Bounding box to mark.
    """
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, ctx)
    if cs < ce and rs < re and rs < ctx.rows and cs < ctx.cols:
        if ctx.area_mask is None:
            ctx.H[rs:re, cs:ce] = 1.0
        else:
            region_mask = ctx.area_mask[rs:re, cs:ce]
            if np.any(region_mask):
                H_slice = ctx.H[rs:re, cs:ce]
                H_slice[region_mask] = 1.0


def _fill_zone(ctx, l_idx, lid, zone, val):
    """
    Fill a copper zone using hit-testing for accurate fill detection.

    This function uses KiCad's HitTestFilledArea to respect zone clearances,
    keepouts, and unfilled areas.

    Parameters
    ----------
    ctx : FillContext
        Grid context.
    l_idx : int
        Layer index in the conductivity array.
    lid : int
        KiCad layer ID for hit testing.
    zone : pcbnew.ZONE
        The zone to fill.
    val : float
        Conductivity value to set.
    """
    bbox = zone.GetBoundingBox()
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, ctx)
    if cs >= ce or rs >= re:
        return

    has_filled_hit = hasattr(zone, "HitTestFilledArea")

    def to_iu(value_mm):
        try:
            return pcbnew.FromMM(value_mm)
        except Exception:
            return int(value_mm * 1e6)

    # Tiny margin to avoid edge quantization misses
    margin_iu = 1

    for r in range(rs, re):
        y = ctx.y_min + (r + 0.5) * ctx.res
        y_iu = to_iu(y)
        for c in range(cs, ce):
            x = ctx.x_min + (c + 0.5) * ctx.res
            pos = pcbnew.VECTOR2I(to_iu(x), y_iu)
            try:
                if ctx.area_mask is not None and not ctx.area_mask[r, c]:
                    continue
                hit = False
                if has_filled_hit:
                    hit = zone.HitTestFilledArea(lid, pos, margin_iu)
                elif hasattr(zone, "HitTest"):
                    hit = zone.HitTest(pos)
                if hit:
                    ctx.K[l_idx, r, c] = max(ctx.K[l_idx, r, c], val)
            except Exception:
                continue


def _state_fill_box(state, l_idx, bbox):
    """Mark copper occupancy in a rectangular region."""
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, state)
    if cs >= ce or rs >= re or rs >= state.rows or cs >= state.cols:
        return

    target = state.copper_mask[l_idx, rs:re, cs:ce]
    if state.area_mask is None:
        target[...] = True
        return

    region_mask = state.area_mask[rs:re, cs:ce]
    if np.any(region_mask):
        target |= region_mask


def _state_apply_shape(
    state,
    l_idx,
    bbox,
    predicate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tile_size=512,
):
    """Rasterize a shape predicate into one copper layer in bounded tiles."""
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, state)
    if cs >= ce or rs >= re or rs >= state.rows or cs >= state.cols:
        return

    for r0 in range(rs, re, tile_size):
        r1 = min(re, r0 + tile_size)
        y = state.y_min + (np.arange(r0, r1, dtype=np.float64) + 0.5) * state.res
        for c0 in range(cs, ce, tile_size):
            c1 = min(ce, c0 + tile_size)
            if state.area_mask is not None:
                allowed = state.area_mask[r0:r1, c0:c1]
                if not np.any(allowed):
                    continue
            else:
                allowed = None
            x = state.x_min + (np.arange(c0, c1, dtype=np.float64) + 0.5) * state.res
            xx, yy = np.meshgrid(x, y)
            inside = np.asarray(predicate(xx, yy), dtype=bool)
            if allowed is not None:
                inside &= allowed
            if np.any(inside):
                state.copper_mask[l_idx, r0:r1, c0:c1] |= inside


def _segment_parameters(start, end):
    """Return a line-segment representation in millimetres."""
    sx = float(start.x) * 1e-6
    sy = float(start.y) * 1e-6
    ex = float(end.x) * 1e-6
    ey = float(end.y) * 1e-6
    vx = ex - sx
    vy = ey - sy
    return sx, sy, vx, vy, vx * vx + vy * vy


def _state_fill_segment(state, l_idx, track):
    """Rasterize a straight track using its centreline and actual width."""
    try:
        start = track.GetStart()
        end = track.GetEnd()
        width_mm = max(float(track.GetWidth()) * 1e-6, 0.0)
        bbox = track.GetBoundingBox()
    except Exception:
        _state_fill_box(state, l_idx, track.GetBoundingBox())
        return

    sx, sy, vx, vy, seg_len_sq = _segment_parameters(start, end)
    radius = 0.5 * width_mm

    def predicate(xx, yy):
        if seg_len_sq <= 1e-24:
            return np.hypot(xx - sx, yy - sy) <= radius
        projection = ((xx - sx) * vx + (yy - sy) * vy) / seg_len_sq
        projection = np.clip(projection, 0.0, 1.0)
        px = sx + projection * vx
        py = sy + projection * vy
        return np.hypot(xx - px, yy - py) <= radius

    _state_apply_shape(state, l_idx, bbox, predicate)


def _circle_from_three_points(start, middle, end):
    """Return circle centre and radius for three KiCad points, if defined."""
    x1, y1 = float(start.x) * 1e-6, float(start.y) * 1e-6
    x2, y2 = float(middle.x) * 1e-6, float(middle.y) * 1e-6
    x3, y3 = float(end.x) * 1e-6, float(end.y) * 1e-6
    det = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(det) <= 1e-18:
        return None
    u1 = x1 * x1 + y1 * y1
    u2 = x2 * x2 + y2 * y2
    u3 = x3 * x3 + y3 * y3
    cx = (u1 * (y2 - y3) + u2 * (y3 - y1) + u3 * (y1 - y2)) / det
    cy = (u1 * (x3 - x2) + u2 * (x1 - x3) + u3 * (x2 - x1)) / det
    return cx, cy, math.hypot(x1 - cx, y1 - cy)


def _state_fill_arc(state, l_idx, track):
    """Rasterize a circular PCB arc when start/mid/end are available."""
    try:
        start = track.GetStart()
        middle = track.GetMid()
        end = track.GetEnd()
        circle = _circle_from_three_points(start, middle, end)
        if circle is None:
            _state_fill_segment(state, l_idx, track)
            return
        cx, cy, radius = circle
        half_width = 0.5 * float(track.GetWidth()) * 1e-6
        bbox = track.GetBoundingBox()
        start_angle = math.atan2(float(start.y) * 1e-6 - cy, float(start.x) * 1e-6 - cx)
        middle_angle = math.atan2(float(middle.y) * 1e-6 - cy, float(middle.x) * 1e-6 - cx)
        end_angle = math.atan2(float(end.y) * 1e-6 - cy, float(end.x) * 1e-6 - cx)
    except Exception:
        _state_fill_segment(state, l_idx, track)
        return

    def predicate(xx, yy):
        dx = xx - cx
        dy = yy - cy
        radial = np.abs(np.hypot(dx, dy) - radius) <= half_width
        angles = np.arctan2(dy, dx)
        tau = 2.0 * np.pi
        ccw_span = (end_angle - start_angle) % tau
        ccw_middle = (middle_angle - start_angle) % tau
        if ccw_middle <= ccw_span:
            angular = np.mod(angles - start_angle, tau) <= ccw_span
        else:
            clockwise_span = (start_angle - end_angle) % tau
            angular = np.mod(start_angle - angles, tau) <= clockwise_span
        return radial & angular

    _state_apply_shape(state, l_idx, bbox, predicate)


def _state_fill_circle(state, layer_indices, bbox, center, diameter_mm):
    """Rasterize a circular pad or via on one or more layers with sub-pixel fallback."""
    cx = float(center.x) * 1e-6
    cy = float(center.y) * 1e-6
    radius = 0.5 * max(float(diameter_mm), 0.0)

    def predicate(xx, yy):
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius

    for layer_idx in layer_indices:
        _state_apply_shape(state, layer_idx, bbox, predicate)

    if state.rows <= 0 or state.cols <= 0 or state.res <= 0.0 or not layer_indices:
        return
    grid_x_max = state.x_min + state.cols * state.res
    grid_y_max = state.y_min + state.rows * state.res
    if (
        cx + radius < state.x_min
        or cx - radius > grid_x_max
        or cy + radius < state.y_min
        or cy - radius > grid_y_max
    ):
        return

    c_c = min(state.cols - 1, max(0, int(math.floor((cx - state.x_min) / state.res))))
    r_c = min(state.rows - 1, max(0, int(math.floor((cy - state.y_min) / state.res))))
    if state.area_mask is not None and not bool(state.area_mask[r_c, c_c]):
        return

    rs, re, cs, ce = _bbox_to_grid_indices(bbox, state)
    for layer_idx in layer_indices:
        if cs >= ce or rs >= re or not np.any(state.copper_mask[layer_idx, rs:re, cs:ce]):
            state.copper_mask[layer_idx, r_c, c_c] = True


def _pad_orientation_radians(pad) -> float:
    """Return pad rotation in radians across supported KiCad APIs."""
    try:
        return math.radians(float(pad.GetOrientationDegrees()))
    except Exception:
        pass
    try:
        orientation = pad.GetOrientation()
        if hasattr(orientation, "AsDegrees"):
            return math.radians(float(orientation.AsDegrees()))
        return math.radians(float(orientation) / 10.0)
    except Exception:
        return 0.0


def get_pad_copper_layer_indices(
    pad: Any,
    copper_ids: List[int],
    lid_to_idx: Optional[Dict[int, int]] = None,
) -> List[int]:
    """
    Resolve the indices in ``copper_ids`` occupied by a KiCad pad.

    KiCad 10 ``PAD.GetLayer()`` returns ``0 (F.Cu)`` even for ``B.Cu`` SMD and
    NPTH pads, so ``GetLayerSet()`` must be inspected first before falling back
    to attribute/layer heuristics for minimal mocks.
    """
    if not copper_ids:
        return []
    if lid_to_idx is None:
        lid_to_idx = {lid: idx for idx, lid in enumerate(copper_ids)}

    try:
        pad_attrib = pad.GetAttribute()
    except Exception:
        pad_attrib = None

    npth_attrib = getattr(pcbnew, "PAD_ATTRIB_NPTH", None)
    pth_attrib = getattr(pcbnew, "PAD_ATTRIB_PTH", None)

    if hasattr(pad, "GetLayerSet"):
        try:
            layer_set = pad.GetLayerSet()
        except Exception:
            layer_set = None
        if layer_set is not None:
            matched: List[int] = []
            if hasattr(layer_set, "Contains"):
                try:
                    matched = [
                        idx for idx, lid in enumerate(copper_ids)
                        if layer_set.Contains(lid)
                    ]
                except Exception:
                    matched = []
            if not matched and hasattr(layer_set, "CuStack"):
                try:
                    matched = [
                        lid_to_idx[int(lid)]
                        for lid in layer_set.CuStack()
                        if int(lid) in lid_to_idx
                    ]
                except Exception:
                    matched = []
            if matched:
                return matched
            if npth_attrib is not None and pad_attrib == npth_attrib:
                return []

    if pth_attrib is not None and pad_attrib == pth_attrib:
        return list(range(len(copper_ids)))

    if npth_attrib is not None and pad_attrib == npth_attrib:
        return []

    try:
        layer_idx = lid_to_idx.get(pad.GetLayer())
        if layer_idx is not None:
            return [layer_idx]
    except Exception:
        pass

    if hasattr(pad, "GetParent"):
        try:
            parent = pad.GetParent()
            if parent is not None and hasattr(parent, "GetLayer"):
                parent_idx = lid_to_idx.get(parent.GetLayer())
                if parent_idx is not None:
                    return [parent_idx]
        except Exception:
            pass

    return []


def get_pad_primary_layer_id(
    pad: Any,
    copper_ids: Optional[List[int]] = None,
) -> int:
    """Return the primary KiCad layer ID for a pad (correcting ``B.Cu`` SMD)."""
    if copper_ids:
        indices = get_pad_copper_layer_indices(pad, copper_ids)
        if indices:
            if len(indices) > 1 and hasattr(pad, "GetParent"):
                try:
                    parent = pad.GetParent()
                    if parent is not None and hasattr(parent, "GetLayer"):
                        parent_lid = int(parent.GetLayer())
                        matched_lids = [int(copper_ids[i]) for i in indices]
                        if parent_lid in matched_lids:
                            return parent_lid
                except Exception:
                    pass
            return int(copper_ids[indices[0]])

    if hasattr(pad, "GetLayerSet"):
        try:
            cu_stack = list(pad.GetLayerSet().CuStack())
            if cu_stack:
                return int(cu_stack[0])
        except Exception:
            pass

    try:
        return int(pad.GetLayer())
    except Exception:
        return int(getattr(pcbnew, "F_Cu", 0))


def _pad_shape_center_mm(
    pad: Any,
    layer_id: Optional[int] = None,
    angle_rad: float = 0.0,
) -> Tuple[float, float]:
    """Return the copper shape center in millimeters, accounting for pad offset."""
    if hasattr(pad, "ShapePos"):
        shape_pos = None
        if layer_id is not None:
            try:
                shape_pos = pad.ShapePos(int(layer_id))
            except TypeError:
                try:
                    shape_pos = pad.ShapePos()
                except Exception:
                    shape_pos = None
            except Exception:
                shape_pos = None
        else:
            try:
                shape_pos = pad.ShapePos()
            except TypeError:
                try:
                    shape_pos = pad.ShapePos(get_pad_primary_layer_id(pad))
                except Exception:
                    shape_pos = None
            except Exception:
                shape_pos = None
        if shape_pos is not None and hasattr(shape_pos, "x") and hasattr(shape_pos, "y"):
            return float(shape_pos.x) * 1e-6, float(shape_pos.y) * 1e-6

    try:
        pos = pad.GetPosition()
        cx = float(pos.x) * 1e-6
        cy = float(pos.y) * 1e-6
        if hasattr(pad, "GetOffset"):
            try:
                offset = pad.GetOffset()
                ox = float(offset.x) * 1e-6
                oy = float(offset.y) * 1e-6
                if ox != 0.0 or oy != 0.0:
                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)
                    cx += ox * cos_a + oy * sin_a
                    cy += -ox * sin_a + oy * cos_a
            except Exception:
                pass
        return cx, cy
    except Exception:
        bbox = pad.GetBoundingBox()
        return (
            (float(bbox.GetX()) + 0.5 * float(bbox.GetWidth())) * 1e-6,
            (float(bbox.GetY()) + 0.5 * float(bbox.GetHeight())) * 1e-6,
        )


def _evaluate_pad_shape_mask(
    pad: Any,
    shape: Any,
    xx: np.ndarray,
    yy: np.ndarray,
    cx: float,
    cy: float,
    cos_a: float,
    sin_a: float,
    half_w: float,
    half_h: float,
    x0: float,
    y0: float,
    w_bb: float,
    h_bb: float,
    layer_id: Optional[int] = None,
) -> np.ndarray:
    """Evaluate a KiCad pad shape mask over cell centers ``(xx, yy)`` in mm."""
    eps = 1e-9
    circle_shape = getattr(pcbnew, "PAD_SHAPE_CIRCLE", object())
    oval_shape = getattr(pcbnew, "PAD_SHAPE_OVAL", object())
    rect_shape = getattr(pcbnew, "PAD_SHAPE_RECT", object())
    rect_alt_shape = getattr(pcbnew, "PAD_SHAPE_RECTANGLE", object())
    roundrect_shape = getattr(pcbnew, "PAD_SHAPE_ROUNDRECT", object())
    chamfered_shape = getattr(pcbnew, "PAD_SHAPE_CHAMFERED_RECT", object())

    dx = xx - cx
    dy = yy - cy

    if shape == circle_shape:
        radius = max(half_w, half_h) + eps
        return dx * dx + dy * dy <= radius * radius

    # KiCad uses +Y down and positive orientation CCW:
    # local_x = cos(a)*dx - sin(a)*dy, local_y = sin(a)*dx + cos(a)*dy
    local_x = cos_a * dx - sin_a * dy
    local_y = sin_a * dx + cos_a * dy

    if shape == oval_shape:
        if half_w >= half_h:
            segment_half = max(0.0, half_w - half_h)
            radius = half_h + eps
            nearest_x = np.clip(local_x, -segment_half, segment_half)
            return (local_x - nearest_x) ** 2 + local_y ** 2 <= radius * radius
        segment_half = max(0.0, half_h - half_w)
        radius = half_w + eps
        nearest_y = np.clip(local_y, -segment_half, segment_half)
        return local_x ** 2 + (local_y - nearest_y) ** 2 <= radius * radius

    if shape == roundrect_shape:
        try:
            corner_radius = float(pad.GetRoundRectCornerRadius()) * 1e-6
        except Exception:
            try:
                corner_radius = (
                    float(pad.GetRoundRectRadiusRatio())
                    * min(2.0 * half_w, 2.0 * half_h)
                )
            except Exception:
                corner_radius = 0.25 * min(2.0 * half_w, 2.0 * half_h)
        corner_radius = min(max(corner_radius, 0.0), half_w, half_h)
        qx = np.abs(local_x) - (half_w - corner_radius)
        qy = np.abs(local_y) - (half_h - corner_radius)
        outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return (outside + inside) <= corner_radius + eps

    if shape == rect_shape or shape == rect_alt_shape:
        return (np.abs(local_x) <= half_w + eps) & (np.abs(local_y) <= half_h + eps)

    if shape == chamfered_shape:
        inside = (np.abs(local_x) <= half_w + eps) & (np.abs(local_y) <= half_h + eps)
        try:
            chamfer_ratio = float(pad.GetChamferRectRatio())
            chamfer_pos = int(pad.GetChamferPositions())
        except Exception:
            chamfer_pos = 0
            chamfer_ratio = 0.0
        if chamfer_pos and chamfer_ratio > 0.0:
            c_dist = min(max(chamfer_ratio, 0.0), 0.5) * min(2.0 * half_w, 2.0 * half_h)
            limit = half_w + half_h - c_dist + eps
            if chamfer_pos & 1:
                inside &= ~((local_x < 0.0) & (local_y < 0.0) & ((-local_x - local_y) > limit))
            if chamfer_pos & 2:
                inside &= ~((local_x > 0.0) & (local_y < 0.0) & ((local_x - local_y) > limit))
            if chamfer_pos & 4:
                inside &= ~((local_x < 0.0) & (local_y > 0.0) & ((-local_x + local_y) > limit))
            if chamfer_pos & 8:
                inside &= ~((local_x > 0.0) & (local_y > 0.0) & ((local_x + local_y) > limit))
        return inside

    if hasattr(pad, "HitTest") and hasattr(pcbnew, "VECTOR2I"):
        out = np.zeros(xx.shape, dtype=bool)
        for r_idx in range(xx.shape[0]):
            for c_idx in range(xx.shape[1]):
                pos = pcbnew.VECTOR2I(
                    int(round(float(xx[r_idx, c_idx]) * 1e6)),
                    int(round(float(yy[r_idx, c_idx]) * 1e6)),
                )
                try:
                    if layer_id is not None:
                        try:
                            out[r_idx, c_idx] = bool(pad.HitTest(pos, 0, int(layer_id)))
                        except TypeError:
                            out[r_idx, c_idx] = bool(pad.HitTest(pos))
                    else:
                        out[r_idx, c_idx] = bool(pad.HitTest(pos))
                except Exception:
                    out[r_idx, c_idx] = False
        return out

    return (
        (xx >= x0 - eps)
        & (xx <= x0 + w_bb + eps)
        & (yy >= y0 - eps)
        & (yy <= y0 + h_bb + eps)
    )


def get_pad_grid_mask(
    pad: Any,
    rows: int,
    cols: int,
    x_min: float,
    y_min: float,
    res: float,
    layer_id: Optional[int] = None,
    area_mask: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, int, np.ndarray]:
    """
    Rasterize a KiCad pad to a tight local boolean grid slice.

    Returns ``(rs, re, cs, ce, mask_2d)`` where ``mask_2d`` has shape
    ``(re - rs, ce - cs)`` and matches the exact copper footprint of ``pad``
    (including rotation, offset, rounded/oval/chamfered/custom shapes, and a
    sub-pixel cell guarantee so tiny or narrow pads never vanish).
    """
    empty = np.zeros((0, 0), dtype=bool)
    if rows <= 0 or cols <= 0 or res <= 0.0:
        return 0, 0, 0, 0, empty

    bbox = pad.GetBoundingBox()
    x0 = float(bbox.GetX()) * 1e-6
    y0 = float(bbox.GetY()) * 1e-6
    w_bb = float(bbox.GetWidth()) * 1e-6
    h_bb = float(bbox.GetHeight()) * 1e-6

    grid_x_max = x_min + cols * res
    grid_y_max = y_min + rows * res
    if x0 + w_bb < x_min or x0 > grid_x_max or y0 + h_bb < y_min or y0 > grid_y_max:
        return 0, 0, 0, 0, empty

    cs = max(0, int(math.floor((x0 - x_min) / res)))
    rs = max(0, int(math.floor((y0 - y_min) / res)))
    ce = min(cols, int(math.floor((x0 + w_bb - x_min) / res)) + 1)
    re = min(rows, int(math.floor((y0 + h_bb - y_min) / res)) + 1)
    if cs >= ce or rs >= re:
        return 0, 0, 0, 0, empty

    x = x_min + (np.arange(cs, ce, dtype=np.float64) + 0.5) * res
    y = y_min + (np.arange(rs, re, dtype=np.float64) + 0.5) * res
    xx, yy = np.meshgrid(x, y)

    angle = _pad_orientation_radians(pad)
    cx, cy = _pad_shape_center_mm(pad, layer_id=layer_id, angle_rad=angle)

    try:
        size = pad.GetSize()
        width_mm = max(float(size.x) * 1e-6, 0.0)
        height_mm = max(float(size.y) * 1e-6, 0.0)
        shape = pad.GetShape()
        has_shape = True
    except Exception:
        has_shape = False
        width_mm = max(w_bb, 0.0)
        height_mm = max(h_bb, 0.0)
        shape = None

    if has_shape:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        half_w = 0.5 * width_mm
        half_h = 0.5 * height_mm
        mask_2d = np.asarray(
            _evaluate_pad_shape_mask(
                pad, shape, xx, yy, cx, cy, cos_a, sin_a,
                half_w, half_h, x0, y0, w_bb, h_bb, layer_id=layer_id,
            ),
            dtype=bool,
        )
        if area_mask is not None:
            mask_2d &= area_mask[rs:re, cs:ce]

        if not np.any(mask_2d):
            eff_half_w = max(half_w, 0.5 * res)
            eff_half_h = max(half_h, 0.5 * res)
            if eff_half_w > half_w or eff_half_h > half_h:
                mask_2d = np.asarray(
                    _evaluate_pad_shape_mask(
                        pad, shape, xx, yy, cx, cy, cos_a, sin_a,
                        eff_half_w, eff_half_h, x0, y0, w_bb, h_bb, layer_id=layer_id,
                    ),
                    dtype=bool,
                )
                if area_mask is not None:
                    mask_2d &= area_mask[rs:re, cs:ce]
    else:
        eps = 1e-9
        mask_2d = (
            (xx >= x0 - eps)
            & (xx <= x0 + w_bb + eps)
            & (yy >= y0 - eps)
            & (yy <= y0 + h_bb + eps)
        )
        if area_mask is not None:
            mask_2d &= area_mask[rs:re, cs:ce]

    if not np.any(mask_2d):
        c_c = min(cols - 1, max(0, int(math.floor((cx - x_min) / res))))
        r_c = min(rows - 1, max(0, int(math.floor((cy - y_min) / res))))
        if area_mask is None or bool(area_mask[r_c, c_c]):
            if r_c < rs or r_c >= re or c_c < cs or c_c >= ce:
                new_rs = min(rs, r_c)
                new_re = max(re, r_c + 1)
                new_cs = min(cs, c_c)
                new_ce = max(ce, c_c + 1)
                new_mask = np.zeros((new_re - new_rs, new_ce - new_cs), dtype=bool)
                new_mask[rs - new_rs:re - new_rs, cs - new_cs:ce - new_cs] = mask_2d
                rs, re, cs, ce, mask_2d = new_rs, new_re, new_cs, new_ce, new_mask
            mask_2d[r_c - rs, c_c - cs] = True

    return rs, re, cs, ce, mask_2d


def get_pad_power_indices(
    pad: Any,
    target_idx: int,
    rows: int,
    cols: int,
    x_min: float,
    y_min: float,
    res: float,
    layer_id: Optional[int] = None,
    area_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return flattened 1D int64 node indices matching ``get_pad_grid_mask``."""
    rs, re, cs, ce, mask_2d = get_pad_grid_mask(
        pad, rows, cols, x_min, y_min, res, layer_id=layer_id, area_mask=area_mask
    )
    if rs >= re or cs >= ce or not np.any(mask_2d):
        return np.empty(0, dtype=np.int64)
    local_r, local_c = np.nonzero(mask_2d)
    base = int(target_idx) * int(rows) * int(cols)
    return (base + (rs + local_r) * int(cols) + (cs + local_c)).astype(np.int64, copy=False)


def _state_fill_pad(state, layer_indices, pad, copper_ids=None):
    """Rasterize KiCad pad shapes onto ``state.copper_mask`` via ``get_pad_grid_mask``."""
    if not layer_indices:
        return
    if copper_ids is None:
        rs, re, cs, ce, mask_2d = get_pad_grid_mask(
            pad,
            state.rows,
            state.cols,
            state.x_min,
            state.y_min,
            state.res,
            layer_id=None,
            area_mask=state.area_mask,
        )
        if rs < re and cs < ce and np.any(mask_2d):
            for layer_idx in layer_indices:
                state.copper_mask[layer_idx, rs:re, cs:ce] |= mask_2d
        return

    for layer_idx in layer_indices:
        lid = copper_ids[layer_idx] if 0 <= layer_idx < len(copper_ids) else None
        rs, re, cs, ce, mask_2d = get_pad_grid_mask(
            pad,
            state.rows,
            state.cols,
            state.x_min,
            state.y_min,
            state.res,
            layer_id=lid,
            area_mask=state.area_mask,
        )
        if rs < re and cs < ce and np.any(mask_2d):
            state.copper_mask[layer_idx, rs:re, cs:ce] |= mask_2d


def _state_fill_box_all_layers(state, bbox):
    """Mark copper occupancy for all layers in a rectangular region."""
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, state)
    if cs >= ce or rs >= re or rs >= state.rows or cs >= state.cols:
        return

    target = state.copper_mask[:, rs:re, cs:ce]
    if state.area_mask is None:
        target[...] = True
        return

    region_mask = state.area_mask[rs:re, cs:ce]
    if np.any(region_mask):
        target |= region_mask[None, :, :]


def _state_fill_via(state, bbox, val):
    """Apply via enhancement in a rectangular region."""
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, state)
    if cs >= ce or rs >= re or rs >= state.rows or cs >= state.cols:
        return

    target = state.via_map[rs:re, cs:ce]
    if state.area_mask is None:
        np.maximum(target, val, out=target)
        return

    region_mask = state.area_mask[rs:re, cs:ce]
    if np.any(region_mask):
        np.maximum(target, val, out=target, where=region_mask)


def _extract_via_drill_mm(via: Any, outer_diameter_mm: float) -> float:
    """Extract finished drill diameter (mm) for a via across KiCad 9/10 and mocks."""
    for method_name in ("GetDrillValue", "GetDrill"):
        fn = getattr(via, method_name, None)
        if callable(fn):
            try:
                val = float(fn())
                if val > 0.0:
                    return val * 1e-6
            except Exception:
                pass
    return 0.5 * max(0.0, float(outer_diameter_mm))


def _extract_pth_drill_mm(pad: Any, outer_diameter_mm: float) -> Tuple[float, float]:
    """Extract PTH drill size ``(dx_mm, dy_mm)`` across KiCad 9/10 and mocks."""
    fn = getattr(pad, "GetDrillSize", None)
    if callable(fn):
        try:
            sz = fn()
            if hasattr(sz, "x") and hasattr(sz, "y"):
                dx = float(sz.x) * 1e-6
                dy = float(sz.y) * 1e-6
            else:
                dx = float(sz[0]) * 1e-6
                dy = float(sz[1]) * 1e-6
            if dx > 0.0 and dy > 0.0:
                return dx, dy
        except Exception:
            pass
    fallback = 0.5 * max(0.0, float(outer_diameter_mm))
    return fallback, fallback


def select_via_grid_cells(
    rows: int,
    cols: int,
    x_min: float,
    y_min: float,
    res: float,
    cx_mm: float,
    cy_mm: float,
    spread_radius_mm: float,
    area_mask: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, int, np.ndarray, int]:
    """Select grid pixels covered by a circular via/pad with nearest-cell fallback.

    Returns ``(rs, re, cs, ce, mask_2d, count)``. Returns ``count == 0`` when the
    via lies completely outside the simulated domain or active ``area_mask``.
    """
    empty = np.zeros((0, 0), dtype=bool)
    if rows <= 0 or cols <= 0 or res <= 0.0:
        return 0, 0, 0, 0, empty, 0

    radius = max(0.0, float(spread_radius_mm))
    grid_x_max = x_min + cols * res
    grid_y_max = y_min + rows * res
    if (
        cx_mm + radius < x_min
        or cx_mm - radius > grid_x_max
        or cy_mm + radius < y_min
        or cy_mm - radius > grid_y_max
    ):
        return 0, 0, 0, 0, empty, 0

    cs = max(0, int(math.floor((cx_mm - radius - x_min) / res)))
    rs = max(0, int(math.floor((cy_mm - radius - y_min) / res)))
    ce = min(cols, int(math.floor((cx_mm + radius - x_min) / res)) + 1)
    re = min(rows, int(math.floor((cy_mm + radius - y_min) / res)) + 1)

    if cs < ce and rs < re:
        y = y_min + (np.arange(rs, re, dtype=np.float64) + 0.5) * res
        x = x_min + (np.arange(cs, ce, dtype=np.float64) + 0.5) * res
        inside = (x[None, :] - cx_mm) ** 2 + (y[:, None] - cy_mm) ** 2 <= radius * radius + 1e-12
        if area_mask is not None:
            inside &= area_mask[rs:re, cs:ce]
        count = int(np.count_nonzero(inside))
        if count > 0:
            return rs, re, cs, ce, inside, count

    c_c = min(cols - 1, max(0, int(math.floor((cx_mm - x_min) / res))))
    r_c = min(rows - 1, max(0, int(math.floor((cy_mm - y_min) / res))))
    if area_mask is not None and not bool(area_mask[r_c, c_c]):
        return 0, 0, 0, 0, empty, 0
    return r_c, r_c + 1, c_c, c_c + 1, np.ones((1, 1), dtype=bool), 1


def _accumulate_via_conductance(
    state: GeometryState,
    cx_mm: float,
    cy_mm: float,
    spread_radius_mm: float,
    delta_ka: float,
    k_fr4: float,
    num_layers: int,
    gap_indices: Optional[List[int]] = None,
) -> None:
    """Conservatively accumulate physical via conductance into ``state.via_map`` (and ``via_map_3d``)."""
    if delta_ka <= 0.0 or k_fr4 <= 0.0 or state.res <= 0.0:
        return
    rs, re, cs, ce, inside, n_pix = select_via_grid_cells(
        state.rows,
        state.cols,
        state.x_min,
        state.y_min,
        state.res,
        cx_mm,
        cy_mm,
        spread_radius_mm,
        area_mask=state.area_mask,
    )
    if n_pix <= 0:
        return

    pixel_area_m2 = (float(state.res) * 1e-3) ** 2
    dv = float(delta_ka / (float(k_fr4) * pixel_area_m2 * n_pix))

    if gap_indices is None or num_layers <= 2:
        if state.via_map_3d is not None:
            for g in range(state.via_map_3d.shape[0]):
                target_g = state.via_map_3d[g, rs:re, cs:ce]
                target_g[inside] += dv
        else:
            target = state.via_map[rs:re, cs:ce]
            target[inside] += dv
        return

    if not gap_indices:
        return
    if state.via_map_3d is None:
        state.via_map_3d = np.broadcast_to(
            state.via_map[None, :, :],
            (num_layers - 1, state.rows, state.cols),
        ).copy()
    for g in gap_indices:
        if 0 <= g < state.via_map_3d.shape[0]:
            target_g = state.via_map_3d[g, rs:re, cs:ce]
            target_g[inside] += dv


def _state_fill_via_circle(state, bbox, center, diameter_mm, val, tile_size=512):
    """Apply via enhancement using the actual circular outer diameter."""
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, state)
    if cs >= ce or rs >= re:
        return
    cx = float(center.x) * 1e-6
    cy = float(center.y) * 1e-6
    radius = 0.5 * max(float(diameter_mm), 0.0)
    radius_sq = radius * radius
    for r0 in range(rs, re, tile_size):
        r1 = min(re, r0 + tile_size)
        y = state.y_min + (np.arange(r0, r1, dtype=np.float64) + 0.5) * state.res
        for c0 in range(cs, ce, tile_size):
            c1 = min(ce, c0 + tile_size)
            x = state.x_min + (np.arange(c0, c1, dtype=np.float64) + 0.5) * state.res
            inside = (x[None, :] - cx) ** 2 + (y[:, None] - cy) ** 2 <= radius_sq
            if state.area_mask is not None:
                inside &= state.area_mask[r0:r1, c0:c1]
            if np.any(inside):
                target = state.via_map[r0:r1, c0:c1]
                np.maximum(target, val, out=target, where=inside)


def _state_fill_heatsink(state, bbox):
    """Mark heatsink occupancy in a rectangular region."""
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, state)
    if cs >= ce or rs >= re or rs >= state.rows or cs >= state.cols:
        return

    target = state.heatsink_mask[rs:re, cs:ce]
    if state.area_mask is None:
        target[...] = True
        return

    region_mask = state.area_mask[rs:re, cs:ce]
    if np.any(region_mask):
        target |= region_mask


def _fill_zone_mask_hit_test_filled(
    mask,
    area_mask,
    x_vals,
    y_vals,
    lid,
    zone,
    progress_callback=None,
    cancel_check=None,
):
    """Populate a boolean mask using HitTestFilledArea."""
    hit_test = zone.HitTestFilledArea
    vector_ctor = pcbnew.VECTOR2I
    margin_iu = 1
    tracker = _ProgressEmitter(progress_callback, cancel_check)
    total_rows = max(1, len(y_vals))

    try:
        if area_mask is None:
            for r_idx, y_iu in enumerate(y_vals):
                tracker.emit(
                    r_idx,
                    total_rows,
                    f"Hit-testing zone row {r_idx + 1} / {total_rows}...",
                    force=(r_idx == 0),
                )
                row_mask = mask[r_idx]
                for c_idx, x_iu in enumerate(x_vals):
                    row_mask[c_idx] = bool(hit_test(lid, vector_ctor(int(x_iu), int(y_iu)), margin_iu))
        else:
            for r_idx, y_iu in enumerate(y_vals):
                tracker.emit(
                    r_idx,
                    total_rows,
                    f"Hit-testing zone row {r_idx + 1} / {total_rows}...",
                    force=(r_idx == 0),
                )
                allowed = area_mask[r_idx]
                if not np.any(allowed):
                    continue
                row_mask = mask[r_idx]
                for c_idx, x_iu in enumerate(x_vals):
                    if allowed[c_idx]:
                        row_mask[c_idx] = bool(hit_test(lid, vector_ctor(int(x_iu), int(y_iu)), margin_iu))
    except GeometryCancelledError:
        raise
    except Exception:
        for r_idx, y_iu in enumerate(y_vals):
            tracker.emit(
                r_idx,
                total_rows,
                f"Hit-testing zone row {r_idx + 1} / {total_rows}...",
                force=(r_idx == 0),
            )
            try:
                row_mask = mask[r_idx]
                if area_mask is None:
                    for c_idx, x_iu in enumerate(x_vals):
                        row_mask[c_idx] = bool(hit_test(lid, vector_ctor(int(x_iu), int(y_iu)), margin_iu))
                else:
                    allowed = area_mask[r_idx]
                    if not np.any(allowed):
                        continue
                    for c_idx, x_iu in enumerate(x_vals):
                        if allowed[c_idx]:
                            row_mask[c_idx] = bool(hit_test(lid, vector_ctor(int(x_iu), int(y_iu)), margin_iu))
            except GeometryCancelledError:
                raise
            except Exception:
                continue


def _fill_zone_mask_hit_test(
    mask,
    area_mask,
    x_vals,
    y_vals,
    zone,
    progress_callback=None,
    cancel_check=None,
):
    """Populate a boolean mask using generic HitTest."""
    hit_test = zone.HitTest
    vector_ctor = pcbnew.VECTOR2I
    tracker = _ProgressEmitter(progress_callback, cancel_check)
    total_rows = max(1, len(y_vals))

    try:
        if area_mask is None:
            for r_idx, y_iu in enumerate(y_vals):
                tracker.emit(
                    r_idx,
                    total_rows,
                    f"Hit-testing zone row {r_idx + 1} / {total_rows}...",
                    force=(r_idx == 0),
                )
                row_mask = mask[r_idx]
                for c_idx, x_iu in enumerate(x_vals):
                    row_mask[c_idx] = bool(hit_test(vector_ctor(int(x_iu), int(y_iu))))
        else:
            for r_idx, y_iu in enumerate(y_vals):
                tracker.emit(
                    r_idx,
                    total_rows,
                    f"Hit-testing zone row {r_idx + 1} / {total_rows}...",
                    force=(r_idx == 0),
                )
                allowed = area_mask[r_idx]
                if not np.any(allowed):
                    continue
                row_mask = mask[r_idx]
                for c_idx, x_iu in enumerate(x_vals):
                    if allowed[c_idx]:
                        row_mask[c_idx] = bool(hit_test(vector_ctor(int(x_iu), int(y_iu))))
    except GeometryCancelledError:
        raise
    except Exception:
        for r_idx, y_iu in enumerate(y_vals):
            tracker.emit(
                r_idx,
                total_rows,
                f"Hit-testing zone row {r_idx + 1} / {total_rows}...",
                force=(r_idx == 0),
            )
            try:
                row_mask = mask[r_idx]
                if area_mask is None:
                    for c_idx, x_iu in enumerate(x_vals):
                        row_mask[c_idx] = bool(hit_test(vector_ctor(int(x_iu), int(y_iu))))
                else:
                    allowed = area_mask[r_idx]
                    if not np.any(allowed):
                        continue
                    for c_idx, x_iu in enumerate(x_vals):
                        if allowed[c_idx]:
                            row_mask[c_idx] = bool(hit_test(vector_ctor(int(x_iu), int(y_iu))))
            except GeometryCancelledError:
                raise
            except Exception:
                continue


def _scanline_fill_polygon(target, x_vals, y_vals, vertices, value=True, row_offset=0):
    """Fill one polygon chain using a bounding-box-sliced even/odd scanline rule."""
    if len(vertices) < 3 or len(x_vals) == 0 or len(y_vals) == 0:
        return

    verts = np.asarray(vertices, dtype=np.float64)
    x_min = float(np.min(verts[:, 0]))
    x_max = float(np.max(verts[:, 0]))
    y_min = float(np.min(verts[:, 1]))
    y_max = float(np.max(verts[:, 1]))

    if (
        x_max < float(x_vals[0])
        or x_min > float(x_vals[-1])
        or y_max < float(y_vals[0])
        or y_min > float(y_vals[-1])
    ):
        return

    r_start = max(int(row_offset), int(np.searchsorted(y_vals, y_min, side="left")))
    r_end = min(
        int(row_offset) + int(target.shape[0]),
        int(np.searchsorted(y_vals, y_max, side="right")),
    )
    if r_start >= r_end:
        return

    x0 = verts[:, 0]
    y0 = verts[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    non_horizontal = y0 != y1
    if not np.any(non_horizontal):
        return
    x0 = x0[non_horizontal]
    y0 = y0[non_horizontal]
    x1 = x1[non_horizontal]
    y1 = y1[non_horizontal]

    e_ymin = np.minimum(y0, y1)
    e_ymax = np.maximum(y0, y1)
    dx = x1 - x0
    dy = y1 - y0

    def _fill_row_range(row_from, row_to, rx0, ry0, rymin, rymax, rdx, rdy):
        for row_idx in range(row_from, row_to):
            y_val = float(y_vals[row_idx])
            crosses = (rymin <= y_val) & (y_val < rymax)
            if not np.any(crosses):
                continue
            intersections = rx0[crosses] + (
                (y_val - ry0[crosses]) * rdx[crosses] / rdy[crosses]
            )
            intersections.sort()
            pair_count = intersections.size // 2
            if pair_count <= 0:
                continue
            if intersections.size != pair_count * 2:
                intersections = intersections[: pair_count * 2]
            starts = np.searchsorted(x_vals, intersections[0::2], side="left")
            ends = np.searchsorted(x_vals, intersections[1::2], side="right")
            row = target[row_idx - row_offset]
            for start, end in zip(starts, ends):
                s_i = int(start)
                e_i = int(end)
                if s_i < e_i:
                    row[s_i:e_i] = value

    n_edges = x0.size
    n_active_rows = r_end - r_start
    if n_edges > 64 and n_active_rows > 16:
        order = np.argsort(e_ymin, kind="mergesort")
        x0 = x0[order]
        y0 = y0[order]
        e_ymin = e_ymin[order]
        e_ymax = e_ymax[order]
        dx = dx[order]
        dy = dy[order]
        strip_size = 32
        for s_start in range(r_start, r_end, strip_size):
            s_end = min(r_end, s_start + strip_size)
            strip_ymin = float(y_vals[s_start])
            strip_ymax = float(y_vals[s_end - 1])
            hi = int(np.searchsorted(e_ymin, strip_ymax, side="right"))
            if hi <= 0:
                continue
            active = e_ymax[:hi] > strip_ymin
            if not np.any(active):
                continue
            _fill_row_range(
                s_start,
                s_end,
                x0[:hi][active],
                y0[:hi][active],
                e_ymin[:hi][active],
                e_ymax[:hi][active],
                dx[:hi][active],
                dy[:hi][active],
            )
    else:
        _fill_row_range(r_start, r_end, x0, y0, e_ymin, e_ymax, dx, dy)


def _fill_zone_mask_polygons(
    mask,
    area_mask,
    x_vals,
    y_vals,
    lid,
    zone,
    progress_callback=None,
    cancel_check=None,
):
    """Rasterize filled zone polygons using local-bbox scanline fill."""
    if not hasattr(zone, "GetFilledPolysList"):
        return False
    tracker = _ProgressEmitter(progress_callback, cancel_check)
    try:
        if hasattr(zone, "IsFilled") and not zone.IsFilled():
            return False
        if hasattr(zone, "IsOnLayer") and not zone.IsOnLayer(lid):
            return False
        poly_set = zone.GetFilledPolysList(lid)
        outline_count = int(poly_set.OutlineCount())
        if outline_count <= 0:
            return False
        if len(x_vals) == 0 or len(y_vals) == 0:
            mask[...] = False
            return True

        filled = np.zeros(mask.shape, dtype=bool)
        x_first = float(x_vals[0])
        x_last = float(x_vals[-1])
        y_first = float(y_vals[0])
        y_last = float(y_vals[-1])

        def chain_vertices(chain):
            count = int(chain.PointCount())
            if count <= 0:
                return np.empty((0, 2), dtype=np.float64)
            cpoint = chain.CPoint
            pts = [None] * count
            for idx in range(count):
                point = cpoint(idx)
                pts[idx] = (point.x, point.y)
            return np.asarray(pts, dtype=np.float64)

        for outline_idx in range(outline_count):
            tracker.emit(
                outline_idx,
                outline_count,
                f"Rasterizing zone outline {outline_idx + 1} / {outline_count}...",
                force=(outline_idx == 0),
            )
            outer = chain_vertices(poly_set.Outline(outline_idx))
            if len(outer) < 3:
                continue

            ox_min = float(np.min(outer[:, 0]))
            ox_max = float(np.max(outer[:, 0]))
            oy_min = float(np.min(outer[:, 1]))
            oy_max = float(np.max(outer[:, 1]))
            if ox_max < x_first or ox_min > x_last or oy_max < y_first or oy_min > y_last:
                continue

            rs_o = int(np.searchsorted(y_vals, oy_min, side="left"))
            re_o = int(np.searchsorted(y_vals, oy_max, side="right"))
            cs_o = int(np.searchsorted(x_vals, ox_min, side="left"))
            ce_o = int(np.searchsorted(x_vals, ox_max, side="right"))
            if rs_o >= re_o or cs_o >= ce_o:
                continue
            if area_mask is not None and not np.any(area_mask[rs_o:re_o, cs_o:ce_o]):
                continue

            x_local = x_vals[cs_o:ce_o]
            y_local = y_vals[rs_o:re_o]
            inside = np.zeros((re_o - rs_o, ce_o - cs_o), dtype=bool)
            _scanline_fill_polygon(inside, x_local, y_local, outer, True)
            if not np.any(inside):
                continue

            try:
                hole_count = int(poly_set.HoleCount(outline_idx))
            except Exception:
                hole_count = 0
            for hole_idx in range(hole_count):
                if tracker.active:
                    tracker.emit(
                        hole_idx + 1 if outline_count == 1 else outline_idx + 1,
                        hole_count if outline_count == 1 else outline_count,
                        f"Rasterizing zone outline {outline_idx + 1} / {outline_count} "
                        f"(hole {hole_idx + 1} / {hole_count})...",
                    )
                hole = chain_vertices(poly_set.Hole(outline_idx, hole_idx))
                if len(hole) >= 3:
                    _scanline_fill_polygon(inside, x_local, y_local, hole, False)
            filled[rs_o:re_o, cs_o:ce_o] |= inside

        if area_mask is not None:
            filled &= area_mask
        mask[...] = filled
        return True
    except GeometryCancelledError:
        raise
    except Exception:
        return False


def _state_fill_zone(
    state,
    l_idx,
    lid,
    zone,
    progress_callback=None,
    cancel_check=None,
):
    """
    Mark copper occupancy for a zone using KiCad polygon scanline or hit testing.

    This keeps the current authoritative geometry path while reducing
    Python overhead in the inner loops.
    """
    bbox = zone.GetBoundingBox()
    rs, re, cs, ce = _bbox_to_grid_indices(bbox, state)
    if cs >= ce or rs >= re:
        return

    zone_mask = np.zeros((re - rs, ce - cs), dtype=bool)
    area_mask = state.area_mask[rs:re, cs:ce] if state.area_mask is not None else None
    x_vals = state.x_centers_iu[cs:ce]
    y_vals = state.y_centers_iu[rs:re]

    if _fill_zone_mask_polygons(
        zone_mask,
        area_mask,
        x_vals,
        y_vals,
        lid,
        zone,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    ):
        pass
    elif hasattr(zone, "HitTestFilledArea"):
        _fill_zone_mask_hit_test_filled(
            zone_mask,
            area_mask,
            x_vals,
            y_vals,
            lid,
            zone,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
    elif hasattr(zone, "HitTest"):
        _fill_zone_mask_hit_test(
            zone_mask,
            area_mask,
            x_vals,
            y_vals,
            zone,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
    else:
        return

    if np.any(zone_mask):
        state.copper_mask[l_idx, rs:re, cs:ce] |= zone_mask


def build_geometry_state(
    board,
    copper_ids,
    rows,
    cols,
    x_min,
    y_min,
    res,
    settings,
    via_factor=1300.0,
    pads_list=None,
    progress_callback=None,
    cancel_check=None,
    via_plating_um=None,
    via_fill=None,
    k_cu=390.0,
    k_fr4=0.3,
):
    """
    Build the internal geometry representation used by the solver.

    Parameters
    ----------
    board : pcbnew.BOARD
        The KiCad board object.
    copper_ids : list of int
        Layer IDs of copper layers in stackup order.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    x_min : float
        X coordinate of grid origin in mm.
    y_min : float
        Y coordinate of grid origin in mm.
    res : float
        Grid resolution in mm.
    settings : dict
        Simulation settings from the dialog.
    via_factor : float, optional
        Legacy via enhancement factor kept for signature compatibility.
    pads_list : list, optional
        List of selected pad objects (heat sources).
    via_plating_um : float, optional
        Via barrel copper plating wall thickness in micrometers (default 25.0).
    via_fill : str, optional
        Via fill type (``'plated'`` or ``'copper'``, default ``'plated'``).
    k_cu : float, optional
        Copper thermal conductivity in W/(m*K) (default 390.0).
    k_fr4 : float, optional
        Dielectric thermal conductivity in W/(m*K) (default 0.3).

    Returns
    -------
    GeometryState
        Internal geometry state with boolean copper occupancy and masks.

    Raises
    ------
    RuntimeError
        If KiCad geometry cannot be mapped completely and safely.
    """
    _ = via_factor
    pads_list = list(pads_list or [])
    if via_plating_um is None:
        raw_plating = settings.get("via_plating_um", 25.0)
        resolved_plating_um = max(0.0, float(25.0 if raw_plating is None else raw_plating))
    else:
        resolved_plating_um = max(0.0, float(via_plating_um))
    resolved_via_fill = normalize_via_fill(
        settings.get("via_fill", "plated") if via_fill is None else via_fill
    )
    k_cu_val = 390.0 if k_cu is None else float(k_cu)
    k_fr4_val = 0.3 if k_fr4 is None else float(k_fr4)

    num_layers = len(copper_ids)
    limit_area = settings.get('limit_area', False)
    radius_mm = settings.get('pad_dist_mm', 0.0) if limit_area else 0.0
    # New area-aware settings crop the rectangular solver domain itself.  The
    # legacy circular pad mask is retained only for settings files that do not
    # yet contain ``area_mode``; applying it to a current path could remove
    # valid copper without reducing the number of solver nodes.
    area_mask = None
    if 'area_mode' not in settings:
        area_mask = build_pad_distance_mask(
            pads_list, rows, cols, x_min, y_min, res, radius_mm
        )

    state = GeometryState(
        copper_mask=np.zeros((num_layers, rows, cols), dtype=bool),
        via_map=np.ones((rows, cols), dtype=np.float32),
        heatsink_mask=np.zeros((rows, cols), dtype=bool),
        area_mask=area_mask,
        x_min=x_min,
        y_min=y_min,
        res=res,
        rows=rows,
        cols=cols,
        x_centers_iu=_grid_centers_to_iu(cols, x_min, res),
        y_centers_iu=_grid_centers_to_iu(rows, y_min, res),
    )

    pad_net_codes: Set[int] = set()
    pad_net_names: Set[str] = set()
    if settings.get('ignore_polygons'):
        for pad in pads_list:
            try:
                pad_net_codes.add(pad.GetNetCode())
            except Exception:
                continue
            try:
                pad_net_names.add(pad.GetNetname())
            except Exception:
                try:
                    pad_net_names.add(pad.GetNet().GetNetname())
                except Exception:
                    pass
        pad_net_codes = {code for code in pad_net_codes if code is not None}
        pad_net_names = {name for name in pad_net_names if name}

    lid_to_idx = {lid: i for i, lid in enumerate(copper_ids)}
    ignore_traces = settings.get('ignore_traces')
    ignore_polygons = settings.get('ignore_polygons')
    use_heatsink = settings.get('use_heatsink')
    tracker = _ProgressEmitter(progress_callback, cancel_check)
    tracker.check_cancel(force=True)

    try:
        tracks = list(board.Tracks() if hasattr(board, 'Tracks') else board.GetTracks())
        footprints = list(board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints())
        zones = list(board.Zones() if hasattr(board, 'Zones') else board.GetZones())
        drawings = list(board.GetDrawings()) if use_heatsink and hasattr(board, 'GetDrawings') else []

        total_tracks = len(tracks)
        total_fps = len(footprints)
        total_zones = len(zones)
        total_drawings = len(drawings)
        total_items = max(1, total_tracks + total_fps + total_zones + total_drawings)
        completed = 0

        for idx, track in enumerate(tracks):
            tracker.emit(
                completed + idx,
                total_items,
                f"Mapping tracks & vias ({idx + 1} / {total_tracks})...",
                force=(idx == 0),
            )
            is_via = "VIA" in str(type(track)).upper()
            if ignore_traces and not is_via:
                continue

            lid = track.GetLayer()
            layer_idx = lid_to_idx.get(lid)
            bbox = track.GetBoundingBox()
            if is_via:
                via_layers = []
                try:
                    layer_set = track.GetLayerSet()
                    via_layers = [
                        idx for idx, copper_lid in enumerate(copper_ids)
                        if layer_set.Contains(copper_lid)
                    ]
                except Exception:
                    pass
                if not via_layers:
                    via_layers = list(range(num_layers))
                via_layer_id = (
                    copper_ids[via_layers[0]]
                    if via_layers and copper_ids
                    else None
                )
                try:
                    width_iu = None
                    if via_layer_id is not None:
                        try:
                            width_iu = track.GetWidth(int(via_layer_id))
                        except TypeError:
                            width_iu = track.GetWidth()
                    else:
                        width_iu = track.GetWidth()
                    diameter_mm = float(width_iu) * 1e-6
                    center = track.GetPosition()
                except Exception:
                    diameter_mm = max(
                        float(bbox.GetWidth()), float(bbox.GetHeight())
                    ) * 1e-6
                    center = pcbnew.VECTOR2I(
                        bbox.GetX() + bbox.GetWidth() // 2,
                        bbox.GetY() + bbox.GetHeight() // 2,
                    )
                _state_fill_circle(state, via_layers, bbox, center, diameter_mm)
                drill_mm = _extract_via_drill_mm(track, diameter_mm)
                delta_ka = compute_via_excess_ka(
                    drill_mm * 1e-3,
                    plating_um=resolved_plating_um,
                    via_fill=resolved_via_fill,
                    k_cu=k_cu_val,
                    k_fr4=k_fr4_val,
                )
                spread_radius_mm = max(
                    0.5 * max(diameter_mm, 0.0),
                    0.5 * drill_mm + resolved_plating_um * 1e-3,
                )
                gap_indices = None
                if num_layers > 2 and via_layers:
                    min_l = min(via_layers)
                    max_l = max(via_layers)
                    if min_l > 0 or max_l < num_layers - 1:
                        gap_indices = list(range(min_l, max_l))
                _accumulate_via_conductance(
                    state,
                    float(center.x) * 1e-6,
                    float(center.y) * 1e-6,
                    spread_radius_mm,
                    delta_ka,
                    k_fr4_val,
                    num_layers,
                    gap_indices=gap_indices,
                )
            elif layer_idx is not None:
                if hasattr(track, "GetMid"):
                    _state_fill_arc(state, layer_idx, track)
                else:
                    _state_fill_segment(state, layer_idx, track)
        completed += total_tracks

        for fp_idx, fp in enumerate(footprints):
            tracker.emit(
                completed + fp_idx,
                total_items,
                f"Mapping footprint pads ({fp_idx + 1} / {total_fps})...",
                force=(fp_idx == 0),
            )
            for pad in fp.Pads():
                pad_layers = get_pad_copper_layer_indices(pad, copper_ids, lid_to_idx)
                if pad_layers:
                    _state_fill_pad(state, pad_layers, pad, copper_ids=copper_ids)
                if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                    bbox = pad.GetBoundingBox()
                    try:
                        size = pad.GetSize()
                        diameter_mm = min(float(size.x), float(size.y)) * 1e-6
                        cx_mm, cy_mm = _pad_shape_center_mm(pad)
                    except Exception:
                        diameter_mm = min(
                            float(bbox.GetWidth()), float(bbox.GetHeight())
                        ) * 1e-6
                        cx_mm = (float(bbox.GetX()) + 0.5 * float(bbox.GetWidth())) * 1e-6
                        cy_mm = (float(bbox.GetY()) + 0.5 * float(bbox.GetHeight())) * 1e-6
                    dx_mm, dy_mm = _extract_pth_drill_mm(pad, diameter_mm)
                    delta_ka = compute_via_excess_ka(
                        dx_mm * 1e-3,
                        dy_mm * 1e-3,
                        plating_um=resolved_plating_um,
                        via_fill=resolved_via_fill,
                        k_cu=k_cu_val,
                        k_fr4=k_fr4_val,
                    )
                    spread_radius_mm = max(
                        0.5 * max(diameter_mm, 0.0),
                        0.5 * max(dx_mm, dy_mm) + resolved_plating_um * 1e-3,
                    )
                    _accumulate_via_conductance(
                        state,
                        cx_mm,
                        cy_mm,
                        spread_radius_mm,
                        delta_ka,
                        k_fr4_val,
                        num_layers,
                        gap_indices=None,
                    )
        completed += total_fps

        for zone_idx, zone in enumerate(zones):
            tracker.emit(
                completed + zone_idx,
                total_items,
                f"Rasterizing copper zone {zone_idx + 1} / {total_zones}...",
                force=(zone_idx == 0),
            )
            if hasattr(zone, "IsFilled") and not zone.IsFilled():
                continue

            if ignore_polygons:
                zone_net_name = None
                zone_net_code = None
                try:
                    zone_net_name = zone.GetNetname()
                except Exception:
                    try:
                        zone_net_name = zone.GetNet().GetNetname()
                    except Exception:
                        zone_net_name = None
                try:
                    zone_net_code = zone.GetNetCode()
                except Exception:
                    zone_net_code = None
                if pad_net_names:
                    if zone_net_name not in pad_net_names:
                        continue
                elif pad_net_codes and zone_net_code not in pad_net_codes:
                    continue

            zone_layers = []
            if hasattr(zone, "IsOnLayer"):
                for lid in copper_ids:
                    try:
                        if zone.IsOnLayer(lid):
                            zone_layers.append(lid)
                    except Exception:
                        continue
            if not zone_layers:
                try:
                    zone_layers = list(zone.GetLayerSet().IntSeq())
                except Exception:
                    zone_layers = []
            if not zone_layers:
                try:
                    zone_layers = [zone.GetLayer()]
                except Exception:
                    zone_layers = []

            zone_progress_cb = None
            if progress_callback is not None:
                def _zone_sub_progress(
                    _sub_done,
                    _sub_total,
                    sub_msg,
                    _z_idx=zone_idx,
                    _z_base=completed + zone_idx,
                ):
                    tracker.emit(
                        _z_base,
                        total_items,
                        f"Rasterizing copper zone {_z_idx + 1} / {total_zones}: {sub_msg}",
                        force=True,
                    )
                zone_progress_cb = _zone_sub_progress

            for lid in zone_layers:
                layer_idx = lid_to_idx.get(lid)
                if layer_idx is not None:
                    _state_fill_zone(
                        state,
                        layer_idx,
                        lid,
                        zone,
                        progress_callback=zone_progress_cb,
                        cancel_check=cancel_check,
                    )

            if use_heatsink:
                try:
                    if zone.GetLayerSet().Contains(pcbnew.Eco1_User):
                        _state_fill_heatsink(state, zone.GetBoundingBox())
                except Exception:
                    pass
        completed += total_zones

        if use_heatsink:
            for d_idx, drawing in enumerate(drawings):
                tracker.emit(
                    completed + d_idx,
                    total_items,
                    "Mapping thermal pad regions...",
                    force=(d_idx == 0),
                )
                try:
                    if drawing.GetLayer() == pcbnew.Eco1_User:
                        _state_fill_heatsink(state, drawing.GetBoundingBox())
                except Exception:
                    continue

        if state.via_map_3d is not None:
            state.via_map = np.max(state.via_map_3d, axis=0).astype(
                np.float32, copy=False
            )

        tracker.emit(total_items, total_items, "Geometry mapping complete.", force=True)

    except GeometryCancelledError:
        raise
    except Exception as exc:
        print(f"[ThermalSim][WARN] Geometry mapping error: {exc}")
        raise RuntimeError(
            "PCB geometry mapping failed; refusing to use a partial map."
        ) from exc

    return state


def build_pad_distance_mask(pads_list, rows, cols, x_min, y_min, res, radius_mm):
    """
    Build a boolean mask limiting simulation to area near pads.

    Parameters
    ----------
    pads_list : list
        List of pcbnew.PAD objects.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    x_min : float
        X coordinate of grid origin in mm.
    y_min : float
        Y coordinate of grid origin in mm.
    res : float
        Grid resolution in mm.
    radius_mm : float
        Maximum distance from pads to include.

    Returns
    -------
    np.ndarray or None
        Boolean mask of shape (rows, cols), or None if no limiting needed.
    """
    if not pads_list:
        return None
    if radius_mm is None or radius_mm <= 0:
        return None

    mask = np.zeros((rows, cols), dtype=bool)
    r_cells = int(math.ceil(radius_mm / res))
    radius_sq = radius_mm * radius_mm

    for pad in pads_list:
        try:
            pos = pad.GetPosition()
        except Exception:
            continue
        x_mm = pos.x * 1e-6
        y_mm = pos.y * 1e-6
        c0 = int((x_mm - x_min) / res)
        r0 = int((y_mm - y_min) / res)
        rs = max(0, r0 - r_cells)
        re = min(rows, r0 + r_cells + 1)
        cs = max(0, c0 - r_cells)
        ce = min(cols, c0 + r_cells + 1)
        if rs >= re or cs >= ce:
            continue
        ys = (np.arange(rs, re) - r0) * res
        xs = (np.arange(cs, ce) - c0) * res
        dist_sq = ys[:, None] * ys[:, None] + xs[None, :] * xs[None, :]
        mask[rs:re, cs:ce] |= dist_sq <= radius_sq

    return mask


def get_pad_pixels(pad, rows, cols, x_min, y_min, res, layer_id=None):
    """
    Get grid pixel coordinates covered by a pad.

    Parameters
    ----------
    pad : pcbnew.PAD
        The pad object.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    x_min : float
        X coordinate of grid origin in mm.
    y_min : float
        Y coordinate of grid origin in mm.
    res : float
        Grid resolution in mm.
    layer_id : int, optional
        Optional KiCad layer ID for ``ShapePos`` / ``HitTest``.

    Returns
    -------
    list of tuple
        List of (row, col) tuples for pixels covered by the pad.
    """
    rs, re, cs, ce, mask_2d = get_pad_grid_mask(
        pad, rows, cols, x_min, y_min, res, layer_id=layer_id
    )
    if rs >= re or cs >= ce or not np.any(mask_2d):
        return []
    local_r, local_c = np.nonzero(mask_2d)
    return [(int(rs + r), int(cs + c)) for r, c in zip(local_r, local_c)]


def create_multilayer_maps(
    board,
    copper_ids,
    rows,
    cols,
    x_min,
    y_min,
    res,
    settings,
    k_fr4,
    k_cu_layers,
    via_factor=1300.0,
    pads_list=None,
    via_plating_um=None,
    via_fill=None,
    k_cu=390.0,
    via_k_fr4=0.3,
):
    """
    Create thermal conductivity maps from PCB geometry.

    This function extracts copper geometry from the PCB and creates
    discretized arrays for thermal simulation.

    Parameters
    ----------
    board : pcbnew.BOARD
        The KiCad board object.
    copper_ids : list of int
        Layer IDs of copper layers in stackup order.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    x_min : float
        X coordinate of grid origin in mm.
    y_min : float
        Y coordinate of grid origin in mm.
    res : float
        Grid resolution in mm.
    settings : dict
        Simulation settings from the dialog.
    k_fr4 : float
        Relative thermal conductivity of FR4 (typically 1.0).
    k_cu_layers : list of float
        Relative conductivity for each copper layer.
    via_factor : float
        Enhancement factor for via thermal conductivity.
    pads_list : list
        List of selected pad objects (heat sources).

    Returns
    -------
    tuple
        (K, V_map, H_map) where:
        - K : np.ndarray, shape (layers, rows, cols)
            Relative thermal conductivity map.
        - V_map : np.ndarray, shape (rows, cols)
            Via enhancement factors for vertical coupling.
        - H_map : np.ndarray, shape (rows, cols)
            Heatsink/thermal-pad mask (1.0 where present).

    Notes
    -----
    The function processes:
    - Tracks and traces (unless ignore_traces is set)
    - Vias (PTH and via objects)
    - Footprint pads (SMD and PTH)
    - Copper zones with filled area hit-testing
    - User.Eco1 layer for thermal pad definition
    """
    state = build_geometry_state(
        board=board,
        copper_ids=copper_ids,
        rows=rows,
        cols=cols,
        x_min=x_min,
        y_min=y_min,
        res=res,
        settings=settings,
        via_factor=via_factor,
        pads_list=pads_list,
        via_plating_um=via_plating_um,
        via_fill=via_fill,
        k_cu=k_cu,
        k_fr4=via_k_fr4,
    )

    num_layers = len(copper_ids)
    K = np.full((num_layers, rows, cols), k_fr4, dtype=np.float64)
    for layer_idx, layer_value in enumerate(k_cu_layers):
        mask = state.copper_mask[layer_idx]
        if np.any(mask):
            K[layer_idx, mask] = float(layer_value)

    V = np.asarray(state.via_map, dtype=np.float64)
    H = state.heatsink_mask.astype(np.float64, copy=False)
    return K, V, H
