"""
PCB geometry to thermal grid mapping.

This module converts KiCad PCB geometry (copper, vias, zones) into
discretized conductivity arrays for thermal simulation.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Set

import numpy as np
import pcbnew


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
    """Rasterize a circular pad or via on one or more layers."""
    cx = float(center.x) * 1e-6
    cy = float(center.y) * 1e-6
    radius = 0.5 * max(float(diameter_mm), 0.0)

    def predicate(xx, yy):
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius

    for layer_idx in layer_indices:
        _state_apply_shape(state, layer_idx, bbox, predicate)


def _pad_orientation_radians(pad):
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


def _state_fill_pad(state, layer_indices, pad):
    """Rasterize common KiCad pad shapes, falling back safely for custom pads."""
    bbox = pad.GetBoundingBox()
    try:
        center = pad.GetPosition()
        size = pad.GetSize()
        width_mm = max(float(size.x) * 1e-6, state.res)
        height_mm = max(float(size.y) * 1e-6, state.res)
        shape = pad.GetShape()
    except Exception:
        for layer_idx in layer_indices:
            _state_fill_box(state, layer_idx, bbox)
        return

    circle_shape = getattr(pcbnew, "PAD_SHAPE_CIRCLE", object())
    oval_shape = getattr(pcbnew, "PAD_SHAPE_OVAL", object())
    rect_shape = getattr(pcbnew, "PAD_SHAPE_RECT", object())
    roundrect_shape = getattr(pcbnew, "PAD_SHAPE_ROUNDRECT", object())
    if shape == circle_shape:
        _state_fill_circle(state, layer_indices, bbox, center, max(width_mm, height_mm))
        return

    cx = float(center.x) * 1e-6
    cy = float(center.y) * 1e-6
    angle = _pad_orientation_radians(pad)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    half_w = 0.5 * width_mm
    half_h = 0.5 * height_mm

    def local_coordinates(xx, yy):
        dx = xx - cx
        dy = yy - cy
        return cos_a * dx + sin_a * dy, -sin_a * dx + cos_a * dy

    if shape == oval_shape:
        if width_mm >= height_mm:
            segment_half = max(0.0, 0.5 * (width_mm - height_mm))
            radius = half_h

            def predicate(xx, yy):
                local_x, local_y = local_coordinates(xx, yy)
                nearest_x = np.clip(local_x, -segment_half, segment_half)
                return (local_x - nearest_x) ** 2 + local_y ** 2 <= radius ** 2
        else:
            segment_half = max(0.0, 0.5 * (height_mm - width_mm))
            radius = half_w

            def predicate(xx, yy):
                local_x, local_y = local_coordinates(xx, yy)
                nearest_y = np.clip(local_y, -segment_half, segment_half)
                return local_x ** 2 + (local_y - nearest_y) ** 2 <= radius ** 2
    elif shape == roundrect_shape:
        try:
            corner_radius = float(pad.GetRoundRectCornerRadius()) * 1e-6
        except Exception:
            try:
                corner_radius = (
                    float(pad.GetRoundRectRadiusRatio())
                    * min(width_mm, height_mm)
                )
            except Exception:
                corner_radius = 0.25 * min(width_mm, height_mm)
        corner_radius = min(max(corner_radius, 0.0), half_w, half_h)

        def predicate(xx, yy):
            local_x, local_y = local_coordinates(xx, yy)
            qx = np.abs(local_x) - (half_w - corner_radius)
            qy = np.abs(local_y) - (half_h - corner_radius)
            outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
            inside = np.minimum(np.maximum(qx, qy), 0.0)
            return outside + inside <= corner_radius
    elif shape == rect_shape:
        def predicate(xx, yy):
            local_x, local_y = local_coordinates(xx, yy)
            return (np.abs(local_x) <= half_w) & (np.abs(local_y) <= half_h)
    else:
        for layer_idx in layer_indices:
            _state_fill_box(state, layer_idx, bbox)
        return

    for layer_idx in layer_indices:
        _state_apply_shape(state, layer_idx, bbox, predicate)


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


def _fill_zone_mask_hit_test_filled(mask, area_mask, x_vals, y_vals, lid, zone):
    """Populate a boolean mask using HitTestFilledArea."""
    hit_test = zone.HitTestFilledArea
    vector_ctor = pcbnew.VECTOR2I
    margin_iu = 1

    try:
        if area_mask is None:
            for r_idx, y_iu in enumerate(y_vals):
                row_mask = mask[r_idx]
                for c_idx, x_iu in enumerate(x_vals):
                    row_mask[c_idx] = bool(hit_test(lid, vector_ctor(int(x_iu), int(y_iu)), margin_iu))
        else:
            for r_idx, y_iu in enumerate(y_vals):
                allowed = area_mask[r_idx]
                if not np.any(allowed):
                    continue
                row_mask = mask[r_idx]
                for c_idx, x_iu in enumerate(x_vals):
                    if allowed[c_idx]:
                        row_mask[c_idx] = bool(hit_test(lid, vector_ctor(int(x_iu), int(y_iu)), margin_iu))
    except Exception:
        for r_idx, y_iu in enumerate(y_vals):
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
            except Exception:
                continue


def _fill_zone_mask_hit_test(mask, area_mask, x_vals, y_vals, zone):
    """Populate a boolean mask using generic HitTest."""
    hit_test = zone.HitTest
    vector_ctor = pcbnew.VECTOR2I

    try:
        if area_mask is None:
            for r_idx, y_iu in enumerate(y_vals):
                row_mask = mask[r_idx]
                for c_idx, x_iu in enumerate(x_vals):
                    row_mask[c_idx] = bool(hit_test(vector_ctor(int(x_iu), int(y_iu))))
        else:
            for r_idx, y_iu in enumerate(y_vals):
                allowed = area_mask[r_idx]
                if not np.any(allowed):
                    continue
                row_mask = mask[r_idx]
                for c_idx, x_iu in enumerate(x_vals):
                    if allowed[c_idx]:
                        row_mask[c_idx] = bool(hit_test(vector_ctor(int(x_iu), int(y_iu))))
    except Exception:
        for r_idx, y_iu in enumerate(y_vals):
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
            except Exception:
                continue


def _scanline_fill_polygon(target, x_vals, y_vals, vertices, value=True):
    """Fill one polygon chain using an even/odd scanline rule."""
    if len(vertices) < 3:
        return
    x0 = vertices[:, 0]
    y0 = vertices[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    non_horizontal = y0 != y1
    if not np.any(non_horizontal):
        return
    x0 = x0[non_horizontal]
    y0 = y0[non_horizontal]
    x1 = x1[non_horizontal]
    y1 = y1[non_horizontal]

    for row_idx, y_value in enumerate(y_vals):
        crosses = ((y0 <= y_value) & (y_value < y1)) | (
            (y1 <= y_value) & (y_value < y0)
        )
        if not np.any(crosses):
            continue
        intersections = x0[crosses] + (
            (float(y_value) - y0[crosses])
            * (x1[crosses] - x0[crosses])
            / (y1[crosses] - y0[crosses])
        )
        intersections.sort()
        pair_count = intersections.size // 2
        if pair_count <= 0:
            continue
        intersections = intersections[:pair_count * 2]
        starts = np.searchsorted(x_vals, intersections[0::2], side="left")
        ends = np.searchsorted(x_vals, intersections[1::2], side="right")
        row = target[row_idx]
        for start, end in zip(starts, ends):
            if start < end:
                row[int(start):int(end)] = value


def _fill_zone_mask_polygons(mask, area_mask, x_vals, y_vals, lid, zone):
    """Rasterize filled zone polygons using a memory-bounded scanline fill."""
    if not hasattr(zone, "GetFilledPolysList"):
        return False
    try:
        if hasattr(zone, "IsFilled") and not zone.IsFilled():
            return False
        if hasattr(zone, "IsOnLayer") and not zone.IsOnLayer(lid):
            return False
        poly_set = zone.GetFilledPolysList(lid)
        outline_count = int(poly_set.OutlineCount())
        if outline_count <= 0:
            return False
        filled = np.zeros(mask.shape, dtype=bool)

        def chain_vertices(chain):
            count = int(chain.PointCount())
            vertices = np.empty((count, 2), dtype=np.float64)
            for idx in range(count):
                point = chain.CPoint(idx)
                vertices[idx] = (point.x, point.y)
            return vertices

        for outline_idx in range(outline_count):
            outer = chain_vertices(poly_set.Outline(outline_idx))
            if len(outer) < 3:
                continue
            inside = np.zeros(mask.shape, dtype=bool)
            _scanline_fill_polygon(inside, x_vals, y_vals, outer, True)
            try:
                hole_count = int(poly_set.HoleCount(outline_idx))
            except Exception:
                hole_count = 0
            for hole_idx in range(hole_count):
                hole = chain_vertices(poly_set.Hole(outline_idx, hole_idx))
                if len(hole) >= 3:
                    _scanline_fill_polygon(inside, x_vals, y_vals, hole, False)
            filled |= inside

        if area_mask is not None:
            filled &= area_mask
        mask[...] = filled
        return True
    except Exception:
        return False


def _state_fill_zone(state, l_idx, lid, zone):
    """
    Mark copper occupancy for a zone using KiCad hit testing.

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

    if _fill_zone_mask_polygons(zone_mask, area_mask, x_vals, y_vals, lid, zone):
        pass
    elif hasattr(zone, "HitTestFilledArea"):
        _fill_zone_mask_hit_test_filled(zone_mask, area_mask, x_vals, y_vals, lid, zone)
    elif hasattr(zone, "HitTest"):
        _fill_zone_mask_hit_test(zone_mask, area_mask, x_vals, y_vals, zone)
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
    via_factor,
    pads_list
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
    via_factor : float
        Enhancement factor for via thermal conductivity.
    pads_list : list
        List of selected pad objects (heat sources).

    Returns
    -------
    GeometryState
        Internal geometry state with boolean copper occupancy and masks.
    """
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
        via_map=np.ones((rows, cols), dtype=np.float64),
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

    try:
        tracks = list(board.Tracks() if hasattr(board, 'Tracks') else board.GetTracks())
        footprints = list(board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints())
        zones = list(board.Zones() if hasattr(board, 'Zones') else board.GetZones())
        drawings = list(board.GetDrawings()) if use_heatsink and hasattr(board, 'GetDrawings') else []

        for track in tracks:
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
                try:
                    diameter_mm = float(track.GetWidth()) * 1e-6
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
                _state_fill_via_circle(
                    state, bbox, center, diameter_mm, via_factor
                )
            elif layer_idx is not None:
                if hasattr(track, "GetMid"):
                    _state_fill_arc(state, layer_idx, track)
                else:
                    _state_fill_segment(state, layer_idx, track)

        for fp in footprints:
            for pad in fp.Pads():
                bbox = pad.GetBoundingBox()
                if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                    pad_layers = list(range(num_layers))
                    _state_fill_pad(state, pad_layers, pad)
                    try:
                        size = pad.GetSize()
                        diameter_mm = min(float(size.x), float(size.y)) * 1e-6
                        center = pad.GetPosition()
                        _state_fill_via_circle(
                            state, bbox, center, diameter_mm, via_factor
                        )
                    except Exception:
                        _state_fill_via(state, bbox, via_factor)
                else:
                    layer_idx = lid_to_idx.get(pad.GetLayer())
                    if layer_idx is not None:
                        _state_fill_pad(state, [layer_idx], pad)

        for zone in zones:
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

            for lid in zone_layers:
                layer_idx = lid_to_idx.get(lid)
                if layer_idx is not None:
                    _state_fill_zone(state, layer_idx, lid, zone)

            if use_heatsink:
                try:
                    if zone.GetLayerSet().Contains(pcbnew.Eco1_User):
                        _state_fill_heatsink(state, zone.GetBoundingBox())
                except Exception:
                    pass

        if use_heatsink:
            for drawing in drawings:
                try:
                    if drawing.GetLayer() == pcbnew.Eco1_User:
                        _state_fill_heatsink(state, drawing.GetBoundingBox())
                except Exception:
                    continue

    except Exception as exc:
        print(f"[ThermalSim][WARN] Geometry mapping error: {exc}")

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


def get_pad_pixels(pad, rows, cols, x_min, y_min, res):
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

    Returns
    -------
    list of tuple
        List of (row, col) tuples for pixels covered by the pad.
    """
    bb = pad.GetBoundingBox()
    x0, y0 = bb.GetX() * 1e-6, bb.GetY() * 1e-6
    w, h = bb.GetWidth() * 1e-6, bb.GetHeight() * 1e-6
    cs = max(0, int((x0 - x_min) / res))
    rs = max(0, int((y0 - y_min) / res))
    ce = min(cols, int((x0 + w - x_min) / res) + 1)
    re = min(rows, int((y0 + h - y_min) / res) + 1)
    pixels = []
    for r in range(rs, re):
        for c in range(cs, ce):
            pixels.append((r, c))
    return pixels


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
    via_factor,
    pads_list
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
