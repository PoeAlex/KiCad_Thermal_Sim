"""
PCB geometry to thermal grid mapping.

This module converts KiCad PCB geometry (copper, vias, zones) into
discretized conductivity arrays for thermal simulation.
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Set

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
    num_layers = len(copper_ids)
    K = np.ones((num_layers, rows, cols)) * k_fr4
    V = np.ones((rows, cols))
    H = np.zeros((rows, cols))

    limit_area = settings.get('limit_area', False)
    radius_mm = settings.get('pad_dist_mm', 0.0) if limit_area else 0.0
    area_mask = build_pad_distance_mask(pads_list, rows, cols, x_min, y_min, res, radius_mm)

    ctx = FillContext(
        K=K, V=V, H=H,
        area_mask=area_mask,
        x_min=x_min, y_min=y_min,
        res=res, rows=rows, cols=cols
    )

    # Collect pad nets if filtering polygons
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
                    net = pad.GetNet()
                    pad_net_names.add(net.GetNetname())
                except Exception:
                    pass
        pad_net_codes = {code for code in pad_net_codes if code is not None}
        pad_net_names = {name for name in pad_net_names if name}

    # Map layer IDs to indices for fast lookup
    lid_to_idx = {lid: i for i, lid in enumerate(copper_ids)}

    def safe_fill(lid, bbox):
        if lid in lid_to_idx:
            idx = lid_to_idx[lid]
            _fill_box(ctx, idx, bbox, k_cu_layers[idx])

    try:
        # Process tracks and vias
        tracks = board.Tracks() if hasattr(board, 'Tracks') else board.GetTracks()
        for t in tracks:
            is_via = "VIA" in str(type(t)).upper()
            if settings.get('ignore_traces') and not is_via:
                continue
            lid = t.GetLayer()
            safe_fill(lid, t.GetBoundingBox())

            if is_via:
                _fill_via(ctx, t.GetBoundingBox(), via_factor)

        # Process footprint pads
        footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
        for fp in footprints:
            for pad in fp.Pads():
                bb = pad.GetBoundingBox()
                if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                    # PTH pads exist on all copper layers
                    for i in range(num_layers):
                        _fill_box(ctx, i, bb, k_cu_layers[i])
                    _fill_via(ctx, bb, via_factor)
                else:
                    # SMD pads
                    safe_fill(pad.GetLayer(), bb)

        # Process zones
        zones = board.Zones() if hasattr(board, 'Zones') else board.GetZones()
        for z in zones:
            if hasattr(z, "IsFilled") and not z.IsFilled():
                is_rule_area = getattr(z, "GetIsRuleArea", lambda: False)()
                is_keepout = getattr(z, "GetIsKeepout", lambda: False)()
                if is_rule_area or is_keepout:
                    continue

            if settings.get('ignore_polygons'):
                zone_net_name = None
                zone_net_code = None
                try:
                    zone_net_name = z.GetNetname()
                except Exception:
                    try:
                        zone_net_name = z.GetNet().GetNetname()
                    except Exception:
                        zone_net_name = None
                try:
                    zone_net_code = z.GetNetCode()
                except Exception:
                    zone_net_code = None
                if pad_net_names:
                    if zone_net_name not in pad_net_names:
                        continue
                elif pad_net_codes:
                    if zone_net_code not in pad_net_codes:
                        continue

            # Check all layers the zone might be on
            z_lids = []
            if hasattr(z, "IsOnLayer"):
                for lid in copper_ids:
                    try:
                        if z.IsOnLayer(lid):
                            z_lids.append(lid)
                    except Exception:
                        continue
            if not z_lids:
                try:
                    z_lids = list(z.GetLayerSet().IntSeq())
                except Exception:
                    z_lids = []
            if not z_lids:
                try:
                    z_lids = [z.GetLayer()]
                except Exception:
                    z_lids = []

            for lid in z_lids:
                if lid in lid_to_idx:
                    idx = lid_to_idx[lid]
                    _fill_zone(ctx, idx, lid, z, k_cu_layers[idx])

            # Check for heatsink on User.Eco1
            if settings['use_heatsink']:
                z_ls = z.GetLayerSet()
                if z_ls.Contains(pcbnew.Eco1_User):
                    _fill_heatsink(ctx, z.GetBoundingBox())

        # Process drawings on User.Eco1 for heatsink
        if settings['use_heatsink']:
            for d in board.GetDrawings():
                if d.GetLayer() == pcbnew.Eco1_User:
                    _fill_heatsink(ctx, d.GetBoundingBox())

    except Exception:
        pass  # Silent fail for single element errors

    return K, V, H
