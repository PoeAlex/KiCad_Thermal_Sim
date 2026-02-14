"""
Current path analysis for I²R heating in PCB traces.

This module solves a DC Laplace equation on the PCB copper geometry
to compute voltage fields, current density, and I²R power loss for
user-defined pad-to-pad current paths. The resulting power density
can be injected into the thermal simulation's Q vector.

Physical basis
--------------
For a conductive domain with electrical conductivity σ (S/m),
Ohm's law gives J = -σ ∇V and continuity requires ∇·J = 0,
yielding the Laplace equation ∇·(σ ∇V) = 0.

Boundary conditions: current injection (+I/N) at pad A cells,
current extraction (-I/N) at pad B cells, one cell pinned to V=0
to remove the singular (constant-shift) mode.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import pcbnew

# Physical constants
SIGMA_CU = 5.8e7     # Copper electrical conductivity (S/m)
SIGMA_BG = 1e-12     # Background conductivity (regularization)


@dataclass
class CurrentPathPair:
    """
    Definition of a pad-to-pad current path.

    Attributes
    ----------
    pad_a : object
        Source pad (pcbnew.PAD or MockPad).
    pad_b : object
        Sink pad (pcbnew.PAD or MockPad).
    current_a : float
        Current in amperes flowing from pad_a to pad_b.
    net_code : int
        Net code shared by both pads.
    label : str
        Human-readable label for this path.
    """
    pad_a: Any
    pad_b: Any
    current_a: float
    net_code: int
    label: str = ""


@dataclass
class CurrentPathResult:
    """
    Results from analyzing a single current path.

    Attributes
    ----------
    resistance_ohm : float
        Path resistance in ohms.
    voltage_drop_v : float
        Voltage drop from pad A to pad B.
    power_loss_w : float
        Total I²R power loss in watts.
    V_field : np.ndarray
        Voltage field, shape (layers, rows, cols).
    J_magnitude : np.ndarray
        Current density magnitude, shape (layers, rows, cols), in A/m².
    Q_i2r : np.ndarray
        Power density per node suitable for adding to thermal Q vector.
    cross_section_data : list
        List of dicts with per-slice cross-section info.
    label : str
        Path label.
    net_code : int
        Net code for this path.
    current_a : float
        Current in amperes.
    """
    resistance_ohm: float
    voltage_drop_v: float
    power_loss_w: float
    V_field: np.ndarray
    J_magnitude: np.ndarray
    Q_i2r: np.ndarray
    cross_section_data: list = field(default_factory=list)
    label: str = ""
    net_code: int = 0
    current_a: float = 0.0


def build_net_conductivity_map(
    board,
    net_code,
    copper_ids,
    rows,
    cols,
    x_min,
    y_min,
    res,
    copper_thickness_m,
    sigma_cu=SIGMA_CU,
    sigma_bg=SIGMA_BG,
):
    """
    Build an electrical conductivity map for a specific net.

    Parameters
    ----------
    board : pcbnew.BOARD
        The KiCad board object.
    net_code : int
        Target net code to extract.
    copper_ids : list of int
        Copper layer IDs in stackup order.
    rows : int
        Grid rows.
    cols : int
        Grid columns.
    x_min : float
        Grid origin X in mm.
    y_min : float
        Grid origin Y in mm.
    res : float
        Grid resolution in mm.
    copper_thickness_m : list of float
        Copper thickness per layer in meters.
    sigma_cu : float
        Copper electrical conductivity (S/m).
    sigma_bg : float
        Background conductivity for non-copper cells.

    Returns
    -------
    np.ndarray
        Conductivity array, shape (layers, rows, cols).
    """
    num_layers = len(copper_ids)
    sigma = np.full((num_layers, rows, cols), sigma_bg, dtype=np.float64)

    lid_to_idx = {lid: i for i, lid in enumerate(copper_ids)}

    def _fill_sigma_box(l_idx, bbox):
        """Fill rectangular region with copper conductivity."""
        x0 = bbox.GetX() * 1e-6
        y0 = bbox.GetY() * 1e-6
        w = bbox.GetWidth() * 1e-6
        h = bbox.GetHeight() * 1e-6
        cs = max(0, int((x0 - x_min) / res))
        rs = max(0, int((y0 - y_min) / res))
        ce = min(cols, int((x0 + w - x_min) / res) + 1)
        re = min(rows, int((y0 + h - y_min) / res) + 1)
        if cs < ce and rs < re:
            sigma[l_idx, rs:re, cs:ce] = sigma_cu

    # Process tracks filtered by net code
    tracks = board.Tracks() if hasattr(board, 'Tracks') else board.GetTracks()
    for t in tracks:
        is_via = "VIA" in str(type(t)).upper()
        try:
            t_net = t.GetNetCode()
        except Exception:
            continue
        if t_net != net_code:
            continue

        lid = t.GetLayer()
        if is_via:
            # Vias fill all layers they span
            for layer_lid in copper_ids:
                if layer_lid in lid_to_idx:
                    _fill_sigma_box(lid_to_idx[layer_lid], t.GetBoundingBox())
        elif lid in lid_to_idx:
            _fill_sigma_box(lid_to_idx[lid], t.GetBoundingBox())

    # Process pads filtered by net code
    footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
    for fp in footprints:
        for pad in fp.Pads():
            try:
                p_net = pad.GetNetCode()
            except Exception:
                continue
            if p_net != net_code:
                continue
            bb = pad.GetBoundingBox()
            if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                for i in range(num_layers):
                    _fill_sigma_box(i, bb)
            else:
                lid = pad.GetLayer()
                if lid in lid_to_idx:
                    _fill_sigma_box(lid_to_idx[lid], bb)

    # Process zones filtered by net code
    zones = board.Zones() if hasattr(board, 'Zones') else board.GetZones()
    for z in zones:
        try:
            z_net = z.GetNetCode()
        except Exception:
            continue
        if z_net != net_code:
            continue
        if hasattr(z, "IsFilled") and not z.IsFilled():
            continue

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

        bbox = z.GetBoundingBox()
        has_hit_test = hasattr(z, "HitTestFilledArea")

        for lid in z_lids:
            if lid not in lid_to_idx:
                continue
            l_idx = lid_to_idx[lid]
            if has_hit_test:
                # Pixel-accurate fill
                x0 = bbox.GetX() * 1e-6
                y0 = bbox.GetY() * 1e-6
                w = bbox.GetWidth() * 1e-6
                h = bbox.GetHeight() * 1e-6
                cs = max(0, int((x0 - x_min) / res))
                rs = max(0, int((y0 - y_min) / res))
                ce = min(cols, int((x0 + w - x_min) / res) + 1)
                re = min(rows, int((y0 + h - y_min) / res) + 1)
                for r in range(rs, re):
                    y = y_min + (r + 0.5) * res
                    y_iu = int(y * 1e6)
                    for c in range(cs, ce):
                        x = x_min + (c + 0.5) * res
                        pos = pcbnew.VECTOR2I(int(x * 1e6), y_iu)
                        try:
                            if z.HitTestFilledArea(lid, pos, 1):
                                sigma[l_idx, r, c] = sigma_cu
                        except Exception:
                            continue
            else:
                _fill_sigma_box(l_idx, bbox)

    return sigma


def _build_net_via_mask(board, net_code, copper_ids, rows, cols, x_min, y_min, res):
    """
    Build a boolean via mask for vias belonging to a specific net.

    Parameters
    ----------
    board : pcbnew.BOARD
        The KiCad board object.
    net_code : int
        Target net code.
    copper_ids : list of int
        Copper layer IDs (unused, kept for API consistency).
    rows : int
        Grid rows.
    cols : int
        Grid columns.
    x_min : float
        Grid origin X in mm.
    y_min : float
        Grid origin Y in mm.
    res : float
        Grid resolution in mm.

    Returns
    -------
    np.ndarray
        Boolean array, shape (rows, cols). True where a via of the target
        net intersects the grid cell.
    """
    via_mask = np.zeros((rows, cols), dtype=bool)
    tracks = board.Tracks() if hasattr(board, 'Tracks') else board.GetTracks()
    for t in tracks:
        if "VIA" not in str(type(t)).upper():
            continue
        try:
            if t.GetNetCode() != net_code:
                continue
        except Exception:
            continue
        bbox = t.GetBoundingBox()
        x0 = bbox.GetX() * 1e-6   # nm to mm
        y0 = bbox.GetY() * 1e-6
        w = bbox.GetWidth() * 1e-6
        h = bbox.GetHeight() * 1e-6
        cs = max(0, int((x0 - x_min) / res))
        rs = max(0, int((y0 - y_min) / res))
        ce = min(cols, int((x0 + w - x_min) / res) + 1)
        re = min(rows, int((y0 + h - y_min) / res) + 1)
        if cs < ce and rs < re:
            via_mask[rs:re, cs:ce] = True
    return via_mask


def build_electrical_stiffness_matrix(
    sigma,
    via_mask,
    layer_count,
    rows,
    cols,
    copper_thickness_m,
    gap_m,
    dx,
    dy,
):
    """
    Build the sparse electrical stiffness matrix (conductance matrix).

    Parameters
    ----------
    sigma : np.ndarray
        Electrical conductivity, shape (layers, rows, cols).
    via_mask : np.ndarray
        Boolean via mask, shape (rows, cols). True at cells that contain
        a via belonging to the analyzed net.
    layer_count : int
        Number of copper layers.
    rows : int
        Grid rows.
    cols : int
        Grid columns.
    copper_thickness_m : list of float
        Copper thickness per layer in meters.
    gap_m : list of float
        Dielectric gap between adjacent layers in meters.
    dx : float
        Grid spacing X in meters.
    dy : float
        Grid spacing Y in meters.

    Returns
    -------
    scipy.sparse.csr_matrix
        Electrical conductance matrix, shape (N, N).
    """
    eps = 1e-30
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

    t_cu = np.array(copper_thickness_m)

    for l in range(layer_count):
        base = l * RC
        s = sigma[l]

        # X-direction coupling: harmonic mean of adjacent conductivities
        s_h = 2.0 * s[:, :-1] * s[:, 1:] / (s[:, :-1] + s[:, 1:] + eps)
        Gx = s_h * (t_cu[l] * dy) / dx

        idx_left = base + row_all * cols + col_right
        idx_right = idx_left + 1
        g = Gx.ravel()
        i_idx = idx_left.ravel()
        j_idx = idx_right.ravel()
        rows_list.extend([i_idx, j_idx, i_idx, j_idx])
        cols_list.extend([i_idx, j_idx, j_idx, i_idx])
        data_list.extend([g, g, -g, -g])

        # Y-direction coupling
        s_h = 2.0 * s[:-1, :] * s[1:, :] / (s[:-1, :] + s[1:, :] + eps)
        Gy = s_h * (t_cu[l] * dx) / dy

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
        plane_idx = np.arange(RC, dtype=np.int64)

        for l in range(layer_count - 1):
            gap_val = max(gap_m[l], 1e-6)
            # At via locations: copper conductivity through the via
            Gz_via = SIGMA_CU * pixel_area / gap_val
            # At non-via locations: effectively zero (insulating FR4)
            Gz = np.where(via_mask.ravel(), Gz_via, SIGMA_BG * pixel_area / gap_val)

            i_idx = l * RC + plane_idx
            j_idx = (l + 1) * RC + plane_idx
            rows_list.extend([i_idx, j_idx, i_idx, j_idx])
            cols_list.extend([i_idx, j_idx, j_idx, i_idx])
            data_list.extend([Gz, Gz, -Gz, -Gz])

    K_elec = sp.coo_matrix(
        (np.concatenate(data_list), (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(N, N),
        dtype=np.float64,
    ).tocsr()

    return K_elec


def solve_current_path(
    K_elec,
    pad_a_pixels,
    pad_b_pixels,
    pad_a_layer_idx,
    pad_b_layer_idx,
    current_a,
    layer_count,
    rows,
    cols,
):
    """
    Solve for the voltage field given current injection at two pads.

    Parameters
    ----------
    K_elec : scipy.sparse.csr_matrix
        Electrical conductance matrix.
    pad_a_pixels : list of (row, col)
        Grid cells covered by pad A.
    pad_b_pixels : list of (row, col)
        Grid cells covered by pad B.
    pad_a_layer_idx : int
        Layer index for pad A.
    pad_b_layer_idx : int
        Layer index for pad B.
    current_a : float
        Current in amperes (positive = into pad A).
    layer_count : int
        Number of layers.
    rows : int
        Grid rows.
    cols : int
        Grid columns.

    Returns
    -------
    tuple
        (V_flat, V_a_avg, V_b_avg) where V_flat is the full voltage
        vector, V_a_avg is mean voltage at pad A, V_b_avg at pad B.
    """
    RC = rows * cols
    N = RC * layer_count

    # Build RHS: current injection
    rhs = np.zeros(N, dtype=np.float64)
    n_a = max(1, len(pad_a_pixels))
    n_b = max(1, len(pad_b_pixels))

    for r, c in pad_a_pixels:
        idx = pad_a_layer_idx * RC + r * cols + c
        if 0 <= idx < N:
            rhs[idx] += current_a / n_a

    for r, c in pad_b_pixels:
        idx = pad_b_layer_idx * RC + r * cols + c
        if 0 <= idx < N:
            rhs[idx] -= current_a / n_b

    # Pin one pad_b cell to V=0 to remove singular mode
    if pad_b_pixels:
        r0, c0 = pad_b_pixels[0]
        pin_idx = pad_b_layer_idx * RC + r0 * cols + c0
    else:
        pin_idx = 0

    # Modify matrix: replace row with identity
    K_mod = K_elec.tolil()
    K_mod[pin_idx, :] = 0
    K_mod[pin_idx, pin_idx] = 1.0
    rhs[pin_idx] = 0.0
    K_mod = K_mod.tocsc()

    # Solve
    lu = spla.splu(K_mod)
    V_flat = lu.solve(rhs)

    # Compute average voltages
    V_a_avg = 0.0
    for r, c in pad_a_pixels:
        idx = pad_a_layer_idx * RC + r * cols + c
        if 0 <= idx < N:
            V_a_avg += V_flat[idx]
    V_a_avg /= n_a

    V_b_avg = 0.0
    for r, c in pad_b_pixels:
        idx = pad_b_layer_idx * RC + r * cols + c
        if 0 <= idx < N:
            V_b_avg += V_flat[idx]
    V_b_avg /= n_b

    return V_flat, V_a_avg, V_b_avg


def compute_current_density(
    V_flat,
    sigma,
    layer_count,
    rows,
    cols,
    dx,
    dy,
):
    """
    Compute current density and power density from voltage field.

    Parameters
    ----------
    V_flat : np.ndarray
        Voltage vector, shape (N,).
    sigma : np.ndarray
        Conductivity, shape (layers, rows, cols).
    layer_count : int
        Number of layers.
    rows : int
        Grid rows.
    cols : int
        Grid columns.
    dx : float
        Grid spacing X in meters.
    dy : float
        Grid spacing Y in meters.

    Returns
    -------
    tuple
        (J_mag, P_density) each of shape (layers, rows, cols).
        J_mag in A/m², P_density in W/m³.
    """
    V = V_flat.reshape((layer_count, rows, cols))

    J_mag = np.zeros((layer_count, rows, cols), dtype=np.float64)
    P_density = np.zeros((layer_count, rows, cols), dtype=np.float64)

    for l in range(layer_count):
        # Gradient via central differences (forward at boundaries)
        Jx = np.zeros((rows, cols), dtype=np.float64)
        Jy = np.zeros((rows, cols), dtype=np.float64)

        # dV/dx
        Jx[:, 1:-1] = -(V[l, :, 2:] - V[l, :, :-2]) / (2.0 * dx)
        Jx[:, 0] = -(V[l, :, 1] - V[l, :, 0]) / dx
        Jx[:, -1] = -(V[l, :, -1] - V[l, :, -2]) / dx

        # dV/dy
        Jy[1:-1, :] = -(V[l, 2:, :] - V[l, :-2, :]) / (2.0 * dy)
        Jy[0, :] = -(V[l, 1, :] - V[l, 0, :]) / dy
        Jy[-1, :] = -(V[l, -1, :] - V[l, -2, :]) / dy

        # J = sigma * E = -sigma * grad(V)
        Jx *= sigma[l]
        Jy *= sigma[l]

        J_mag[l] = np.sqrt(Jx**2 + Jy**2)

        # P = J² / sigma  (or equivalently sigma * |grad V|²)
        P_density[l] = np.where(
            sigma[l] > SIGMA_BG * 10,
            J_mag[l]**2 / sigma[l],
            0.0,
        )

    return J_mag, P_density


def _compute_nodal_power(K_elec, V_flat, N):
    """
    Compute per-node power dissipation from the stiffness matrix.

    Uses P_node[i] = 0.5 * sum_j G_ij * (V_i - V_j)^2 where G_ij are
    the off-diagonal conductances (positive).  Lower-triangular extraction
    ensures each edge is counted exactly once, with half the power assigned
    to each endpoint.

    Parameters
    ----------
    K_elec : scipy.sparse.csr_matrix
        Electrical conductance matrix, shape (N, N).
    V_flat : np.ndarray
        Voltage vector, shape (N,).
    N : int
        Total number of nodes.

    Returns
    -------
    np.ndarray
        Per-node dissipated power in watts, shape (N,).
    """
    K_lower = sp.tril(K_elec, k=-1, format='coo')
    G = -K_lower.data                          # conductances (positive)
    dV = V_flat[K_lower.row] - V_flat[K_lower.col]
    P_edge = G * dV**2
    P_node = np.zeros(N, dtype=np.float64)
    np.add.at(P_node, K_lower.row, 0.5 * P_edge)
    np.add.at(P_node, K_lower.col, 0.5 * P_edge)
    return P_node


def compute_cross_section_profile(
    V_field,
    sigma,
    J_mag,
    copper_thickness_m,
    dx,
    dy,
    n_slices=20,
):
    """
    Compute cross-section profile by binning cells into equipotential slices.

    Parameters
    ----------
    V_field : np.ndarray
        Voltage field, shape (layers, rows, cols).
    sigma : np.ndarray
        Conductivity, shape (layers, rows, cols).
    J_mag : np.ndarray
        Current density magnitude, shape (layers, rows, cols).
    copper_thickness_m : list of float
        Copper thickness per layer in meters.
    dx : float
        Grid spacing X in meters.
    dy : float
        Grid spacing Y in meters.
    n_slices : int
        Number of equipotential slices.

    Returns
    -------
    list of dict
        Per-slice data: v_low, v_high, copper_area_mm2, avg_j, max_j, n_cells.
    """
    # Only consider copper cells
    copper_mask = sigma > SIGMA_BG * 100

    if not np.any(copper_mask):
        return []

    V_copper = V_field[copper_mask]
    v_min, v_max = float(np.min(V_copper)), float(np.max(V_copper))
    if abs(v_max - v_min) < 1e-15:
        return []

    t_cu = np.array(copper_thickness_m)
    layer_count = V_field.shape[0]

    slices = []
    edges = np.linspace(v_min, v_max, n_slices + 1)

    for k in range(n_slices):
        v_lo, v_hi = edges[k], edges[k + 1]
        # Include upper edge in last bin
        if k == n_slices - 1:
            in_slice = copper_mask & (V_field >= v_lo) & (V_field <= v_hi)
        else:
            in_slice = copper_mask & (V_field >= v_lo) & (V_field < v_hi)

        n_cells = int(np.sum(in_slice))
        if n_cells == 0:
            slices.append({
                "v_low": v_lo, "v_high": v_hi,
                "copper_area_mm2": 0.0,
                "avg_j": 0.0, "max_j": 0.0,
                "n_cells": 0,
            })
            continue

        # Copper cross-section area: sum of (dx or dy) * t_cu for each cell
        # This is an approximation: each cell contributes pixel_width * thickness
        area = 0.0
        for l in range(layer_count):
            n_on_layer = int(np.sum(in_slice[l]))
            area += n_on_layer * min(dx, dy) * t_cu[l]
        area_mm2 = area * 1e6  # m² -> mm²

        J_in_slice = J_mag[in_slice]
        slices.append({
            "v_low": v_lo,
            "v_high": v_hi,
            "copper_area_mm2": area_mm2,
            "avg_j": float(np.mean(J_in_slice)),
            "max_j": float(np.max(J_in_slice)),
            "n_cells": n_cells,
        })

    return slices


def analyze_current_path(
    pair,
    K,
    board,
    copper_ids,
    rows,
    cols,
    x_min,
    y_min,
    res,
    copper_thickness_m,
    gap_m,
    get_pad_pixels_func,
):
    """
    Analyze a single current path from pad A to pad B.

    Parameters
    ----------
    pair : CurrentPathPair
        Path definition.
    K : np.ndarray
        Thermal conductivity map (used only for geometry reference).
    board : pcbnew.BOARD
        Board object.
    copper_ids : list of int
        Copper layer IDs.
    rows : int
        Grid rows.
    cols : int
        Grid columns.
    x_min : float
        Grid origin X in mm.
    y_min : float
        Grid origin Y in mm.
    res : float
        Grid resolution in mm.
    copper_thickness_m : list of float
        Copper thickness per layer in meters.
    gap_m : list of float
        Inter-layer gap in meters.
    get_pad_pixels_func : callable
        Function(pad, rows, cols, x_min, y_min, res) -> list of (r, c).

    Returns
    -------
    CurrentPathResult
        Analysis results.
    """
    layer_count = len(copper_ids)
    dx = res * 1e-3
    dy = dx
    pixel_area = dx * dy
    RC = rows * cols
    N = RC * layer_count

    # 1. Build conductivity map for this net
    sigma = build_net_conductivity_map(
        board, pair.net_code, copper_ids, rows, cols,
        x_min, y_min, res, copper_thickness_m,
    )

    # Validate: ensure at least some copper was found
    if not np.any(sigma > SIGMA_BG * 100):
        raise ValueError(
            f"No copper found for net_code={pair.net_code} — "
            f"cannot analyze current path. Check that the net "
            f"has filled copper geometry (tracks, pads, or zones)."
        )

    # 2. Build net-specific via mask and electrical stiffness matrix
    via_mask = _build_net_via_mask(
        board, pair.net_code, copper_ids, rows, cols, x_min, y_min, res,
    )
    K_elec = build_electrical_stiffness_matrix(
        sigma, via_mask, layer_count, rows, cols,
        copper_thickness_m, gap_m, dx, dy,
    )

    # 3. Get pad pixel locations and layer indices
    pad_a_pixels = get_pad_pixels_func(pair.pad_a, rows, cols, x_min, y_min, res)
    pad_b_pixels = get_pad_pixels_func(pair.pad_b, rows, cols, x_min, y_min, res)

    lid_to_idx = {lid: i for i, lid in enumerate(copper_ids)}
    pad_a_lid = pair.pad_a.GetLayer()
    pad_b_lid = pair.pad_b.GetLayer()

    # For PTH pads, use top layer
    pad_a_layer_idx = lid_to_idx.get(pad_a_lid, 0)
    pad_b_layer_idx = lid_to_idx.get(pad_b_lid, 0)

    # 4. Solve for voltage field
    V_flat, V_a_avg, V_b_avg = solve_current_path(
        K_elec, pad_a_pixels, pad_b_pixels,
        pad_a_layer_idx, pad_b_layer_idx,
        pair.current_a, layer_count, rows, cols,
    )

    # 5. Compute current density and power density
    J_mag, P_density = compute_current_density(
        V_flat, sigma, layer_count, rows, cols, dx, dy,
    )

    # 6. Derive resistance and power
    voltage_drop = abs(V_a_avg - V_b_avg)
    resistance = voltage_drop / pair.current_a if pair.current_a > 0 else 0.0
    power_loss = pair.current_a * voltage_drop

    # 7. Build Q_i2r vector (W per node) from stiffness matrix
    # Direct computation avoids gradient artefacts at copper/background boundaries
    Q_i2r = _compute_nodal_power(K_elec, V_flat, N)

    # 8. Cross-section profile
    V_field = V_flat.reshape((layer_count, rows, cols))
    cross_section = compute_cross_section_profile(
        V_field, sigma, J_mag, copper_thickness_m, dx, dy,
    )

    return CurrentPathResult(
        resistance_ohm=resistance,
        voltage_drop_v=voltage_drop,
        power_loss_w=power_loss,
        V_field=V_field,
        J_magnitude=J_mag,
        Q_i2r=Q_i2r,
        cross_section_data=cross_section,
        label=pair.label,
        net_code=pair.net_code,
        current_a=pair.current_a,
    )


def analyze_all_paths(
    pairs,
    K,
    board,
    copper_ids,
    rows,
    cols,
    x_min,
    y_min,
    res,
    copper_thickness_m,
    gap_m,
    get_pad_pixels_func,
):
    """
    Analyze all current path pairs.

    Parameters
    ----------
    pairs : list of CurrentPathPair
        Path definitions.
    K, board, copper_ids, rows, cols, x_min, y_min, res,
    copper_thickness_m, gap_m, get_pad_pixels_func :
        See analyze_current_path.

    Returns
    -------
    list of CurrentPathResult
        Results for each path.
    """
    results = []
    for pair in pairs:
        try:
            result = analyze_current_path(
                pair, K, board, copper_ids,
                rows, cols, x_min, y_min, res,
                copper_thickness_m, gap_m, get_pad_pixels_func,
            )
            results.append(result)
            print(
                f"[ThermalSim][CurrentPath] {pair.label}: "
                f"R={result.resistance_ohm*1e3:.3f} mOhm, "
                f"V_drop={result.voltage_drop_v*1e3:.3f} mV, "
                f"P_loss={result.power_loss_w*1e3:.3f} mW"
            )
        except Exception as e:
            print(f"[ThermalSim][CurrentPath][ERROR] {pair.label}: {e}")
    return results


def merge_i2r_into_q(Q, results):
    """
    Add I²R power losses from current path analysis into the thermal Q vector.

    Parameters
    ----------
    Q : np.ndarray
        Base thermal heat source vector, shape (N,).
    results : list of CurrentPathResult
        Current path results containing Q_i2r vectors.

    Returns
    -------
    np.ndarray
        Updated Q vector with I²R losses added.
    """
    Q_updated = Q.copy()
    for r in results:
        if r.Q_i2r is not None and r.Q_i2r.shape == Q.shape:
            Q_updated += r.Q_i2r
    return Q_updated
