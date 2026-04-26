"""
Electrical DC solver for trace and copper-area Joule heating.

This module builds a net-isolated resistor network on the thermal grid. It
solves the copper potential for configured pad currents and converts the edge
losses into a thermal heat-source vector.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

import pcbnew


@dataclass
class CurrentTerminal:
    """
    Current injection or extraction terminal on a PCB pad.

    Parameters
    ----------
    pad : object
        KiCad pad object.
    name : str
        Human-readable pad name.
    net_name : str
        KiCad net name.
    net_code : int
        KiCad net code.
    current_a : float
        Current in amperes. Positive injects into the PCB, negative extracts.
    """

    pad: Any
    name: str
    net_name: str
    net_code: int
    current_a: float


@dataclass
class ElectricalConfig:
    """
    Geometry and material settings for the electrical solve.

    Parameters
    ----------
    copper_ids : list of int
        Copper layer IDs in stackup order.
    rows, cols : int
        Thermal/electrical grid dimensions.
    x_min, y_min : float
        Grid origin in millimeters.
    res : float
        Grid resolution in millimeters.
    t_cu : np.ndarray
        Copper thickness per copper layer in meters.
    rho_cu : float
        Copper resistivity in ohm-meters.
    via_resistance_ohm : float
        Approximate adjacent-layer via resistance for one occupied grid cell.
    balance_abs_tol : float
        Absolute current-balance tolerance in amperes.
    balance_rel_tol : float
        Relative current-balance tolerance.
    layer_names : dict, optional
        Optional mapping from KiCad layer ID to display name.
    """

    copper_ids: List[int]
    rows: int
    cols: int
    x_min: float
    y_min: float
    res: float
    t_cu: np.ndarray
    rho_cu: float = 1.724e-8
    via_resistance_ohm: float = 1.0e-3
    balance_abs_tol: float = 1.0e-9
    balance_rel_tol: float = 1.0e-6
    layer_names: Optional[Dict[int, str]] = None


@dataclass
class ElectricalTerminalDiagnostics:
    """Diagnostics for one current terminal."""

    name: str
    net_name: str
    current_a: float
    layer: str
    x_mm: float
    y_mm: float
    bbox_mm: Tuple[float, float, float, float]
    cell_count: int
    component_ids: List[int] = field(default_factory=list)
    mean_potential_v: Optional[float] = None


@dataclass
class ElectricalPrimitiveDiagnostics:
    """Geometry primitive summary for one active net and layer/type."""

    net_name: str
    primitive_type: str
    layer: str
    count: int = 0
    track_length_mm: float = 0.0
    track_width_min_mm: Optional[float] = None
    track_width_avg_mm: Optional[float] = None
    track_width_max_mm: Optional[float] = None
    bbox_area_mm2: float = 0.0
    mapped_cell_count: int = 0


@dataclass
class ElectricalNetSummary:
    """Summary diagnostics for one solved electrical net."""

    net_key: str
    net_name: str
    terminal_count: int
    total_current_a: float
    total_abs_current_a: float
    total_loss_w: float
    max_node_power_w: float
    connected_component_count: int
    source_current_a: float = 0.0
    sink_current_a: float = 0.0
    current_balance_a: float = 0.0
    effective_resistance_ohm: Optional[float] = None
    equivalent_voltage_drop_v: Optional[float] = None
    copper_cell_count: int = 0
    edge_count: int = 0
    via_edge_count: int = 0
    pad_voltage_drop_v: Optional[float] = None
    pad_resistance_ohm: Optional[float] = None
    pad_iv_power_w: Optional[float] = None
    source_pad_potential_v: Optional[float] = None
    sink_pad_potential_v: Optional[float] = None
    terminal_diagnostics: List[ElectricalTerminalDiagnostics] = field(default_factory=list)
    primitive_diagnostics: List[ElectricalPrimitiveDiagnostics] = field(default_factory=list)


@dataclass
class ElectricalResult:
    """
    Result of the electrical Joule-heating solve.

    Attributes
    ----------
    q_joule : np.ndarray
        Heat source vector in watts per thermal node.
    net_summaries : list of ElectricalNetSummary
        Per-net diagnostics.
    warnings : list of str
        Non-blocking diagnostics.
    errors : list of str
        Blocking validation failures.
    """

    q_joule: np.ndarray
    net_summaries: List[ElectricalNetSummary]
    warnings: List[str]
    errors: List[str]

    @property
    def valid(self) -> bool:
        """Return True when no blocking validation errors occurred."""
        return not self.errors

    @property
    def total_loss_w(self) -> float:
        """Return total Joule loss over all solved nets."""
        return float(np.sum(self.q_joule))


def net_key_from_values(net_code: Optional[int], net_name: Optional[str]) -> str:
    """
    Build a stable net key from KiCad net identifiers.

    Parameters
    ----------
    net_code : int or None
        KiCad net code.
    net_name : str or None
        KiCad net name.

    Returns
    -------
    str
        Stable key used for grouping.
    """
    try:
        code = int(net_code)
    except Exception:
        code = 0
    name = (net_name or "").strip()
    if code > 0:
        return f"C:{code}"
    if name:
        return f"N:{name}"
    return "NO_NET"


def net_key_from_obj(obj: Any) -> Tuple[str, str, int]:
    """
    Extract a stable net key, display name, and net code from a KiCad object.

    Parameters
    ----------
    obj : object
        KiCad item with optional net methods.

    Returns
    -------
    tuple
        (net_key, net_name, net_code).
    """
    net_name = ""
    net_code = 0
    try:
        net_code = int(obj.GetNetCode())
    except Exception:
        net_code = 0
    try:
        net_name = obj.GetNetname() or ""
    except Exception:
        try:
            net = obj.GetNet()
            net_name = net.GetNetname() or ""
            if not net_code:
                net_code = int(net.GetNetCode())
        except Exception:
            net_name = ""
    return net_key_from_values(net_code, net_name), net_name, net_code


def solve_electrical_heating(
    board: Any,
    terminals: List[CurrentTerminal],
    config: ElectricalConfig,
) -> ElectricalResult:
    """
    Solve electrical DC current flow and return Joule heat per thermal node.

    Parameters
    ----------
    board : pcbnew.BOARD
        Active KiCad board.
    terminals : list of CurrentTerminal
        Pad currents to solve.
    config : ElectricalConfig
        Geometry and material settings.

    Returns
    -------
    ElectricalResult
        Electrical diagnostics and the Joule heat vector.
    """
    layer_count = len(config.copper_ids)
    total_nodes = layer_count * config.rows * config.cols
    q_total = np.zeros(total_nodes, dtype=np.float64)
    errors: List[str] = []
    warnings: List[str] = []
    summaries: List[ElectricalNetSummary] = []

    active_terms = [t for t in terminals if abs(float(t.current_a)) > 0.0]
    if not active_terms:
        return ElectricalResult(q_total, summaries, warnings, errors)

    terms_by_net: Dict[str, List[CurrentTerminal]] = {}
    net_display: Dict[str, str] = {}
    for term in active_terms:
        key = net_key_from_values(term.net_code, term.net_name)
        terms_by_net.setdefault(key, []).append(term)
        net_display[key] = term.net_name or key
        if key == "NO_NET":
            errors.append(f"{term.name}: current terminal has no KiCad net.")

    for key, terms in terms_by_net.items():
        total = float(sum(t.current_a for t in terms))
        total_abs = float(sum(abs(t.current_a) for t in terms))
        tol = max(config.balance_abs_tol, config.balance_rel_tol * total_abs)
        if abs(total) > tol:
            errors.append(
                f"Net {net_display.get(key, key)} is not current-balanced: "
                f"sum(I)={total:.9g} A, tolerance={tol:.3g} A."
            )

    if errors:
        return ElectricalResult(q_total, summaries, warnings, errors)

    net_masks, via_masks, primitive_summaries, collision_count = _build_relevant_net_masks(
        board, config, set(terms_by_net)
    )
    if collision_count:
        errors.append(
            "Copper cells from multiple active nets overlap at the current "
            f"resolution ({collision_count} grid cells). Use a finer resolution."
        )
        return ElectricalResult(q_total, summaries, warnings, errors)

    for key, terms in terms_by_net.items():
        copper_mask = net_masks.get(key)
        if copper_mask is None or not np.any(copper_mask):
            errors.append(f"Net {net_display.get(key, key)} has no mapped copper.")
            continue

        result = _solve_one_net(
            key,
            net_display.get(key, key),
            copper_mask,
            via_masks.get(key),
            terms,
            config,
            primitive_summaries.get(key, []),
        )
        q_total += result.q_joule
        summaries.extend(result.net_summaries)
        warnings.extend(result.warnings)
        errors.extend(result.errors)

    return ElectricalResult(q_total, summaries, warnings, errors)


def _solve_one_net(
    net_key: str,
    net_name: str,
    copper_mask: np.ndarray,
    via_mask: Optional[np.ndarray],
    terms: List[CurrentTerminal],
    config: ElectricalConfig,
    primitive_diagnostics: Optional[List[ElectricalPrimitiveDiagnostics]] = None,
) -> ElectricalResult:
    """Solve one isolated net and return a full-size heat vector."""
    layer_count = len(config.copper_ids)
    rc = config.rows * config.cols
    total_nodes = layer_count * rc
    q_full = np.zeros(total_nodes, dtype=np.float64)
    errors: List[str] = []
    warnings: List[str] = []

    flat_mask = copper_mask.reshape(-1)
    global_indices = np.flatnonzero(flat_mask)
    node_count = int(global_indices.size)
    if node_count == 0:
        errors.append(f"Net {net_name} has no active copper nodes.")
        return ElectricalResult(q_full, [], warnings, errors)

    node_ids = np.full(flat_mask.shape, -1, dtype=np.int64)
    node_ids[global_indices] = np.arange(node_count, dtype=np.int64)
    node_ids = node_ids.reshape(copper_mask.shape)

    edge_i, edge_j, edge_g, via_edge_count = _build_net_edges(copper_mask, via_mask, node_ids, config)
    if edge_i.size:
        adj = sp.coo_matrix(
            (
                np.ones(edge_i.size * 2, dtype=np.int8),
                (np.concatenate([edge_i, edge_j]), np.concatenate([edge_j, edge_i])),
            ),
            shape=(node_count, node_count),
        ).tocsr()
        comp_count, labels = connected_components(adj, directed=False, return_labels=True)
    else:
        comp_count = node_count
        labels = np.arange(node_count, dtype=np.int64)

    rhs = np.zeros(node_count, dtype=np.float64)
    terminal_components = set()
    terminal_component_current: Dict[int, float] = {}
    terminal_records = []

    for term in terms:
        pad_nodes = _pad_node_indices(term.pad, node_ids, config)
        if pad_nodes.size == 0:
            errors.append(f"{term.name}: no copper cell found for current injection on net {net_name}.")
            continue
        current = float(term.current_a)
        unique_pad_nodes = np.unique(pad_nodes)
        rhs[pad_nodes] += current / float(pad_nodes.size)
        comps = set(int(labels[node]) for node in unique_pad_nodes)
        terminal_components.update(comps)
        for comp in comps:
            in_comp = labels[pad_nodes] == comp
            terminal_component_current[comp] = terminal_component_current.get(comp, 0.0) + (
                current * float(np.count_nonzero(in_comp)) / float(pad_nodes.size)
            )
        terminal_records.append((term, unique_pad_nodes, sorted(comps)))

    if len(terminal_components) > 1:
        errors.append(
            f"Current pads on net {net_name} are not electrically connected "
            f"({len(terminal_components)} separate copper islands)."
        )

    total_abs = float(sum(abs(t.current_a) for t in terms))
    tol = max(config.balance_abs_tol, config.balance_rel_tol * total_abs)
    for comp, comp_current in terminal_component_current.items():
        if abs(comp_current) > tol:
            errors.append(
                f"Net {net_name} copper island is not current-balanced: "
                f"sum(I)={comp_current:.9g} A."
            )

    if errors:
        return ElectricalResult(q_full, [], warnings, errors)

    if edge_i.size:
        rows = np.concatenate([edge_i, edge_j, edge_i, edge_j])
        cols = np.concatenate([edge_i, edge_j, edge_j, edge_i])
        data = np.concatenate([edge_g, edge_g, -edge_g, -edge_g])
        lap = sp.coo_matrix((data, (rows, cols)), shape=(node_count, node_count)).tocsr()
    else:
        lap = sp.csr_matrix((node_count, node_count), dtype=np.float64)

    potentials = np.zeros(node_count, dtype=np.float64)
    active_nodes = np.flatnonzero(np.abs(rhs) > 0.0)
    active_components = sorted({int(labels[node]) for node in active_nodes})

    for comp in active_components:
        comp_nodes = np.flatnonzero(labels == comp)
        if comp_nodes.size <= 1:
            continue
        ref = int(comp_nodes[0])
        solve_nodes = comp_nodes[comp_nodes != ref]
        try:
            sub_lap = lap[solve_nodes][:, solve_nodes]
            sub_rhs = rhs[solve_nodes]
            potentials[solve_nodes] = spla.spsolve(sub_lap, sub_rhs)
        except Exception as exc:
            errors.append(f"Electrical solve failed for net {net_name}: {exc}")
            return ElectricalResult(q_full, [], warnings, errors)

    q_nodes = np.zeros(node_count, dtype=np.float64)
    if edge_i.size:
        dv = potentials[edge_i] - potentials[edge_j]
        p_edge = edge_g * dv * dv
        np.add.at(q_nodes, edge_i, 0.5 * p_edge)
        np.add.at(q_nodes, edge_j, 0.5 * p_edge)

    q_full[global_indices] = q_nodes
    total_loss = float(np.sum(q_nodes))
    source_current = float(sum(max(float(t.current_a), 0.0) for t in terms))
    sink_current = float(-sum(min(float(t.current_a), 0.0) for t in terms))
    current_balance = float(sum(t.current_a for t in terms))
    effective_current = source_current if source_current > 0.0 else 0.5 * total_abs
    effective_resistance = (
        total_loss / (effective_current * effective_current)
        if effective_current > 0.0 else None
    )
    equivalent_voltage = (
        total_loss / effective_current
        if effective_current > 0.0 else None
    )

    terminal_diagnostics = []
    for term, pad_nodes, comps in terminal_records:
        mean_potential = float(np.mean(potentials[pad_nodes])) if pad_nodes.size else None
        x_mm, y_mm = _pad_center_mm(term.pad)
        terminal_diagnostics.append(ElectricalTerminalDiagnostics(
            name=str(term.name),
            net_name=str(net_name),
            current_a=float(term.current_a),
            layer=_pad_layer_label(term.pad, config),
            x_mm=x_mm,
            y_mm=y_mm,
            bbox_mm=_bbox_mm(term.pad.GetBoundingBox()),
            cell_count=int(pad_nodes.size),
            component_ids=[int(comp) for comp in comps],
            mean_potential_v=mean_potential,
        ))

    pad_voltage = None
    pad_resistance = None
    pad_iv_power = None
    source_pad_potential = None
    sink_pad_potential = None
    source_terms = [item for item in terminal_diagnostics if item.current_a > 0.0]
    sink_terms = [item for item in terminal_diagnostics if item.current_a < 0.0]
    if len(source_terms) == 1 and len(sink_terms) == 1:
        src = source_terms[0]
        sink = sink_terms[0]
        if src.mean_potential_v is not None and sink.mean_potential_v is not None:
            source_pad_potential = src.mean_potential_v
            sink_pad_potential = sink.mean_potential_v
            pad_voltage = abs(src.mean_potential_v - sink.mean_potential_v)
            current_mag = abs(src.current_a)
            if current_mag > 0.0:
                pad_resistance = pad_voltage / current_mag
                pad_iv_power = pad_voltage * current_mag

    primitives = []
    for item in primitive_diagnostics or []:
        primitives.append(ElectricalPrimitiveDiagnostics(
            net_name=net_name,
            primitive_type=item.primitive_type,
            layer=item.layer,
            count=item.count,
            track_length_mm=item.track_length_mm,
            track_width_min_mm=item.track_width_min_mm,
            track_width_avg_mm=item.track_width_avg_mm,
            track_width_max_mm=item.track_width_max_mm,
            bbox_area_mm2=item.bbox_area_mm2,
            mapped_cell_count=item.mapped_cell_count,
        ))

    summary = ElectricalNetSummary(
        net_key=net_key,
        net_name=net_name,
        terminal_count=len(terms),
        total_current_a=current_balance,
        total_abs_current_a=total_abs,
        total_loss_w=total_loss,
        max_node_power_w=float(np.max(q_nodes)) if q_nodes.size else 0.0,
        connected_component_count=int(comp_count),
        source_current_a=source_current,
        sink_current_a=sink_current,
        current_balance_a=current_balance,
        effective_resistance_ohm=effective_resistance,
        equivalent_voltage_drop_v=equivalent_voltage,
        copper_cell_count=int(node_count),
        edge_count=int(edge_i.size),
        via_edge_count=int(via_edge_count),
        pad_voltage_drop_v=pad_voltage,
        pad_resistance_ohm=pad_resistance,
        pad_iv_power_w=pad_iv_power,
        source_pad_potential_v=source_pad_potential,
        sink_pad_potential_v=sink_pad_potential,
        terminal_diagnostics=terminal_diagnostics,
        primitive_diagnostics=primitives,
    )
    if comp_count > 1:
        warnings.append(
            f"Net {net_name} has {comp_count} mapped copper islands; "
            "only islands with current terminals affect Joule heating."
        )
    return ElectricalResult(q_full, [summary], warnings, errors)


def _build_relevant_net_masks(
    board: Any,
    config: ElectricalConfig,
    relevant_nets: set,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, List[ElectricalPrimitiveDiagnostics]],
    int,
]:
    """Rasterize copper geometry for only the active current nets."""
    layer_count = len(config.copper_ids)
    shape = (layer_count, config.rows, config.cols)
    net_masks = {key: np.zeros(shape, dtype=bool) for key in relevant_nets}
    via_masks = {key: np.zeros((config.rows, config.cols), dtype=bool) for key in relevant_nets}
    primitive_stats: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {
        key: {} for key in relevant_nets
    }
    lid_to_idx = {lid: idx for idx, lid in enumerate(config.copper_ids)}

    def record_primitive(
        key: str,
        primitive_type: str,
        layer: str,
        bbox: Optional[Any],
        mapped_cells: int = 0,
        track_length_mm: float = 0.0,
        track_width_mm: Optional[float] = None,
    ):
        stats = primitive_stats.setdefault(key, {}).setdefault(
            (primitive_type, layer),
            {
                "count": 0,
                "track_length_mm": 0.0,
                "widths": [],
                "bbox_area_mm2": 0.0,
                "mapped_cell_count": 0,
            },
        )
        stats["count"] += 1
        stats["track_length_mm"] += float(track_length_mm or 0.0)
        if track_width_mm is not None:
            stats["widths"].append(float(track_width_mm))
        stats["bbox_area_mm2"] += _bbox_area_mm2(bbox) if bbox is not None else 0.0
        stats["mapped_cell_count"] += int(mapped_cells)

    def fill_for_obj(obj: Any, layer_ids: List[int], bbox=None, as_via=False, use_track_shape=False):
        key, _, _ = net_key_from_obj(obj)
        if key not in net_masks:
            return 0
        mapped_cells = 0
        if as_via:
            bbox_obj = bbox or obj.GetBoundingBox()
            before_via = int(np.count_nonzero(via_masks[key]))
            _fill_bbox_2d(via_masks[key], bbox_obj, config)
            mapped_cells += int(np.count_nonzero(via_masks[key])) - before_via
            for lid in layer_ids:
                layer_idx = lid_to_idx.get(lid)
                if layer_idx is not None:
                    before = int(np.count_nonzero(net_masks[key][layer_idx]))
                    _fill_bbox_3d(net_masks[key], layer_idx, bbox_obj, config)
                    mapped_cells += int(np.count_nonzero(net_masks[key][layer_idx])) - before
            return mapped_cells
        for lid in layer_ids:
            layer_idx = lid_to_idx.get(lid)
            if layer_idx is None:
                continue
            before = int(np.count_nonzero(net_masks[key][layer_idx]))
            if use_track_shape:
                _fill_track(net_masks[key], layer_idx, obj, config)
            else:
                _fill_bbox_3d(net_masks[key], layer_idx, bbox or obj.GetBoundingBox(), config)
            mapped_cells += int(np.count_nonzero(net_masks[key][layer_idx])) - before
        return mapped_cells

    try:
        footprints = list(board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints())
    except Exception:
        footprints = []
    for fp in footprints:
        for pad in fp.Pads():
            key, _, _ = net_key_from_obj(pad)
            if key not in net_masks:
                continue
            bbox = pad.GetBoundingBox()
            if _is_pth_pad(pad):
                mapped_cells = fill_for_obj(pad, config.copper_ids, bbox=bbox, as_via=True)
                record_primitive(
                    key, "Pad", "All copper", bbox,
                    mapped_cells=max(0, mapped_cells)
                )
                record_primitive(
                    key, "Via/PTH", "All copper", bbox,
                    mapped_cells=max(0, mapped_cells)
                )
            else:
                layer_id = pad.GetLayer()
                layer_idx = lid_to_idx.get(layer_id)
                if layer_idx is not None:
                    mapped_cells = fill_for_obj(pad, [layer_id], bbox=bbox)
                    record_primitive(
                        key, "Pad", _layer_label(layer_id, config), bbox,
                        mapped_cells=max(0, mapped_cells)
                    )

    try:
        tracks = list(board.Tracks() if hasattr(board, "Tracks") else board.GetTracks())
    except Exception:
        tracks = []
    for track in tracks:
        key, _, _ = net_key_from_obj(track)
        if key not in net_masks:
            continue
        bbox = track.GetBoundingBox()
        is_via = "VIA" in str(type(track)).upper()
        if is_via:
            layer_ids = _via_layer_ids(track, config.copper_ids)
            mapped_cells = fill_for_obj(track, layer_ids, bbox=bbox, as_via=True)
            record_primitive(
                key, "Via/PTH", _layers_label(layer_ids, config), bbox,
                mapped_cells=max(0, mapped_cells)
            )
        else:
            layer_id = track.GetLayer()
            mapped_cells = fill_for_obj(track, [layer_id], use_track_shape=True)
            record_primitive(
                key,
                "Track",
                _layer_label(layer_id, config),
                bbox,
                mapped_cells=max(0, mapped_cells),
                track_length_mm=_track_length_mm(track),
                track_width_mm=_track_width_mm(track),
            )

    try:
        zones = list(board.Zones() if hasattr(board, "Zones") else board.GetZones())
    except Exception:
        zones = []
    for zone in zones:
        key, _, _ = net_key_from_obj(zone)
        if key not in net_masks:
            continue
        if hasattr(zone, "IsFilled") and not zone.IsFilled():
            continue
        bbox = zone.GetBoundingBox()
        for lid in _zone_layer_ids(zone, config.copper_ids):
            layer_idx = lid_to_idx.get(lid)
            if layer_idx is not None:
                before = int(np.count_nonzero(net_masks[key][layer_idx]))
                _fill_zone(net_masks[key], layer_idx, lid, zone, config)
                mapped_cells = int(np.count_nonzero(net_masks[key][layer_idx])) - before
                record_primitive(
                    key, "Zone", _layer_label(lid, config), bbox,
                    mapped_cells=max(0, mapped_cells)
                )

    collision_count = 0
    if len(net_masks) > 1:
        occupancy = np.zeros(shape, dtype=np.uint8)
        for mask in net_masks.values():
            occupancy += mask.astype(np.uint8)
        collision_count = int(np.count_nonzero(occupancy > 1))

    primitive_summaries: Dict[str, List[ElectricalPrimitiveDiagnostics]] = {}
    for key, stats_by_key in primitive_stats.items():
        summaries = []
        for (primitive_type, layer), stats in sorted(stats_by_key.items()):
            widths = stats["widths"]
            summaries.append(ElectricalPrimitiveDiagnostics(
                net_name=key,
                primitive_type=primitive_type,
                layer=layer,
                count=int(stats["count"]),
                track_length_mm=float(stats["track_length_mm"]),
                track_width_min_mm=min(widths) if widths else None,
                track_width_avg_mm=(sum(widths) / len(widths)) if widths else None,
                track_width_max_mm=max(widths) if widths else None,
                bbox_area_mm2=float(stats["bbox_area_mm2"]),
                mapped_cell_count=int(stats["mapped_cell_count"]),
            ))
        primitive_summaries[key] = summaries

    return net_masks, via_masks, primitive_summaries, collision_count


def _build_net_edges(
    mask: np.ndarray,
    via_mask: Optional[np.ndarray],
    node_ids: np.ndarray,
    config: ElectricalConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build graph edges and conductances for one net mask."""
    edge_i = []
    edge_j = []
    edge_g = []
    via_edge_count = 0
    dx = config.res * 1e-3
    dy = dx
    sigma = 1.0 / max(config.rho_cu, 1e-20)

    for layer_idx in range(mask.shape[0]):
        t_layer = float(config.t_cu[layer_idx])
        gx = sigma * t_layer * dy / dx
        gy = sigma * t_layer * dx / dy

        both = mask[layer_idx, :, :-1] & mask[layer_idx, :, 1:]
        if np.any(both):
            i_idx = node_ids[layer_idx, :, :-1][both]
            j_idx = node_ids[layer_idx, :, 1:][both]
            edge_i.append(i_idx)
            edge_j.append(j_idx)
            edge_g.append(np.full(i_idx.shape, gx, dtype=np.float64))

        both = mask[layer_idx, :-1, :] & mask[layer_idx, 1:, :]
        if np.any(both):
            i_idx = node_ids[layer_idx, :-1, :][both]
            j_idx = node_ids[layer_idx, 1:, :][both]
            edge_i.append(i_idx)
            edge_j.append(j_idx)
            edge_g.append(np.full(i_idx.shape, gy, dtype=np.float64))

    if mask.shape[0] > 1 and via_mask is not None and np.any(via_mask):
        gz = 1.0 / max(float(config.via_resistance_ohm), 1e-12)
        for layer_idx in range(mask.shape[0] - 1):
            both = via_mask & mask[layer_idx] & mask[layer_idx + 1]
            if np.any(both):
                i_idx = node_ids[layer_idx][both]
                j_idx = node_ids[layer_idx + 1][both]
                edge_i.append(i_idx)
                edge_j.append(j_idx)
                edge_g.append(np.full(i_idx.shape, gz, dtype=np.float64))
                via_edge_count += int(i_idx.size)

    if not edge_i:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            0,
        )
    return (
        np.concatenate(edge_i).astype(np.int64, copy=False),
        np.concatenate(edge_j).astype(np.int64, copy=False),
        np.concatenate(edge_g).astype(np.float64, copy=False),
        via_edge_count,
    )


def _pad_node_indices(pad: Any, node_ids: np.ndarray, config: ElectricalConfig) -> np.ndarray:
    """Return electrical node IDs under a pad."""
    layers = []
    lid_to_idx = {lid: idx for idx, lid in enumerate(config.copper_ids)}
    if _is_pth_pad(pad):
        layers = list(range(len(config.copper_ids)))
    else:
        layer_idx = lid_to_idx.get(pad.GetLayer())
        if layer_idx is not None:
            layers = [layer_idx]
    rs, re, cs, ce = _bbox_indices(pad.GetBoundingBox(), config)
    if rs >= re or cs >= ce or not layers:
        return np.empty(0, dtype=np.int64)
    nodes = []
    for layer_idx in layers:
        sub = node_ids[layer_idx, rs:re, cs:ce]
        valid = sub[sub >= 0]
        if valid.size:
            nodes.append(valid.reshape(-1))
    if not nodes:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(nodes))


def _pad_center_mm(pad: Any) -> Tuple[float, float]:
    """Return the pad center in millimeters."""
    try:
        pos = pad.GetPosition()
        return float(pos.x) * 1e-6, float(pos.y) * 1e-6
    except Exception:
        bbox = pad.GetBoundingBox()
        x, y, w, h = _bbox_mm(bbox)
        return x + 0.5 * w, y + 0.5 * h


def _bbox_mm(bbox: Any) -> Tuple[float, float, float, float]:
    """Return a KiCad bounding box as (x, y, w, h) in millimeters."""
    return (
        float(bbox.GetX()) * 1e-6,
        float(bbox.GetY()) * 1e-6,
        float(bbox.GetWidth()) * 1e-6,
        float(bbox.GetHeight()) * 1e-6,
    )


def _bbox_area_mm2(bbox: Optional[Any]) -> float:
    """Return the bounding-box area in square millimeters."""
    if bbox is None:
        return 0.0
    _, _, w, h = _bbox_mm(bbox)
    return max(0.0, w) * max(0.0, h)


def _track_length_mm(track: Any) -> float:
    """Return track centerline length in millimeters."""
    try:
        start = track.GetStart()
        end = track.GetEnd()
        return float(np.hypot(end.x - start.x, end.y - start.y)) * 1e-6
    except Exception:
        bbox = track.GetBoundingBox()
        return max(float(bbox.GetWidth()), float(bbox.GetHeight())) * 1e-6


def _track_width_mm(track: Any) -> Optional[float]:
    """Return track width in millimeters where available."""
    try:
        return float(track.GetWidth()) * 1e-6
    except Exception:
        return None


def _layer_label(layer_id: int, config: ElectricalConfig) -> str:
    """Return a user-facing layer label."""
    if config.layer_names and layer_id in config.layer_names:
        return str(config.layer_names[layer_id])
    known = {}
    for attr in ("F_Cu", "B_Cu", "In1_Cu", "In2_Cu", "In3_Cu", "In4_Cu"):
        if hasattr(pcbnew, attr):
            known[getattr(pcbnew, attr)] = attr.replace("_", ".")
    return known.get(layer_id, f"Layer {layer_id}")


def _layers_label(layer_ids: List[int], config: ElectricalConfig) -> str:
    """Return a compact label for a group of layers."""
    if not layer_ids:
        return "n/a"
    labels = [_layer_label(lid, config) for lid in layer_ids]
    if len(labels) <= 2:
        return " -> ".join(labels)
    return f"{labels[0]} -> {labels[-1]} ({len(labels)} layers)"


def _pad_layer_label(pad: Any, config: ElectricalConfig) -> str:
    """Return the current terminal layer label."""
    if _is_pth_pad(pad):
        return "All copper (PTH)"
    try:
        return _layer_label(pad.GetLayer(), config)
    except Exception:
        return "n/a"


def _bbox_indices(bbox: Any, config: ElectricalConfig) -> Tuple[int, int, int, int]:
    """Convert a KiCad bounding box to grid slice indices."""
    x0 = bbox.GetX() * 1e-6
    y0 = bbox.GetY() * 1e-6
    w = bbox.GetWidth() * 1e-6
    h = bbox.GetHeight() * 1e-6
    cs = max(0, int((x0 - config.x_min) / config.res))
    rs = max(0, int((y0 - config.y_min) / config.res))
    ce = min(config.cols, int((x0 + w - config.x_min) / config.res) + 1)
    re = min(config.rows, int((y0 + h - config.y_min) / config.res) + 1)
    return rs, re, cs, ce


def _fill_bbox_3d(mask: np.ndarray, layer_idx: int, bbox: Any, config: ElectricalConfig):
    """Fill a rectangular region on one layer."""
    rs, re, cs, ce = _bbox_indices(bbox, config)
    if rs < re and cs < ce:
        mask[layer_idx, rs:re, cs:ce] = True


def _fill_bbox_2d(mask: np.ndarray, bbox: Any, config: ElectricalConfig):
    """Fill a rectangular region in a 2D mask."""
    rs, re, cs, ce = _bbox_indices(bbox, config)
    if rs < re and cs < ce:
        mask[rs:re, cs:ce] = True


def _fill_track(mask: np.ndarray, layer_idx: int, track: Any, config: ElectricalConfig):
    """Rasterize a track segment, falling back to its bounding box."""
    if not all(hasattr(track, attr) for attr in ("GetStart", "GetEnd", "GetWidth")):
        _fill_bbox_3d(mask, layer_idx, track.GetBoundingBox(), config)
        return
    try:
        start = track.GetStart()
        end = track.GetEnd()
        width_mm = max(float(track.GetWidth()) * 1e-6, config.res)
    except Exception:
        _fill_bbox_3d(mask, layer_idx, track.GetBoundingBox(), config)
        return

    bbox = track.GetBoundingBox()
    rs, re, cs, ce = _bbox_indices(bbox, config)
    if rs >= re or cs >= ce:
        return

    sx, sy = start.x * 1e-6, start.y * 1e-6
    ex, ey = end.x * 1e-6, end.y * 1e-6
    vx, vy = ex - sx, ey - sy
    seg_len_sq = vx * vx + vy * vy
    radius = 0.5 * width_mm + 0.5 * config.res

    y = config.y_min + (np.arange(rs, re, dtype=np.float64) + 0.5) * config.res
    x = config.x_min + (np.arange(cs, ce, dtype=np.float64) + 0.5) * config.res
    xx, yy = np.meshgrid(x, y)
    if seg_len_sq <= 1e-24:
        dist = np.hypot(xx - sx, yy - sy)
    else:
        t = ((xx - sx) * vx + (yy - sy) * vy) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        px = sx + t * vx
        py = sy + t * vy
        dist = np.hypot(xx - px, yy - py)
    mask[layer_idx, rs:re, cs:ce] |= dist <= radius


def _fill_zone(mask: np.ndarray, layer_idx: int, lid: int, zone: Any, config: ElectricalConfig):
    """Rasterize a filled copper zone with KiCad hit-testing where available."""
    bbox = zone.GetBoundingBox()
    rs, re, cs, ce = _bbox_indices(bbox, config)
    if rs >= re or cs >= ce:
        return

    has_hit = hasattr(zone, "HitTestFilledArea")
    for r in range(rs, re):
        y_mm = config.y_min + (r + 0.5) * config.res
        y_iu = _to_iu(y_mm)
        for c in range(cs, ce):
            if not has_hit:
                mask[layer_idx, r, c] = True
                continue
            x_mm = config.x_min + (c + 0.5) * config.res
            pos = pcbnew.VECTOR2I(_to_iu(x_mm), y_iu)
            try:
                if zone.HitTestFilledArea(lid, pos, 1):
                    mask[layer_idx, r, c] = True
            except TypeError:
                if zone.HitTestFilledArea(lid, pos):
                    mask[layer_idx, r, c] = True


def _zone_layer_ids(zone: Any, copper_ids: List[int]) -> List[int]:
    """Return copper layer IDs occupied by a zone."""
    layer_ids = []
    if hasattr(zone, "IsOnLayer"):
        for lid in copper_ids:
            try:
                if zone.IsOnLayer(lid):
                    layer_ids.append(lid)
            except Exception:
                pass
    if not layer_ids:
        try:
            layer_ids = list(zone.GetLayerSet().IntSeq())
        except Exception:
            layer_ids = []
    if not layer_ids:
        try:
            layer_ids = [zone.GetLayer()]
        except Exception:
            layer_ids = []
    return [lid for lid in layer_ids if lid in copper_ids]


def _via_layer_ids(via: Any, copper_ids: List[int]) -> List[int]:
    """Return layer IDs connected by a via-like object."""
    try:
        layer_set = via.GetLayerSet()
        ids = [lid for lid in copper_ids if layer_set.Contains(lid)]
        if ids:
            return ids
    except Exception:
        pass
    try:
        ids = list(via.GetLayerPair())
        if ids:
            return [lid for lid in copper_ids if min(ids) <= lid <= max(ids)]
    except Exception:
        pass
    if hasattr(via, "_layers"):
        return [lid for lid in getattr(via, "_layers") if lid in copper_ids]
    return list(copper_ids)


def _is_pth_pad(pad: Any) -> bool:
    """Return True for plated-through-hole pads."""
    try:
        return pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH
    except Exception:
        return False


def _to_iu(value_mm: float) -> int:
    """Convert millimeters to KiCad internal units."""
    try:
        return pcbnew.FromMM(value_mm)
    except Exception:
        return int(value_mm * 1e6)
