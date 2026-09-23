"""
Electrical DC solver for trace and copper-area Joule heating.

This module builds a net-isolated resistor network on the thermal grid. It
solves the copper potential for configured pad currents and converts the edge
losses into a thermal heat-source vector.
"""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

import pcbnew

from .geometry_mapper import _circle_from_three_points, _fill_zone_mask_polygons

# Electrical topology tolerances in millimetres.
# Keep centreline topology strict, but allow small endpoint-to-endpoint
# gaps from CAD/generated routing segmentation to snap together.
TRACK_TOPOLOGY_TOL_MM = 1.0e-6
TRACK_ENDPOINT_SNAP_TOL_MM = 0.025
# Conservative fallback for wide routed copper whose generated centreline
# primitives stop just short of one another while the physical copper still
# overlaps.  Keep this disabled for the 0.25 mm mains winding so it cannot
# reintroduce turn-to-turn or segment-skipping shortcuts.
TRACK_ENDPOINT_COPPER_OVERLAP_MIN_WIDTH_MM = 0.35
TRACK_ENDPOINT_COPPER_OVERLAP_MAX_GAP_MM = 0.18


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
    connectivity_report_path: Optional[str] = None


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
    raw_net_names: List[str] = field(default_factory=list)
    diagonal_edge_count: int = 0
    rejected_diagonal_candidate_count: int = 0
    net_tie_edge_count: int = 0
    component_count_before_ties: int = 0
    component_count_after_ties: int = 0
    cardinal_edge_count: int = 0
    contact_edge_count: int = 0
    primitive_count: int = 0
    max_node_degree: int = 0
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


@dataclass(frozen=True)
class ElectricalSupernet:
    """Declared KiCad net-tie-connected electrical nets."""

    key: str
    member_keys: Tuple[str, ...]
    member_names: Tuple[str, ...]

    @property
    def display_name(self) -> str:
        """Return a deterministic user-facing supernet label."""
        names = [
            name for name in self.member_names
            if not name.lower().startswith("unconnected-(")
        ]
        return " + ".join(names or self.member_names)


@dataclass
class PreparedElectricalGeometry:
    """Raster and primitive data shared by electrical solving and preview."""

    rasters: Dict[str, Any]
    primitive_diagnostics: Dict[str, List[ElectricalPrimitiveDiagnostics]]
    raw_net_names: Dict[str, str]
    collision_count: int
    raw_to_group: Dict[str, str]
    contact_points: Dict[str, List[Tuple[Any, Any, int, float, float, str]]]
    net_tie_links: Dict[str, List[Tuple[Any, Any]]]


@dataclass
class _RawNetRaster:
    """Union occupancy plus primitive-owned electrical geometry for one raw net."""

    copper_mask: np.ndarray
    via_mask: np.ndarray
    connect_right: np.ndarray
    connect_down: np.ndarray
    connect_down_right: np.ndarray
    connect_down_left: np.ndarray
    primitives: List[Any] = field(default_factory=list)


@dataclass
class _PrimitiveRaster:
    """Local electrical raster owned by one KiCad primitive.

    Primitive masks are deliberately bounded to the primitive's grid window
    instead of allocating one full-board mask per track/pad/via.  ``row0`` and
    ``col0`` locate the local arrays inside the global electrical grid.
    """

    obj: Any
    kind: str
    layer_id: Optional[int]
    row0: int
    col0: int
    copper_mask: np.ndarray
    via_mask: np.ndarray
    connect_right: np.ndarray
    connect_down: np.ndarray
    connect_down_right: np.ndarray
    connect_down_left: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.copper_mask.shape[1])

    @property
    def cols(self) -> int:
        return int(self.copper_mask.shape[2])


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


def build_electrical_supernet_map(board: Any) -> Dict[str, ElectricalSupernet]:
    """
    Map raw KiCad nets joined by declared footprint net ties to supernets.

    Parameters
    ----------
    board : pcbnew.BOARD
        Active KiCad board.

    Returns
    -------
    dict of str to ElectricalSupernet
        Lookup from each raw net key in a declared net tie to its deterministic
        electrical supernet. Untied nets are intentionally omitted.
    """
    parents: Dict[str, str] = {}
    names: Dict[str, str] = {}

    def find(key: str) -> str:
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    def union(left: str, right: str):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    try:
        footprints = list(board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints())
    except Exception:
        footprints = []
    for footprint in footprints:
        try:
            pads = list(footprint.Pads())
        except Exception:
            continue
        get_tie_pads = getattr(footprint, "GetNetTiePads", None)
        if not callable(get_tie_pads):
            continue
        for pad in pads:
            try:
                tie_pads = list(get_tie_pads(pad))
            except Exception:
                continue
            if not tie_pads:
                continue
            tie_pads.insert(0, pad)
            keys = []
            for tie_pad in tie_pads:
                key, name, _ = net_key_from_obj(tie_pad)
                if key == "NO_NET":
                    continue
                parents.setdefault(key, key)
                names.setdefault(key, name or key)
                keys.append(key)
            keys = list(dict.fromkeys(keys))
            if len(keys) < 2:
                continue
            for key in keys[1:]:
                union(keys[0], key)

    members: Dict[str, List[str]] = {}
    for key in parents:
        members.setdefault(find(key), []).append(key)
    result = {}
    for keys in members.values():
        member_keys = tuple(sorted(keys))
        supernet = ElectricalSupernet(
            key=member_keys[0],
            member_keys=member_keys,
            member_names=tuple(sorted(names[key] for key in member_keys)),
        )
        result.update({key: supernet for key in member_keys})
    return result


def _build_net_tie_links(
    board: Any,
    relevant_nets: set,
    supernet_map: Dict[str, ElectricalSupernet],
) -> Dict[str, List[Tuple[Any, Any]]]:
    """Return declared net-tie pad links for active electrical supernets."""
    links: Dict[str, List[Tuple[Any, Any]]] = {key: [] for key in relevant_nets}
    seen = set()
    try:
        footprints = list(board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints())
    except Exception:
        return links

    for footprint in footprints:
        get_tie_pads = getattr(footprint, "GetNetTiePads", None)
        if not callable(get_tie_pads):
            continue
        try:
            reference = str(footprint.GetReference())
            pads = list(footprint.Pads())
        except Exception:
            continue
        for pad in pads:
            raw_key, _, _ = net_key_from_obj(pad)
            supernet = supernet_map.get(raw_key)
            key = supernet.key if supernet else raw_key
            if key not in links:
                continue
            try:
                tie_pads = list(get_tie_pads(pad))
            except Exception:
                continue
            for tie_pad in tie_pads:
                tie_key, _, _ = net_key_from_obj(tie_pad)
                tie_supernet = supernet_map.get(tie_key)
                if (tie_supernet.key if tie_supernet else tie_key) != key:
                    continue
                try:
                    pad_number = str(pad.GetNumber())
                    tie_number = str(tie_pad.GetNumber())
                except Exception:
                    pad_number, tie_number = str(id(pad)), str(id(tie_pad))
                if pad_number == tie_number:
                    continue
                identity = (reference,) + tuple(sorted((pad_number, tie_number)))
                if identity not in seen:
                    seen.add(identity)
                    links[key].append((pad, tie_pad))
    return links


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

    supernet_map = build_electrical_supernet_map(board)
    terms_by_net: Dict[str, List[CurrentTerminal]] = {}
    net_display: Dict[str, str] = {}
    active_supernets: Dict[str, ElectricalSupernet] = {}
    for term in active_terms:
        raw_key = net_key_from_values(term.net_code, term.net_name)
        supernet = supernet_map.get(raw_key)
        key = supernet.key if supernet else raw_key
        terms_by_net.setdefault(key, []).append(term)
        net_display[key] = supernet.display_name if supernet else (term.net_name or key)
        if supernet:
            active_supernets[key] = supernet
        if raw_key == "NO_NET":
            errors.append(f"{term.name}: current terminal has no KiCad net.")

    for key, terms in terms_by_net.items():
        total = float(sum(t.current_a for t in terms))
        total_abs = float(sum(abs(t.current_a) for t in terms))
        tol = max(config.balance_abs_tol, config.balance_rel_tol * total_abs)
        if key in active_supernets:
            print(
                f"[ThermalSim] Net-tie supernet {net_display[key]}: "
                f"sum(I)={total:.9g} A"
            )
        if abs(total) > tol:
            errors.append(
                f"Net {net_display.get(key, key)} is not current-balanced: "
                f"sum(I)={total:.9g} A, tolerance={tol:.3g} A."
            )

    if errors:
        return ElectricalResult(q_total, summaries, warnings, errors)

    prepared = prepare_electrical_geometry(board, active_terms, config)
    report_path = config.connectivity_report_path
    raw_rasters = prepared.rasters
    primitive_summaries = prepared.primitive_diagnostics
    raw_net_names = prepared.raw_net_names
    raw_to_group = prepared.raw_to_group
    collision_count = prepared.collision_count
    net_tie_links = prepared.net_tie_links
    if collision_count:
        errors.append(
            "Copper cells from multiple active nets overlap at the current "
            f"resolution ({collision_count} grid cells). Use a finer resolution."
        )
        result = ElectricalResult(q_total, summaries, warnings, errors)
        if report_path:
            _try_write_electrical_connectivity_report(
                report_path, prepared, active_terms, config, result, warnings
            )
        return result

    for key, terms in terms_by_net.items():
        raw_keys = [raw for raw, group in raw_to_group.items() if group == key]
        net_rasters = {raw: raw_rasters[raw] for raw in raw_keys if raw in raw_rasters}
        if not net_rasters or not any(np.any(raster.copper_mask) for raster in net_rasters.values()):
            errors.append(f"Net {net_display.get(key, key)} has no mapped copper.")
            continue

        result = _solve_one_net(
            key,
            net_display.get(key, key),
            net_rasters,
            terms,
            config,
            [item for raw in raw_keys for item in primitive_summaries.get(raw, [])],
            net_tie_links.get(key, []),
            raw_net_names,
            prepared.contact_points,
        )
        q_total += result.q_joule
        summaries.extend(result.net_summaries)
        warnings.extend(result.warnings)
        errors.extend(result.errors)

    result = ElectricalResult(q_total, summaries, warnings, errors)
    if report_path:
        _try_write_electrical_connectivity_report(
            report_path, prepared, active_terms, config, result, warnings
        )
    return result


def prepare_electrical_geometry(
    board: Any,
    terminals: List[CurrentTerminal],
    config: ElectricalConfig,
) -> PreparedElectricalGeometry:
    """Prepare primitive-owned electrical geometry for solving and preview.

    Parameters
    ----------
    board : pcbnew.BOARD
        Board containing the active terminal nets.
    terminals : list of CurrentTerminal
        Active current terminals that determine which raw nets are rasterized.
    config : ElectricalConfig
        Grid, copper-layer, and material configuration.

    Returns
    -------
    PreparedElectricalGeometry
        Per-net primitive rasters, contact locations, net-tie links, and diagnostics.
    """
    supernet_map = build_electrical_supernet_map(board)
    raw_to_group = {}
    active_raw_nets = set()
    for terminal in terminals:
        raw_key = net_key_from_values(terminal.net_code, terminal.net_name)
        supernet = supernet_map.get(raw_key)
        group = supernet.key if supernet else raw_key
        members = supernet.member_keys if supernet else (raw_key,)
        for member in members:
            active_raw_nets.add(member)
            raw_to_group[member] = group
    rasters, diagnostics, names, collisions = _build_relevant_net_masks(
        board, config, active_raw_nets, raw_to_group
    )
    contacts = _primitive_contact_points(rasters, config)
    ties = _build_net_tie_links(board, set(raw_to_group.values()), supernet_map)
    return PreparedElectricalGeometry(rasters, diagnostics, names, collisions, raw_to_group, contacts, ties)


def _electrical_geometry_memory_stats(prepared):
    """Return compact allocation diagnostics for electrical geometry rasters."""
    raw_bytes = 0
    primitive_bytes = 0
    primitive_count = 0
    max_primitive_bytes = 0
    max_primitive_shape = None
    for raster in prepared.rasters.values():
        raw_bytes += sum(
            array.nbytes
            for array in (
                raster.copper_mask,
                raster.via_mask,
                raster.connect_right,
                raster.connect_down,
                raster.connect_down_right,
                raster.connect_down_left,
            )
        )
        for primitive in raster.primitives:
            primitive_count += 1
            current = sum(
                array.nbytes
                for array in (
                    primitive.copper_mask,
                    primitive.via_mask,
                    primitive.connect_right,
                    primitive.connect_down,
                    primitive.connect_down_right,
                    primitive.connect_down_left,
                )
            )
            primitive_bytes += current
            if current > max_primitive_bytes:
                max_primitive_bytes = current
                max_primitive_shape = list(primitive.copper_mask.shape)
    return {
        "raw_raster_bytes": int(raw_bytes),
        "primitive_raster_bytes": int(primitive_bytes),
        "primitive_count": int(primitive_count),
        "max_primitive_raster_bytes": int(max_primitive_bytes),
        "max_primitive_copper_shape": max_primitive_shape,
    }


def build_electrical_connectivity_report(prepared, terminals, config):
    """Describe the exact post-contact-collapse graph used by the solver."""
    report_nets = []
    groups = {}
    for raw_key, group in prepared.raw_to_group.items():
        groups.setdefault(group, []).append(raw_key)

    for group, raw_keys in sorted(groups.items()):
        owned_by_net = {}
        local_edge_i, local_edge_j = [], []
        node_count_before_collapse = 0
        node_owners = []
        node_ids_by_net = {}
        edge_stats = {
            "internal_edges_before_collapse": 0,
            "via_edges": 0,
            "diagonal_edges": 0,
            "rejected_diagonal_candidates": 0,
        }

        for raw_key in raw_keys:
            raster = prepared.rasters.get(raw_key)
            owned = []
            if raster:
                for primitive in raster.primitives:
                    occupied = np.flatnonzero(primitive.copper_mask.reshape(-1))
                    if not occupied.size:
                        continue
                    ids = np.full(primitive.copper_mask.size, -1, dtype=np.int64)
                    ids[occupied] = np.arange(
                        node_count_before_collapse,
                        node_count_before_collapse + occupied.size,
                    )
                    ids = ids.reshape(primitive.copper_mask.shape)
                    owned.append((primitive, ids))
                    node_owners.extend([(raw_key, primitive)] * occupied.size)
                    edges = _build_net_edges(
                        primitive.copper_mask, primitive.via_mask, ids, config,
                        primitive.connect_right, primitive.connect_down,
                        primitive.connect_down_right, primitive.connect_down_left,
                    )
                    if edges[0].size:
                        local_edge_i.append(edges[0])
                        local_edge_j.append(edges[1])
                        edge_stats["internal_edges_before_collapse"] += int(edges[0].size)
                    edge_stats["via_edges"] += int(edges[3])
                    edge_stats["diagonal_edges"] += int(edges[4])
                    edge_stats["rejected_diagonal_candidates"] += int(edges[5])
                    node_count_before_collapse += int(occupied.size)
            owned_by_net[raw_key] = owned
            node_ids_by_net[raw_key] = owned

        internal_i = (
            np.concatenate(local_edge_i)
            if local_edge_i else np.empty(0, dtype=np.int64)
        )
        internal_j = (
            np.concatenate(local_edge_j)
            if local_edge_j else np.empty(0, dtype=np.int64)
        )

        contact_i, contact_j = _primitive_contact_edges(
            owned_by_net, config, prepared.contact_points
        )
        old_to_new, node_count = _collapse_contact_nodes(
            node_count_before_collapse, contact_i, contact_j
        )

        if internal_i.size:
            internal_i = old_to_new[internal_i]
            internal_j = old_to_new[internal_j]
            keep = internal_i != internal_j
            internal_i, internal_j = internal_i[keep], internal_j[keep]

        component_count_before_ties = _component_count(
            node_count, internal_i, internal_j
        )

        tie_i, tie_j, tie_errors = _build_net_tie_edges(
            prepared.net_tie_links.get(group, []), node_ids_by_net, config
        )
        if tie_i.size:
            tie_i = old_to_new[tie_i]
            tie_j = old_to_new[tie_j]
            keep = tie_i != tie_j
            tie_i, tie_j = tie_i[keep], tie_j[keep]

        edge_i_parts = [array for array in (internal_i, tie_i) if array.size]
        edge_j_parts = [array for array in (internal_j, tie_j) if array.size]
        all_i = np.concatenate(edge_i_parts) if edge_i_parts else np.empty(0, dtype=np.int64)
        all_j = np.concatenate(edge_j_parts) if edge_j_parts else np.empty(0, dtype=np.int64)

        if node_count and all_i.size:
            adjacency = sp.coo_matrix(
                (
                    np.ones(all_i.size * 2, dtype=np.int8),
                    (
                        np.concatenate((all_i, all_j)),
                        np.concatenate((all_j, all_i)),
                    ),
                ),
                shape=(node_count, node_count),
            ).tocsr()
            component_count, labels = connected_components(
                adjacency, directed=False, return_labels=True
            )
            degree = np.bincount(
                np.concatenate((all_i, all_j)), minlength=node_count
            )
        else:
            component_count = node_count
            labels = np.arange(node_count, dtype=np.int64)
            degree = np.zeros(node_count, dtype=np.int64)

        component_data = {}
        for old_node_id, (raw_key, primitive) in enumerate(node_owners):
            collapsed_node_id = int(old_to_new[old_node_id])
            component_id = int(labels[collapsed_node_id])
            component = component_data.setdefault(component_id, {
                "component_id": component_id,
                "collapsed_node_count": 0,
                "original_cell_node_count": 0,
                "layers": set(),
                "raw_nets": set(),
                "primitives": {},
                "terminals": [],
            })
            component["original_cell_node_count"] += 1
            component["raw_nets"].add(
                prepared.raw_net_names.get(raw_key, raw_key)
            )
            layer_indices = np.flatnonzero(
                np.any(primitive.copper_mask, axis=(1, 2))
            )
            component["layers"].update(
                config.layer_names.get(
                    config.copper_ids[i], f"Layer {config.copper_ids[i]}"
                ) if config.layer_names else str(config.copper_ids[i])
                for i in layer_indices
            )
            primitive_key = (raw_key, id(primitive))
            if primitive_key not in component["primitives"]:
                component["primitives"][primitive_key] = _primitive_report_descriptor(
                    primitive,
                    prepared.raw_net_names.get(raw_key, raw_key),
                )
                component["primitives"][primitive_key]["cell_count"] = 0
            component["primitives"][primitive_key]["cell_count"] += 1

        for component_id, component in component_data.items():
            component["collapsed_node_count"] = int(
                np.count_nonzero(labels == component_id)
            )
            # Backward-compatible field used by the electrical preview renderer.
            # Keep the historical meaning: number of original occupied cell nodes
            # represented by this connected component before contact contraction.
            component["cell_count"] = int(component["original_cell_node_count"])

        terminals_out = []
        for terminal in terminals:
            raw_key = net_key_from_values(terminal.net_code, terminal.net_name)
            if prepared.raw_to_group.get(raw_key) != group:
                continue
            pad_nodes = _owned_pad_nodes(
                terminal.pad, owned_by_net.get(raw_key), config
            )
            collapsed_pad_nodes = (
                np.unique(old_to_new[pad_nodes])
                if pad_nodes.size else np.empty(0, dtype=np.int64)
            )
            component_ids = sorted({
                int(labels[node]) for node in collapsed_pad_nodes
            }) if collapsed_pad_nodes.size else []
            terminal_row = {
                "name": str(terminal.name),
                "net": str(terminal.net_name),
                "current_a": float(terminal.current_a),
                "component_ids": component_ids,
                "original_pad_node_count": int(pad_nodes.size),
                "collapsed_pad_node_count": int(collapsed_pad_nodes.size),
            }
            terminals_out.append(terminal_row)
            for component_id in component_ids:
                if component_id in component_data:
                    component_data[component_id]["terminals"].append(
                        str(terminal.name)
                    )

        components = []
        terminal_components = [
            set(terminal["component_ids"]) for terminal in terminals_out
        ]
        active_terminal_components = (
            set().union(*terminal_components) if terminal_components else set()
        )
        terminals_connected = bool(terminal_components) and all(
            len(item) == 1 and item == terminal_components[0]
            for item in terminal_components
        )
        for component in sorted(
            component_data.values(), key=lambda item: item["component_id"]
        ):
            component["layers"] = sorted(component["layers"])
            component["raw_nets"] = sorted(component["raw_nets"])
            component["primitives"] = list(component["primitives"].values())
            component["terminals"] = sorted(set(component["terminals"]))
            components.append(component)

        contacts = []
        contacts_by_kind = {}
        for raw_key in raw_keys:
            for left, right, layer, x, y, kind in prepared.contact_points.get(raw_key, []):
                contacts_by_kind[kind] = contacts_by_kind.get(kind, 0) + 1
                contacts.append({
                    "kind": kind,
                    "raw_net": prepared.raw_net_names.get(raw_key, raw_key),
                    "layer_index": int(layer),
                    "x_mm": float(x),
                    "y_mm": float(y),
                    "left": _primitive_report_descriptor(
                        left, prepared.raw_net_names.get(raw_key, raw_key)
                    ),
                    "right": _primitive_report_descriptor(
                        right, prepared.raw_net_names.get(raw_key, raw_key)
                    ),
                })

        raw_net_names = sorted(
            prepared.raw_net_names.get(key, key) for key in raw_keys
        )
        report_nets.append({
            "net_key": group,
            "net_name": " + ".join(raw_net_names),
            "raw_nets": raw_net_names,
            "connected": bool(components) and len(components) == 1,
            "terminals_connected": terminals_connected,
            "active_terminal_component_ids": sorted(active_terminal_components),
            "component_count_before_net_ties": int(component_count_before_ties),
            "component_count": int(component_count),
            "nodes_before_contact_collapse": int(node_count_before_collapse),
            "nodes_after_contact_collapse": int(node_count),
            "nodes_collapsed_by_contacts": int(
                node_count_before_collapse - node_count
            ),
            "contact_edge_count": int(contact_i.size),
            "contacts_by_kind": contacts_by_kind,
            "net_tie_edge_count": int(tie_i.size),
            "edge_count_after_collapse": int(all_i.size),
            "max_node_degree": int(np.max(degree)) if degree.size else 0,
            "edge_stats": edge_stats,
            "net_tie_errors": list(tie_errors),
            "terminals": terminals_out,
            "contacts": contacts,
            "components": components,
        })

    return {
        # Keep version 1 for compatibility with the existing preview renderer.
        # New diagnostic fields are additive and identified by schema_revision.
        "format_version": 1,
        "schema_revision": 2,
        "graph_semantics": (
            "Components and node counts mirror the solver after primitive-contact "
            "node contraction and before/after explicit net-tie edges."
        ),
        "grid": {
            "rows": config.rows,
            "cols": config.cols,
            "resolution_mm": config.res,
            "origin_mm": [config.x_min, config.y_min],
        },
        "collision_cell_count": int(prepared.collision_count),
        "memory": _electrical_geometry_memory_stats(prepared),
        "nets": report_nets,
    }


def write_electrical_connectivity_report(
    path, prepared, terminals, config, result=None
):
    """Write the exact electrical connectivity debug report as JSON."""
    report = build_electrical_connectivity_report(prepared, terminals, config)
    if result is not None:
        report["solve_result"] = {
            "valid": bool(result.valid),
            "total_loss_w": float(result.total_loss_w),
            "warnings": list(result.warnings),
            "errors": list(result.errors),
            "net_summaries": [asdict(summary) for summary in result.net_summaries],
        }
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, sort_keys=True)
    return path


def _try_write_electrical_connectivity_report(
    path, prepared, terminals, config, result, warnings
):
    """Best-effort debug report writer that never invalidates a solve."""
    try:
        write_electrical_connectivity_report(
            path, prepared, terminals, config, result=result
        )
        print(f"[ThermalSim] Electrical connectivity JSON: {path}")
    except Exception as exc:
        message = f"Could not write electrical connectivity JSON: {exc}"
        warnings.append(message)
        print(f"[ThermalSim][WARN] {message}")


def _primitive_report_descriptor(primitive, net_name):
    """Return a stable JSON-friendly primitive description."""
    descriptor = {
        "type": str(primitive.kind),
        "net": str(net_name),
        "raster_row0": int(getattr(primitive, "row0", 0)),
        "raster_col0": int(getattr(primitive, "col0", 0)),
        "raster_rows": int(getattr(primitive, "rows", primitive.copper_mask.shape[1])),
        "raster_cols": int(getattr(primitive, "cols", primitive.copper_mask.shape[2])),
    }
    try:
        descriptor["bbox_mm"] = list(_bbox_mm(primitive.obj.GetBoundingBox()))
    except Exception:
        descriptor["bbox_mm"] = None
    try:
        layer_id = getattr(primitive, "layer_id", None)
        descriptor["layer_id"] = int(
            primitive.obj.GetLayer() if layer_id is None else layer_id
        )
    except Exception:
        descriptor["layer_id"] = None
    try:
        uuid = primitive.obj.GetUuid()
        descriptor["uuid"] = str(uuid.AsString() if hasattr(uuid, "AsString") else uuid)
    except Exception:
        descriptor["uuid"] = None
    if primitive.kind == "Track":
        try:
            start = primitive.obj.GetStart()
            end = primitive.obj.GetEnd()
            descriptor["start_mm"] = [start.x * 1e-6, start.y * 1e-6]
            descriptor["end_mm"] = [end.x * 1e-6, end.y * 1e-6]
            descriptor["width_mm"] = float(primitive.obj.GetWidth()) * 1e-6
        except Exception:
            pass
    return descriptor


def _contact_report_kind(left, right):
    if left.kind == "Track" and right.kind == "Track":
        _point, kind = _track_contact_with_kind(left.obj, right.obj)
        return kind or "track_centerline_topology"
    if left.kind == "Track" or right.kind == "Track":
        return "track_to_shape"
    return "shape_overlap"


def _solve_one_net(
    net_key: str,
    net_name: str,
    net_rasters: Dict[str, _RawNetRaster],
    terms: List[CurrentTerminal],
    config: ElectricalConfig,
    primitive_diagnostics: Optional[List[ElectricalPrimitiveDiagnostics]] = None,
    net_tie_links: Optional[List[Tuple[Any, Any]]] = None,
    raw_net_names: Optional[Dict[str, str]] = None,
    contact_points: Optional[Dict[str, List[Tuple[Any, Any, int, float, float, str]]]] = None,
) -> ElectricalResult:
    """Solve one supernet while retaining separate raw-net node ownership."""
    layer_count = len(config.copper_ids)
    rc = config.rows * config.cols
    total_nodes = layer_count * rc
    q_full = np.zeros(total_nodes, dtype=np.float64)
    errors: List[str] = []
    warnings: List[str] = []

    node_ids_by_net: Dict[str, List[Tuple[_PrimitiveRaster, np.ndarray]]] = {}
    node_grid_indices = []
    edge_parts = []
    via_edge_count = diagonal_edge_count = rejected_diagonal_count = 0
    node_count = 0
    for raw_key, raster in net_rasters.items():
        node_ids_by_net[raw_key] = []
        for primitive in raster.primitives:
            global_indices = np.flatnonzero(primitive.copper_mask.reshape(-1))
            if not global_indices.size:
                continue
            ids = np.full(primitive.copper_mask.size, -1, dtype=np.int64)
            ids[global_indices] = np.arange(node_count, node_count + global_indices.size, dtype=np.int64)
            ids = ids.reshape(primitive.copper_mask.shape)
            node_ids_by_net[raw_key].append((primitive, ids))
            local_edges = _build_net_edges(
                primitive.copper_mask, primitive.via_mask, ids, config,
                primitive.connect_right, primitive.connect_down,
                primitive.connect_down_right, primitive.connect_down_left,
            )
            edge_parts.append(local_edges[:3])
            via_edge_count += local_edges[3]
            diagonal_edge_count += local_edges[4]
            rejected_diagonal_count += local_edges[5]
            node_count += int(global_indices.size)

    node_grid_indices = []
    for primitives in node_ids_by_net.values():
        for primitive, ids in primitives:
            layer, row, col = np.nonzero(ids >= 0)
            global_row = row + primitive.row0
            global_col = col + primitive.col0
            node_grid_indices.extend(
                (layer * rc + global_row * config.cols + global_col).tolist()
            )

    # Join separate primitives only when their physical copper touches.
    contact_i, contact_j = _primitive_contact_edges(node_ids_by_net, config, contact_points)

    if node_count == 0:
        errors.append(f"Net {net_name} has no active copper nodes.")
        return ElectricalResult(q_full, [], warnings, errors)

    edge_i = np.concatenate([part[0] for part in edge_parts if part[0].size]) if any(part[0].size for part in edge_parts) else np.empty(0, dtype=np.int64)
    edge_j = np.concatenate([part[1] for part in edge_parts if part[1].size]) if edge_i.size else np.empty(0, dtype=np.int64)
    edge_g = np.concatenate([part[2] for part in edge_parts if part[2].size]) if edge_i.size else np.empty(0, dtype=np.float64)
    old_to_new, node_count = _collapse_contact_nodes(node_count, contact_i, contact_j)
    if edge_i.size:
        edge_i = old_to_new[edge_i]
        edge_j = old_to_new[edge_j]
        keep = edge_i != edge_j
        edge_i, edge_j, edge_g = edge_i[keep], edge_j[keep], edge_g[keep]
    component_count_before_ties = _component_count(node_count, edge_i, edge_j)
    tie_i, tie_j, tie_errors = _build_net_tie_edges(net_tie_links or [], node_ids_by_net, config)
    errors.extend(tie_errors)
    if errors:
        return ElectricalResult(q_full, [], warnings, errors)
    net_tie_edge_count = int(tie_i.size)
    if tie_i.size:
        tie_i = old_to_new[tie_i]
        tie_j = old_to_new[tie_j]
        tie_g = np.full(
            tie_i.shape,
            1.0 / max(float(config.via_resistance_ohm), 1e-12),
            dtype=np.float64,
        )
        edge_i = np.concatenate([edge_i, tie_i])
        edge_j = np.concatenate([edge_j, tie_j])
        edge_g = np.concatenate([edge_g, tie_g])
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
    degree = np.bincount(np.concatenate((edge_i, edge_j)), minlength=node_count) if edge_i.size else np.zeros(node_count, dtype=np.int64)

    rhs = np.zeros(node_count, dtype=np.float64)
    terminal_components = set()
    terminal_component_current: Dict[int, float] = {}
    terminal_records = []

    for term in terms:
        raw_key = net_key_from_values(term.net_code, term.net_name)
        node_ids = node_ids_by_net.get(raw_key)
        pad_nodes = _owned_pad_nodes(term.pad, node_ids, config) if node_ids is not None else np.empty(0, dtype=np.int64)
        if pad_nodes.size:
            pad_nodes = old_to_new[pad_nodes]
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

    group_sizes = np.bincount(old_to_new, minlength=node_count)
    old_node_power = q_nodes[old_to_new] / np.maximum(group_sizes[old_to_new], 1)
    np.add.at(q_full, np.asarray(node_grid_indices, dtype=np.int64), old_node_power)
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
        raw_net_names=sorted({(raw_net_names or {}).get(raw, raw) for raw in net_rasters}),
        diagonal_edge_count=int(diagonal_edge_count),
        rejected_diagonal_candidate_count=int(rejected_diagonal_count),
        net_tie_edge_count=net_tie_edge_count,
        component_count_before_ties=int(component_count_before_ties),
        component_count_after_ties=int(comp_count),
        cardinal_edge_count=max(
            0, int(edge_i.size) - via_edge_count - diagonal_edge_count
            - net_tie_edge_count
        ),
        contact_edge_count=int(contact_i.size),
        primitive_count=sum(len(raster.primitives) for raster in net_rasters.values()),
        max_node_degree=int(np.max(degree)) if degree.size else 0,
        pad_voltage_drop_v=pad_voltage,
        pad_resistance_ohm=pad_resistance,
        pad_iv_power_w=pad_iv_power,
        source_pad_potential_v=source_pad_potential,
        sink_pad_potential_v=sink_pad_potential,
        terminal_diagnostics=terminal_diagnostics,
        primitive_diagnostics=primitives,
    )
    if rejected_diagonal_count > max(10, int(node_count * 0.05)):
        warnings.append(
            f"Net {net_name} has {rejected_diagonal_count} diagonal cell adjacencies "
            "without track continuity; consider a finer electrical grid."
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
    raw_to_group: Optional[Dict[str, str]] = None,
) -> Tuple[
    Dict[str, _RawNetRaster],
    Dict[str, List[ElectricalPrimitiveDiagnostics]],
    Dict[str, str],
    int,
]:
    """Rasterize active raw-net geometry without merging net-tie members."""
    layer_count = len(config.copper_ids)
    shape = (layer_count, config.rows, config.cols)
    rasters = {
        key: _RawNetRaster(
            np.zeros(shape, dtype=bool),
            np.zeros((config.rows, config.cols), dtype=bool),
            np.zeros((layer_count, config.rows, max(config.cols - 1, 0)), dtype=bool),
            np.zeros((layer_count, max(config.rows - 1, 0), config.cols), dtype=bool),
            np.zeros((layer_count, max(config.rows - 1, 0), max(config.cols - 1, 0)), dtype=bool),
            np.zeros((layer_count, max(config.rows - 1, 0), max(config.cols - 1, 0)), dtype=bool),
        ) for key in relevant_nets
    }
    raw_net_names = {}
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
        raw_key, raw_name, _ = net_key_from_obj(obj)
        if raw_key not in rasters:
            return 0
        if raw_name:
            raw_net_names[raw_key] = raw_name
        raster = rasters[raw_key]
        bbox_obj = bbox or obj.GetBoundingBox()
        primitive_layer_id = layer_ids[0] if len(layer_ids) == 1 else None
        primitive, local_config = _new_local_primitive(
            obj,
            "Via/PTH" if as_via else ("Track" if use_track_shape else "Pad"),
            bbox_obj,
            config,
            layer_id=primitive_layer_id,
        )
        if primitive is None:
            return 0

        mapped_cells = 0
        if as_via:
            _fill_bbox_2d(primitive.via_mask, bbox_obj, local_config)
            for lid in layer_ids:
                layer_idx = lid_to_idx.get(lid)
                if layer_idx is not None:
                    _fill_bbox_3d(
                        primitive.copper_mask, layer_idx, bbox_obj, local_config
                    )
                    _connect_full_shape(primitive, layer_idx)
            mapped_cells = int(np.count_nonzero(primitive.copper_mask))
            _merge_primitive(raster, primitive)
            return mapped_cells

        for lid in layer_ids:
            layer_idx = lid_to_idx.get(lid)
            if layer_idx is None:
                continue
            if use_track_shape:
                _fill_track(primitive, layer_idx, obj, local_config)
            elif hasattr(obj, "GetSize") and hasattr(obj, "GetPosition"):
                _fill_pad(
                    primitive.copper_mask, layer_idx, obj, lid, local_config
                )
                _connect_full_shape(primitive, layer_idx)
            else:
                _fill_bbox_3d(
                    primitive.copper_mask,
                    layer_idx,
                    bbox_obj,
                    local_config,
                )
                _connect_full_shape(primitive, layer_idx)
            mapped_cells += int(
                np.count_nonzero(primitive.copper_mask[layer_idx])
            )
        _merge_primitive(raster, primitive)
        return mapped_cells

    try:
        footprints = list(board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints())
    except Exception:
        footprints = []
    for fp in footprints:
        for pad in fp.Pads():
            raw_key, _, _ = net_key_from_obj(pad)
            if raw_key not in rasters:
                continue
            key = raw_key
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
                layer_ids = _pad_copper_layer_ids(pad, config.copper_ids)
                if layer_ids:
                    mapped_cells = fill_for_obj(pad, layer_ids, bbox=bbox)
                    record_primitive(
                        key, "Pad", _layers_label(layer_ids, config), bbox,
                        mapped_cells=max(0, mapped_cells)
                    )

    try:
        tracks = list(board.Tracks() if hasattr(board, "Tracks") else board.GetTracks())
    except Exception:
        tracks = []
    for track in tracks:
        raw_key, _, _ = net_key_from_obj(track)
        key = raw_key
        if key not in rasters:
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
        raw_key, raw_name, _ = net_key_from_obj(zone)
        key = raw_key
        if key not in rasters:
            continue
        if raw_name:
            raw_net_names[key] = raw_name
        raster = rasters[key]
        if hasattr(zone, "IsFilled") and not zone.IsFilled():
            continue
        bbox = zone.GetBoundingBox()
        for lid in _zone_layer_ids(zone, config.copper_ids):
            layer_idx = lid_to_idx.get(lid)
            if layer_idx is not None:
                primitive, local_config = _new_local_primitive(
                    zone, "Zone", bbox, config, layer_id=lid
                )
                if primitive is None:
                    continue
                _fill_zone(
                    primitive.copper_mask, layer_idx, lid, zone, local_config
                )
                _connect_zone_boundaries(
                    primitive, layer_idx, lid, zone, local_config
                )
                mapped_cells = int(
                    np.count_nonzero(primitive.copper_mask[layer_idx])
                )
                _merge_primitive(raster, primitive)
                record_primitive(
                    key, "Zone", _layer_label(lid, config), bbox,
                    mapped_cells=max(0, mapped_cells)
                )

    collision_count = 0
    groups = {}
    for raw_key, raster in rasters.items():
        groups.setdefault((raw_to_group or {}).get(raw_key, raw_key), np.zeros(shape, dtype=bool))
        groups[(raw_to_group or {}).get(raw_key, raw_key)] |= raster.copper_mask
    if len(groups) > 1:
        occupancy = np.zeros(shape, dtype=np.uint8)
        for mask in groups.values():
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

    return rasters, primitive_summaries, raw_net_names, collision_count


def _build_net_edges(
    mask: np.ndarray,
    via_mask: Optional[np.ndarray],
    node_ids: np.ndarray,
    config: ElectricalConfig,
    connect_right: np.ndarray,
    connect_down: np.ndarray,
    diag_down_right: np.ndarray,
    diag_down_left: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Build graph edges and conductances for one net mask."""
    edge_i = []
    edge_j = []
    edge_g = []
    via_edge_count = 0
    diagonal_edge_count = 0
    rejected_diagonal_count = 0
    dx = config.res * 1e-3
    dy = dx
    sigma = 1.0 / max(config.rho_cu, 1e-20)

    for layer_idx in range(mask.shape[0]):
        t_layer = float(config.t_cu[layer_idx])
        gx = sigma * t_layer * dy / dx
        gy = sigma * t_layer * dx / dy

        both = mask[layer_idx, :, :-1] & mask[layer_idx, :, 1:] & connect_right[layer_idx]
        if np.any(both):
            i_idx = node_ids[layer_idx, :, :-1][both]
            j_idx = node_ids[layer_idx, :, 1:][both]
            edge_i.append(i_idx)
            edge_j.append(j_idx)
            edge_g.append(np.full(i_idx.shape, gx, dtype=np.float64))

        both = mask[layer_idx, :-1, :] & mask[layer_idx, 1:, :] & connect_down[layer_idx]
        if np.any(both):
            i_idx = node_ids[layer_idx, :-1, :][both]
            j_idx = node_ids[layer_idx, 1:, :][both]
            edge_i.append(i_idx)
            edge_j.append(j_idx)
            edge_g.append(np.full(i_idx.shape, gy, dtype=np.float64))

        diagonal_g = sigma * t_layer * dx / np.hypot(dx, dy)
        candidates = mask[layer_idx, :-1, :-1] & mask[layer_idx, 1:, 1:]
        has_cardinal_route = mask[layer_idx, :-1, 1:] | mask[layer_idx, 1:, :-1]
        both = candidates & diag_down_right[layer_idx] & ~has_cardinal_route
        rejected_diagonal_count += int(np.count_nonzero(candidates & ~diag_down_right[layer_idx]))
        if np.any(both):
            i_idx = node_ids[layer_idx, :-1, :-1][both]
            j_idx = node_ids[layer_idx, 1:, 1:][both]
            edge_i.append(i_idx)
            edge_j.append(j_idx)
            edge_g.append(np.full(i_idx.shape, diagonal_g, dtype=np.float64))
            diagonal_edge_count += int(i_idx.size)

        candidates = mask[layer_idx, :-1, 1:] & mask[layer_idx, 1:, :-1]
        has_cardinal_route = mask[layer_idx, :-1, :-1] | mask[layer_idx, 1:, 1:]
        both = candidates & diag_down_left[layer_idx] & ~has_cardinal_route
        rejected_diagonal_count += int(np.count_nonzero(candidates & ~diag_down_left[layer_idx]))
        if np.any(both):
            i_idx = node_ids[layer_idx, :-1, 1:][both]
            j_idx = node_ids[layer_idx, 1:, :-1][both]
            edge_i.append(i_idx)
            edge_j.append(j_idx)
            edge_g.append(np.full(i_idx.shape, diagonal_g, dtype=np.float64))
            diagonal_edge_count += int(i_idx.size)

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
            diagonal_edge_count,
            rejected_diagonal_count,
        )
    return (
        np.concatenate(edge_i).astype(np.int64, copy=False),
        np.concatenate(edge_j).astype(np.int64, copy=False),
        np.concatenate(edge_g).astype(np.float64, copy=False),
        via_edge_count,
        diagonal_edge_count,
        rejected_diagonal_count,
    )


def _component_count(node_count: int, edge_i: np.ndarray, edge_j: np.ndarray) -> int:
    """Return the number of undirected connected components in a graph."""
    if not edge_i.size:
        return node_count
    adj = sp.coo_matrix(
        (np.ones(edge_i.size * 2, dtype=np.int8),
         (np.concatenate([edge_i, edge_j]), np.concatenate([edge_j, edge_i]))),
        shape=(node_count, node_count),
    ).tocsr()
    return int(connected_components(adj, directed=False, return_labels=False))


def _collapse_contact_nodes(node_count, edge_i, edge_j):
    """Contract physical copper contacts to ideal equipotential graph nodes."""
    parent = np.arange(node_count, dtype=np.int64)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return int(node)

    for left, right in zip(edge_i, edge_j):
        left_root, right_root = find(int(left)), find(int(right))
        if left_root != right_root:
            parent[right_root] = left_root
    roots = np.fromiter((find(index) for index in range(node_count)), dtype=np.int64, count=node_count)
    _, old_to_new = np.unique(roots, return_inverse=True)
    return old_to_new.astype(np.int64, copy=False), int(old_to_new.max() + 1 if node_count else 0)


def _stable_board_item_key(obj):
    """Return a wrapper-stable identity for KiCad board items when possible.

    SWIG may return a new Python wrapper for the same underlying PAD object, so
    Python object identity (``is``) is not reliable for terminal/net-tie lookup.
    Pads are identified primarily by footprint reference + pad number; UUID is
    used when exposed by the KiCad build.
    """
    for getter_name in ("GetUuid",):
        getter = getattr(obj, getter_name, None)
        if callable(getter):
            try:
                value = getter()
                text = str(value.AsString() if hasattr(value, "AsString") else value)
                if text and text.lower() not in ("none", "null"):
                    return ("uuid", text)
            except Exception:
                pass
    try:
        value = getattr(obj, "m_Uuid")
        text = str(value.AsString() if hasattr(value, "AsString") else value)
        if text and text.lower() not in ("none", "null"):
            return ("uuid", text)
    except Exception:
        pass

    get_number = getattr(obj, "GetNumber", None)
    if callable(get_number):
        try:
            number = str(get_number())
            parent = obj.GetParent() if hasattr(obj, "GetParent") else None
            reference = str(parent.GetReference()) if parent is not None and hasattr(parent, "GetReference") else ""
            net_code = int(obj.GetNetCode()) if hasattr(obj, "GetNetCode") else 0
            return ("pad", reference, number, net_code)
        except Exception:
            pass

    return None


def _same_board_item(left, right):
    """Return True when two Python wrappers represent the same board item."""
    if left is right:
        return True
    try:
        if left == right:
            return True
    except Exception:
        pass
    left_key = _stable_board_item_key(left)
    right_key = _stable_board_item_key(right)
    return left_key is not None and left_key == right_key


def _owned_pad_nodes(pad, primitive_nodes, config):
    """Return nodes owned by the terminal or net-tie pad primitive.

    Do not rely on Python wrapper identity: KiCad/SWIG can hand back distinct
    Python objects for the same underlying pad.
    """
    nodes = []
    for primitive, ids in primitive_nodes or []:
        if _same_board_item(primitive.obj, pad):
            found = ids[ids >= 0]
            if found.size:
                nodes.append(found.reshape(-1))
    return np.unique(np.concatenate(nodes)) if nodes else np.empty(0, dtype=np.int64)


def _primitive_contact_points(rasters, config):
    """Resolve geometry-proven same-net primitive contact locations once."""
    contacts = {}
    for raw_key, raster in rasters.items():
        found = []
        for left, right in _candidate_primitive_pairs(raster.primitives):
            point = _primitive_contact_point(left, right, config)
            if point is not None:
                layer, x, y, kind = point
                found.append((left, right, layer, x, y, kind))
        contacts[raw_key] = found
    return contacts


def _candidate_primitive_pairs(primitives):
    """Sweep primitive bounds to limit expensive same-net contact checks."""
    # ponytail: overlapping wide zones can still produce quadratic candidates; add a spatial index if profiling shows it matters.
    bounds = []
    for primitive in primitives:
        try:
            x, y, width, height = _bbox_mm(primitive.obj.GetBoundingBox())
            bounds.append((x, y, x + width, y + height, primitive))
        except Exception:
            bounds.append((-float("inf"), -float("inf"), float("inf"), float("inf"), primitive))
    bounds.sort(key=lambda item: item[0])
    for index, (x0, y0, x1, y1, left) in enumerate(bounds):
        for candidate in range(index + 1, len(bounds)):
            rx0, ry0, rx1, ry1, right = bounds[candidate]
            if rx0 > x1:
                break
            if y0 <= ry1 and ry0 <= y1:
                yield left, right


def _primitive_contact_edges(nodes_by_net, config, contact_points=None):
    """Connect primitive-owned nodes using prepared physical contacts."""
    edge_i, edge_j = [], []
    for raw_key, owned in nodes_by_net.items():
        by_primitive = {id(primitive): (primitive, ids) for primitive, ids in owned}
        for left, right, layer, x, y, kind in (contact_points or {}).get(raw_key, []):
            left_entry = by_primitive.get(id(left))
            right_entry = by_primitive.get(id(right))
            if left_entry is None or right_entry is None:
                continue
            if kind.endswith("_raster_overlap"):
                left_nodes, right_nodes = _overlapping_primitive_nodes(
                    left_entry, right_entry, layer
                )
                edge_i.extend(left_nodes)
                edge_j.extend(right_nodes)
                continue
            point = (layer, x, y)
            left_node = _nearest_primitive_node(
                left_entry[0], left_entry[1], point, config
            )
            right_node = _nearest_primitive_node(
                right_entry[0], right_entry[1], point, config
            )
            if left_node >= 0 and right_node >= 0 and left_node != right_node:
                edge_i.append(left_node)
                edge_j.append(right_node)
    return np.asarray(edge_i, dtype=np.int64), np.asarray(edge_j, dtype=np.int64)


def _overlapping_primitive_nodes(left_entry, right_entry, layer):
    """Return matching node pairs for every shared raster cell."""
    left, left_ids = left_entry
    right, right_ids = right_entry
    if not (0 <= layer < left_ids.shape[0] and layer < right_ids.shape[0]):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    row0 = max(left.row0, right.row0)
    row1 = min(left.row0 + left.rows, right.row0 + right.rows)
    col0 = max(left.col0, right.col0)
    col1 = min(left.col0 + left.cols, right.col0 + right.cols)
    if row0 >= row1 or col0 >= col1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    left_nodes = left_ids[
        layer, row0 - left.row0:row1 - left.row0, col0 - left.col0:col1 - left.col0
    ]
    right_nodes = right_ids[
        layer, row0 - right.row0:row1 - right.row0, col0 - right.col0:col1 - right.col0
    ]
    shared = (left_nodes >= 0) & (right_nodes >= 0)
    return left_nodes[shared], right_nodes[shared]


def _nearest_primitive_node(primitive, ids, point, config):
    """Find the nearest occupied local node to a global physical contact point."""
    layer, x, y = point
    layer_idx = min(max(int(layer), 0), ids.shape[0] - 1)
    global_row = int(math.floor((y - config.y_min) / config.res))
    global_col = int(math.floor((x - config.x_min) / config.res))
    row = global_row - primitive.row0
    col = global_col - primitive.col0
    r0, r1 = max(0, row - 1), min(primitive.rows, row + 2)
    c0, c1 = max(0, col - 1), min(primitive.cols, col + 2)
    if r0 >= r1 or c0 >= c1:
        return -1
    sub = ids[layer_idx, r0:r1, c0:c1]
    rr, cc = np.nonzero(sub >= 0)
    if not rr.size:
        return -1
    rr, cc = rr + r0, cc + c0
    global_rr = rr + primitive.row0
    global_cc = cc + primitive.col0
    distances = (
        config.x_min + (global_cc + 0.5) * config.res - x
    ) ** 2 + (
        config.y_min + (global_rr + 0.5) * config.res - y
    ) ** 2
    best = int(np.argmin(distances))
    return int(ids[layer_idx, rr[best], cc[best]])


def _primitive_raster_overlap_contact(left, right, config):
    """Return a same-layer contact where two primitive copper rasters overlap.

    This is intentionally used for contacts involving pads, vias/PTHs and
    filled zones.  Their physical copper is already represented by the exact
    local primitive rasters used by the resistor graph, so using a single
    bounding-box probe point is both unnecessary and unreliable (especially
    for annular zones and drilled PTHs).

    Track-to-track contacts remain governed by routed-centreline topology and
    are never accepted from raster overlap, preserving winding-turn isolation.
    """
    row0 = max(int(left.row0), int(right.row0))
    col0 = max(int(left.col0), int(right.col0))
    row1 = min(int(left.row0 + left.rows), int(right.row0 + right.rows))
    col1 = min(int(left.col0 + left.cols), int(right.col0 + right.cols))
    if row0 >= row1 or col0 >= col1:
        return None

    lrs = slice(row0 - left.row0, row1 - left.row0)
    lcs = slice(col0 - left.col0, col1 - left.col0)
    rrs = slice(row0 - right.row0, row1 - right.row0)
    rcs = slice(col0 - right.col0, col1 - right.col0)

    overlap = left.copper_mask[:, lrs, lcs] & right.copper_mask[:, rrs, rcs]
    locations = np.argwhere(overlap)
    if not locations.size:
        return None

    # Pick an overlap cell close to the overlap cloud centroid.  This keeps the
    # diagnostic marker representative while guaranteeing that the returned
    # point maps to an occupied node in both primitive-local graphs.
    centroid = np.mean(locations[:, 1:3], axis=0)
    dist2 = np.sum((locations[:, 1:3] - centroid) ** 2, axis=1)
    layer_idx, local_row, local_col = locations[int(np.argmin(dist2))]
    global_row = row0 + int(local_row)
    global_col = col0 + int(local_col)
    x_mm = float(config.x_min + (global_col + 0.5) * config.res)
    y_mm = float(config.y_min + (global_row + 0.5) * config.res)
    return int(layer_idx), x_mm, y_mm


def _primitive_contact_point(left, right, config):
    """Return an accepted electrical contact between two same-net primitives.

    Track/track topology remains geometry based.  Any pair involving a zone,
    pad or via/PTH first uses actual overlap of the primitive copper rasters so
    annular zones, drilled PTHs and tracks entering zones are connected where
    the resistor-network copper really overlaps.
    """
    left_track = _track_geometry(left.obj) if left.kind == "Track" else None
    right_track = _track_geometry(right.obj) if right.kind == "Track" else None
    if left_track and right_track:
        contact, contact_kind = _track_contact_with_kind(left.obj, right.obj)
        if contact is not None:
            lid = getattr(left.obj, "GetLayer", lambda: 0)()
            located = _layer_index_for_obj(lid, left, right, contact)
            if located is not None:
                return (*located, contact_kind)
        return None

    # For all non-track/track pairs, the primitive rasters are the authoritative
    # representation of copper used by the electrical graph.  This replaces
    # the previous bounding-box midpoint/nearest-rectangle probing, which could
    # miss a real zone contact when the sampled point fell in an annular hole.
    overlap = _primitive_raster_overlap_contact(left, right, config)
    if overlap is not None:
        if left.kind == "Track" or right.kind == "Track":
            return (*overlap, "track_to_shape_raster_overlap")
        return (*overlap, "shape_raster_overlap")

    # Retain the former exact physical checks as a conservative fallback for a
    # true boundary touch that happens to fall between raster-cell centres.
    track_primitive, box_primitive, track = (left, right, left_track) if left_track else (
        right, left, right_track
    )
    if track:
        try:
            bbox = box_primitive.obj.GetBoundingBox()
            x0, y0, w, h = _bbox_mm(bbox)
            x1, y1 = x0 + w, y0 + h
            sx, sy, ex, ey, width = track
            (px, py), (qx, qy), distance = _closest_segment_rect_points(
                sx, sy, ex, ey, x0, y0, x1, y1
            )
            if distance <= width * 0.5 + 1e-9 and _point_in_primitive(
                box_primitive, qx, qy
            ):
                lid = getattr(track_primitive.obj, "GetLayer", lambda: 0)()
                located = _layer_index_for_obj(
                    lid, left, right, ((px + qx) * 0.5, (py + qy) * 0.5)
                )
                if located is not None:
                    return (*located, "track_to_shape")
        except Exception:
            return None
        return None

    try:
        a, b = left.obj.GetBoundingBox(), right.obj.GetBoundingBox()
        ax, ay, aw, ah = _bbox_mm(a)
        bx, by, bw, bh = _bbox_mm(b)
        x0, x1 = max(ax, bx), min(ax + aw, bx + bw)
        y0, y1 = max(ay, by), min(ay + ah, by + bh)
        if x0 <= x1 and y0 <= y1:
            point = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
            if _point_in_primitive(left, *point) and _point_in_primitive(right, *point):
                lid = getattr(left.obj, "GetLayer", lambda: 0)()
                located = _layer_index_for_obj(lid, left, right, point)
                if located is not None:
                    return (*located, "shape_overlap")
    except Exception:
        pass
    return None


def _point_in_primitive(primitive, x_mm, y_mm):
    """Use KiCad's physical hit testing when a primitive exposes it."""
    pos = pcbnew.VECTOR2I(_to_iu(x_mm), _to_iu(y_mm))
    if primitive.kind == "Zone" and hasattr(primitive.obj, "HitTestFilledArea"):
        try:
            layer = getattr(primitive, "layer_id", None)
            if layer is None:
                layer = primitive.obj.GetLayer()
            return bool(primitive.obj.HitTestFilledArea(layer, pos, 1))
        except TypeError:
            try:
                return bool(primitive.obj.HitTestFilledArea(layer, pos))
            except Exception:
                return False
        except Exception:
            return False
    if primitive.kind == "Pad" and hasattr(primitive.obj, "HitTest"):
        try:
            return bool(primitive.obj.HitTest(pos, 1))
        except TypeError:
            try:
                return bool(primitive.obj.HitTest(pos))
            except Exception:
                return False
        except Exception:
            return False
    try:
        bbox = primitive.obj.GetBoundingBox()
        x0, y0, w, h = _bbox_mm(bbox)
        return x0 <= x_mm <= x0 + w and y0 <= y_mm <= y0 + h
    except Exception:
        return False


def _layer_index_for_obj(layer_id, left, right, xy=None):
    """Return grid layer slot and contact coordinates for a primitive pair."""
    # Primitive masks are ordered by config layer list; their occupied layers
    # identify the contact plane without relying on KiCad's numeric layer IDs.
    common = np.flatnonzero(np.any(left.copper_mask, axis=(1, 2)) & np.any(right.copper_mask, axis=(1, 2)))
    if not common.size:
        return None
    if xy is None:
        a = left.obj.GetStart()
        b = right.obj.GetStart()
        xy = ((a.x + b.x) * 0.5e-6, (a.y + b.y) * 0.5e-6)
    return int(common[0]), float(xy[0]), float(xy[1])


def _track_geometry(track):
    """Return straight centreline and width in millimetres, if available."""
    if not all(hasattr(track, name) for name in ("GetStart", "GetEnd", "GetWidth")):
        return None
    try:
        start, end = track.GetStart(), track.GetEnd()
        return (start.x * 1e-6, start.y * 1e-6, end.x * 1e-6, end.y * 1e-6,
                float(track.GetWidth()) * 1e-6)
    except Exception:
        return None



def _track_contact_with_kind(left_track, right_track):
    """Return an accepted track contact point and its diagnostic kind.

    Contact priority is intentionally conservative:
    1. exact routed centreline topology / proper crossing;
    2. <=25 um endpoint snap for generated routing gaps;
    3. endpoint-only physical copper overlap for wide tracks.

    The third rule never performs generic sidewall/interior overlap and is
    disabled for narrow tracks such as the 0.25 mm mains winding.
    """
    endpoint_gap = _minimum_track_endpoint_gap(left_track, right_track)
    contact = _track_centerline_contact(left_track, right_track)
    if contact is not None:
        if (
            endpoint_gap is not None
            and TRACK_TOPOLOGY_TOL_MM < endpoint_gap <= TRACK_ENDPOINT_SNAP_TOL_MM
        ):
            return contact, "track_endpoint_snap"
        return contact, "track_centerline_topology"

    contact = _track_endpoint_copper_overlap_contact(left_track, right_track)
    if contact is not None:
        return contact, "track_endpoint_copper_overlap"
    return None, None


def _minimum_track_endpoint_gap(left_track, right_track):
    left_endpoints = _track_endpoints_mm(left_track)
    right_endpoints = _track_endpoints_mm(right_track)
    if not left_endpoints or not right_endpoints:
        return None
    return min(
        math.hypot(lx - rx, ly - ry)
        for lx, ly in left_endpoints
        for rx, ry in right_endpoints
    )


def _track_endpoint_copper_overlap_contact(left_track, right_track):
    """Return a conservative endpoint-only copper-overlap contact.

    This handles wide generated tracks whose centreline primitives have a
    small gap while their physical copper caps overlap.  It deliberately does
    not connect parallel interiors.  Narrow tracks below 0.35 mm are excluded
    so the planar mains winding keeps its turn/segment separation.
    """
    try:
        left_width = float(left_track.GetWidth()) * 1e-6
        right_width = float(right_track.GetWidth()) * 1e-6
    except Exception:
        return None
    if min(left_width, right_width) < TRACK_ENDPOINT_COPPER_OVERLAP_MIN_WIDTH_MM:
        return None

    physical_limit = 0.5 * (left_width + right_width)
    max_gap = min(TRACK_ENDPOINT_COPPER_OVERLAP_MAX_GAP_MM, physical_limit)
    if max_gap <= TRACK_ENDPOINT_SNAP_TOL_MM:
        return None

    best = None
    pairs = (
        (_track_endpoints_mm(left_track), _track_centerline_segments(right_track)),
        (_track_endpoints_mm(right_track), _track_centerline_segments(left_track)),
    )
    for endpoints, segments in pairs:
        for px, py in endpoints:
            for sx, sy, ex, ey in segments:
                qx, qy = _project_point_segment(px, py, sx, sy, ex, ey)
                distance = math.hypot(px - qx, py - qy)
                if (
                    TRACK_ENDPOINT_SNAP_TOL_MM < distance <= max_gap + 1e-9
                    and (best is None or distance < best[0])
                ):
                    best = (
                        distance,
                        (0.5 * (px + qx), 0.5 * (py + qy)),
                    )
    return best[1] if best is not None else None


def _track_centerline_contact(
    left_track,
    right_track,
    tol=TRACK_TOPOLOGY_TOL_MM,
    endpoint_snap_tol=TRACK_ENDPOINT_SNAP_TOL_MM,
):
    """Return a routed centreline junction, excluding width-only overlap.

    Real KiCad/generated winding geometry can contain tiny gaps between the
    endpoints of consecutive track primitives.  Snap only *original track
    endpoints* within ``endpoint_snap_tol``.  All endpoint-to-interior and
    interior crossing checks remain on the much stricter ``tol`` so nearby
    parallel winding turns cannot connect merely because their copper widths
    overlap.
    """
    left_endpoints = _track_endpoints_mm(left_track)
    right_endpoints = _track_endpoints_mm(right_track)
    if left_endpoints and right_endpoints:
        best = None
        for lx, ly in left_endpoints:
            for rx, ry in right_endpoints:
                distance = math.hypot(lx - rx, ly - ry)
                if distance <= endpoint_snap_tol and (best is None or distance < best[0]):
                    best = (distance, (0.5 * (lx + rx), 0.5 * (ly + ry)))
        if best is not None:
            return best[1]

    for left_segment in _track_centerline_segments(left_track):
        for right_segment in _track_centerline_segments(right_track):
            point = _segment_topology_contact(*left_segment, *right_segment, tol=tol)
            if point is not None:
                return point
    return None


def _track_endpoints_mm(track):
    """Return the two original KiCad track endpoints in millimetres."""
    try:
        start, end = track.GetStart(), track.GetEnd()
        return (
            (float(start.x) * 1e-6, float(start.y) * 1e-6),
            (float(end.x) * 1e-6, float(end.y) * 1e-6),
        )
    except Exception:
        return ()


def _track_centerline_segments(track):
    """Return straight centreline pieces for straight tracks or true circular arcs."""
    try:
        start, end = track.GetStart(), track.GetEnd()
    except Exception:
        return []
    start_xy = (start.x * 1e-6, start.y * 1e-6)
    end_xy = (end.x * 1e-6, end.y * 1e-6)
    if not hasattr(track, "GetMid"):
        return [(start_xy[0], start_xy[1], end_xy[0], end_xy[1])]
    try:
        middle = track.GetMid()
        circle = _circle_from_three_points(start, middle, end)
        if circle is None:
            return [(start_xy[0], start_xy[1], end_xy[0], end_xy[1])]
        cx, cy, radius = circle
        angles = [
            math.atan2(point.y * 1e-6 - cy, point.x * 1e-6 - cx)
            for point in (start, middle, end)
        ]
        tau = 2.0 * math.pi
        ccw = (angles[2] - angles[0]) % tau
        middle_ccw = (angles[1] - angles[0]) % tau
        span = ccw if middle_ccw <= ccw else -((angles[0] - angles[2]) % tau)
        # Contact topology only needs a faithful centreline. Cap chord error near 1 um.
        chord_error = 0.001
        max_angle = math.pi / 32.0
        if radius > chord_error:
            max_angle = min(
                max_angle,
                2.0 * math.acos(max(-1.0, 1.0 - chord_error / radius)),
            )
        steps = max(2, int(math.ceil(abs(span) / max(max_angle, 1e-6))))
        points = []
        for index in range(steps + 1):
            angle = angles[0] + span * index / steps
            points.append((
                cx + radius * math.cos(angle),
                cy + radius * math.sin(angle),
            ))
        points[0] = start_xy
        points[-1] = end_xy
        return [
            (left[0], left[1], right[0], right[1])
            for left, right in zip(points, points[1:])
        ]
    except Exception:
        return [(start_xy[0], start_xy[1], end_xy[0], end_xy[1])]


def _segment_topology_contact(ax, ay, bx, by, cx, cy, dx, dy, tol=1e-6):
    """Return endpoint/T/crossing contact between centrelines, not width overlap."""
    # Shared endpoints or one endpoint lying on the other routed centreline.
    for px, py in ((ax, ay), (bx, by)):
        qx, qy = _project_point_segment(px, py, cx, cy, dx, dy)
        if math.hypot(px - qx, py - qy) <= tol:
            return ((px + qx) * 0.5, (py + qy) * 0.5)
    for px, py in ((cx, cy), (dx, dy)):
        qx, qy = _project_point_segment(px, py, ax, ay, bx, by)
        if math.hypot(px - qx, py - qy) <= tol:
            return ((px + qx) * 0.5, (py + qy) * 0.5)

    # Proper interior crossing of two non-parallel centreline segments.
    vx, vy = bx - ax, by - ay
    wx, wy = dx - cx, dy - cy
    det = vx * wy - vy * wx
    if abs(det) <= 1e-15:
        return None
    t = ((cx - ax) * wy - (cy - ay) * wx) / det
    u = ((cx - ax) * vy - (cy - ay) * vx) / det
    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        return (ax + t * vx, ay + t * vy)
    return None

def _closest_segment_points(ax, ay, bx, by, cx, cy, dx, dy):
    """Return closest points and distance between two finite 2D segments."""
    candidates = []
    for x, y in ((ax, ay), (bx, by)):
        qx, qy = _project_point_segment(x, y, cx, cy, dx, dy)
        candidates.append(((x, y), (qx, qy)))
    for x, y in ((cx, cy), (dx, dy)):
        qx, qy = _project_point_segment(x, y, ax, ay, bx, by)
        candidates.append(((qx, qy), (x, y)))
    vx, vy, wx, wy = bx - ax, by - ay, dx - cx, dy - cy
    det = vx * wy - vy * wx
    if abs(det) > 1e-15:
        t = ((cx - ax) * wy - (cy - ay) * wx) / det
        u = ((cx - ax) * vy - (cy - ay) * vx) / det
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            point = (ax + t * vx, ay + t * vy)
            return point, point, 0.0
    return min(candidates, key=lambda pair: math.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1])) + (
        min(math.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]) for pair in candidates),
    )


def _project_point_segment(px, py, sx, sy, ex, ey):
    vx, vy = ex - sx, ey - sy
    length_sq = vx * vx + vy * vy
    t = 0.0 if length_sq == 0.0 else min(1.0, max(0.0, ((px - sx) * vx + (py - sy) * vy) / length_sq))
    return sx + t * vx, sy + t * vy


def _closest_segment_rect_points(sx, sy, ex, ey, x0, y0, x1, y1):
    """Return closest points and distance between a segment and rectangle."""
    intersections = []
    if x0 <= sx <= x1 and y0 <= sy <= y1:
        return (sx, sy), (sx, sy), 0.0
    if x0 <= ex <= x1 and y0 <= ey <= y1:
        return (ex, ey), (ex, ey), 0.0
    vx, vy = ex - sx, ey - sy
    if abs(vx) > 1e-15:
        for edge_x in (x0, x1):
            t = (edge_x - sx) / vx
            edge_y = sy + t * vy
            if 0.0 <= t <= 1.0 and y0 <= edge_y <= y1:
                intersections.append((t, (edge_x, edge_y)))
    if abs(vy) > 1e-15:
        for edge_y in (y0, y1):
            t = (edge_y - sy) / vy
            edge_x = sx + t * vx
            if 0.0 <= t <= 1.0 and x0 <= edge_x <= x1:
                intersections.append((t, (edge_x, edge_y)))
    if intersections:
        point = min(intersections, key=lambda item: item[0])[1]
        return point, point, 0.0
    candidates = []
    for px, py in ((sx, sy), (ex, ey)):
        qx, qy = min(max(px, x0), x1), min(max(py, y0), y1)
        candidates.append(((px, py), (qx, qy)))
    for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        qx, qy = _project_point_segment(px, py, sx, sy, ex, ey)
        candidates.append(((qx, qy), (px, py)))
    a, b = min(candidates, key=lambda pair: math.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]))
    return a, b, math.hypot(a[0] - b[0], a[1] - b[1])


def _build_net_tie_edges(
    net_tie_links: List[Tuple[Any, Any]],
    node_ids_by_net: Dict[str, np.ndarray],
    config: ElectricalConfig,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Bridge the mapped copper nodes of each declared KiCad net tie."""
    edge_i = []
    edge_j = []
    errors = []
    for left_pad, right_pad in net_tie_links:
        left_key = net_key_from_obj(left_pad)[0]
        right_key = net_key_from_obj(right_pad)[0]
        left_ids = node_ids_by_net.get(left_key)
        right_ids = node_ids_by_net.get(right_key)
        left_nodes = _owned_pad_nodes(left_pad, left_ids, config) if left_ids is not None else np.empty(0, dtype=np.int64)
        right_nodes = _owned_pad_nodes(right_pad, right_ids, config) if right_ids is not None else np.empty(0, dtype=np.int64)
        if not left_nodes.size or not right_nodes.size:
            errors.append(
                f"Declared net-tie pads {left_key} and {right_key} do not both map to copper."
            )
            continue
        edge_i.append(int(np.min(left_nodes)))
        edge_j.append(int(np.min(right_nodes)))
    return (
        np.asarray(edge_i, dtype=np.int64),
        np.asarray(edge_j, dtype=np.int64),
        errors,
    )


def _pad_copper_layer_ids(pad: Any, copper_ids: List[int]) -> List[int]:
    """Return copper layer IDs occupied by a pad."""
    try:
        layer_set = pad.GetLayerSet()
        layer_ids = [lid for lid in copper_ids if layer_set.Contains(lid)]
        if layer_ids:
            return layer_ids
    except Exception:
        pass
    try:
        layer_id = pad.GetLayer()
        if layer_id in copper_ids:
            return [layer_id]
    except Exception:
        pass
    return []


def _pad_node_indices(pad: Any, node_ids: np.ndarray, config: ElectricalConfig) -> np.ndarray:
    """Return electrical node IDs under a pad."""
    layers = []
    lid_to_idx = {lid: idx for idx, lid in enumerate(config.copper_ids)}
    if _is_pth_pad(pad):
        layers = list(range(len(config.copper_ids)))
    else:
        layers = [
            lid_to_idx[layer_id]
            for layer_id in _pad_copper_layer_ids(pad, config.copper_ids)
            if layer_id in lid_to_idx
        ]
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
    layer_ids = _pad_copper_layer_ids(pad, config.copper_ids)
    return _layers_label(layer_ids, config)


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


def _new_local_primitive(
    obj: Any,
    kind: str,
    bbox: Any,
    config: ElectricalConfig,
    padding_cells: int = 1,
    layer_id: Optional[int] = None,
):
    """Allocate a primitive raster only for its local grid window.

    The previous implementation allocated six full-board boolean arrays for
    every KiCad primitive.  Highly segmented windings therefore consumed tens
    of gigabytes.  This helper keeps the same layer axis and electrical
    semantics while bounding the row/column dimensions to the primitive.
    """
    rs, re, cs, ce = _bbox_indices(bbox, config)
    pad = max(0, int(padding_cells))
    rs = max(0, rs - pad)
    re = min(config.rows, re + pad)
    cs = max(0, cs - pad)
    ce = min(config.cols, ce + pad)
    if rs >= re or cs >= ce:
        return None, None

    layer_count = len(config.copper_ids)
    rows = re - rs
    cols = ce - cs
    shape = (layer_count, rows, cols)
    primitive = _PrimitiveRaster(
        obj=obj,
        kind=kind,
        layer_id=layer_id,
        row0=rs,
        col0=cs,
        copper_mask=np.zeros(shape, dtype=bool),
        via_mask=np.zeros((rows, cols), dtype=bool),
        connect_right=np.zeros(
            (layer_count, rows, max(cols - 1, 0)), dtype=bool
        ),
        connect_down=np.zeros(
            (layer_count, max(rows - 1, 0), cols), dtype=bool
        ),
        connect_down_right=np.zeros(
            (layer_count, max(rows - 1, 0), max(cols - 1, 0)), dtype=bool
        ),
        connect_down_left=np.zeros(
            (layer_count, max(rows - 1, 0), max(cols - 1, 0)), dtype=bool
        ),
    )
    local_config = SimpleNamespace(
        copper_ids=config.copper_ids,
        rows=rows,
        cols=cols,
        x_min=config.x_min + cs * config.res,
        y_min=config.y_min + rs * config.res,
        res=config.res,
        t_cu=config.t_cu,
        rho_cu=config.rho_cu,
        via_resistance_ohm=config.via_resistance_ohm,
        balance_abs_tol=config.balance_abs_tol,
        balance_rel_tol=config.balance_rel_tol,
        layer_names=config.layer_names,
    )
    return primitive, local_config


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


def _fill_pad(mask, layer_idx, pad, layer_id, config):
    """Rasterize a native pad shape with KiCad's own point hit testing."""
    bbox = pad.GetBoundingBox()
    rs, re, cs, ce = _bbox_indices(bbox, config)
    hit_test = getattr(pad, "HitTest", None)
    if not callable(hit_test):
        _fill_bbox_3d(mask, layer_idx, bbox, config)
        return
    offsets = ((0.5, 0.5), (0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0))
    for row in range(rs, re):
        for col in range(cs, ce):
            for dx, dy in offsets:
                point = pcbnew.VECTOR2I(
                    _to_iu(config.x_min + (col + dx) * config.res),
                    _to_iu(config.y_min + (row + dy) * config.res),
                )
                try:
                    inside = bool(hit_test(point, 1))
                except TypeError:
                    try:
                        inside = bool(hit_test(point))
                    except Exception:
                        inside = False
                except Exception:
                    inside = False
                if inside:
                    mask[layer_idx, row, col] = True
                    break


def _merge_primitive(raster: _RawNetRaster, primitive: _PrimitiveRaster):
    """Merge a local primitive raster into the full raw-net union masks."""
    r0, c0 = primitive.row0, primitive.col0
    r1, c1 = r0 + primitive.rows, c0 + primitive.cols

    raster.copper_mask[:, r0:r1, c0:c1] |= primitive.copper_mask
    raster.via_mask[r0:r1, c0:c1] |= primitive.via_mask

    if primitive.cols > 1:
        raster.connect_right[:, r0:r1, c0:c1 - 1] |= primitive.connect_right
    if primitive.rows > 1:
        raster.connect_down[:, r0:r1 - 1, c0:c1] |= primitive.connect_down
    if primitive.rows > 1 and primitive.cols > 1:
        raster.connect_down_right[:, r0:r1 - 1, c0:c1 - 1] |= (
            primitive.connect_down_right
        )
        raster.connect_down_left[:, r0:r1 - 1, c0:c1 - 1] |= (
            primitive.connect_down_left
        )
    raster.primitives.append(primitive)


def _connect_full_shape(primitive: _PrimitiveRaster, layer_idx: int):
    """Connect neighbouring cells within one continuous pad or filled zone."""
    mask = primitive.copper_mask[layer_idx]
    primitive.connect_right[layer_idx] |= mask[:, :-1] & mask[:, 1:]
    primitive.connect_down[layer_idx] |= mask[:-1, :] & mask[1:, :]
    primitive.connect_down_right[layer_idx] |= mask[:-1, :-1] & mask[1:, 1:]
    primitive.connect_down_left[layer_idx] |= mask[:-1, 1:] & mask[1:, :-1]


def _connect_zone_boundaries(primitive, layer_idx, layer_id, zone, config):
    """Permit zone edges only where KiCad reports filled copper on the boundary."""
    hit_test = getattr(zone, "HitTestFilledArea", None)
    if not callable(hit_test):
        return
    mask = primitive.copper_mask[layer_idx]
    right_rows, right_cols = np.nonzero(mask[:, :-1] & mask[:, 1:])
    for row, col in zip(right_rows, right_cols):
        point = pcbnew.VECTOR2I(
            _to_iu(config.x_min + (col + 1) * config.res),
            _to_iu(config.y_min + (row + 0.5) * config.res),
        )
        try:
            hit = hit_test(layer_id, point, 1)
        except TypeError:
            hit = hit_test(layer_id, point)
        except Exception:
            hit = False
        if hit:
            primitive.connect_right[layer_idx, row, col] = True
    down_rows, down_cols = np.nonzero(mask[:-1, :] & mask[1:, :])
    for row, col in zip(down_rows, down_cols):
        point = pcbnew.VECTOR2I(
            _to_iu(config.x_min + (col + 0.5) * config.res),
            _to_iu(config.y_min + (row + 1) * config.res),
        )
        try:
            hit = hit_test(layer_id, point, 1)
        except TypeError:
            hit = hit_test(layer_id, point)
        except Exception:
            hit = False
        if hit:
            primitive.connect_down[layer_idx, row, col] = True
    # Diagonal zone steps need a filled shared-corner point.
    for candidates, target in (
        (mask[:-1, :-1] & mask[1:, 1:], primitive.connect_down_right),
        (mask[:-1, 1:] & mask[1:, :-1], primitive.connect_down_left),
    ):
        rr, cc = np.nonzero(candidates)
        for row, col in zip(rr, cc):
            point = pcbnew.VECTOR2I(
                _to_iu(config.x_min + (col + 1) * config.res),
                _to_iu(config.y_min + (row + 1) * config.res),
            )
            try:
                hit = hit_test(layer_id, point, 1)
            except TypeError:
                hit = hit_test(layer_id, point)
            except Exception:
                hit = False
            if hit:
                target[layer_idx, row, col] = True


def _fill_track(raster: _PrimitiveRaster, layer_idx: int, track: Any, config: ElectricalConfig):
    """Rasterize straight tracks and flatten circular arcs to bounded capsules."""
    if hasattr(track, "GetMid"):
        try:
            start, middle, end = track.GetStart(), track.GetMid(), track.GetEnd()
            circle = _circle_from_three_points(start, middle, end)
            if circle is not None:
                cx, cy, radius = circle
                angles = [math.atan2(p.y * 1e-6 - cy, p.x * 1e-6 - cx)
                          for p in (start, middle, end)]
                tau = 2.0 * math.pi
                ccw = (angles[2] - angles[0]) % tau
                middle_ccw = (angles[1] - angles[0]) % tau
                span = ccw if middle_ccw <= ccw else -((angles[0] - angles[2]) % tau)
                tolerance = max(1e-6, min(0.01, config.res / 8.0))
                max_angle = math.pi / 8.0
                if radius > tolerance:
                    max_angle = min(max_angle, 2.0 * math.acos(max(-1.0, 1.0 - tolerance / radius)))
                steps = max(2, int(math.ceil(abs(span) / max(max_angle, 1e-6))))
                points = []
                width = int(track.GetWidth())
                for index in range(steps + 1):
                    angle = angles[0] + span * index / steps
                    points.append(SimpleNamespace(
                        x=int(round((cx + radius * math.cos(angle)) * 1e6)),
                        y=int(round((cy + radius * math.sin(angle)) * 1e6)),
                    ))
                for left, right in zip(points, points[1:]):
                    segment = SimpleNamespace(
                        GetStart=lambda p=left: p,
                        GetEnd=lambda p=right: p,
                        GetWidth=lambda w=width: w,
                        GetBoundingBox=track.GetBoundingBox,
                    )
                    _fill_straight_track(raster, layer_idx, segment, config)
                return
        except Exception:
            pass
    _fill_straight_track(raster, layer_idx, track, config)


def _fill_straight_track(raster: _PrimitiveRaster, layer_idx: int, track: Any, config: ElectricalConfig):
    """Rasterize one physical-width straight track segment."""
    mask = raster.copper_mask
    if not all(hasattr(track, attr) for attr in ("GetStart", "GetEnd", "GetWidth")):
        _fill_bbox_3d(mask, layer_idx, track.GetBoundingBox(), config)
        return
    try:
        start = track.GetStart()
        end = track.GetEnd()
        width_mm = float(track.GetWidth()) * 1e-6
    except Exception:
        _fill_bbox_3d(mask, layer_idx, track.GetBoundingBox(), config)
        return
    if width_mm <= 0.0:
        return

    sx, sy = start.x * 1e-6, start.y * 1e-6
    ex, ey = end.x * 1e-6, end.y * 1e-6
    radius = 0.5 * width_mm
    eps = max(width_mm * 1e-8, 1e-10)
    x0, x1 = min(sx, ex) - radius, max(sx, ex) + radius
    y0, y1 = min(sy, ey) - radius, max(sy, ey) + radius
    cs = max(0, int(np.floor((x0 - config.x_min) / config.res)))
    ce = min(config.cols, int(np.ceil((x1 - config.x_min) / config.res)))
    rs = max(0, int(np.floor((y0 - config.y_min) / config.res)))
    re = min(config.rows, int(np.ceil((y1 - config.y_min) / config.res)))
    if rs >= re or cs >= ce:
        return

    x_left = config.x_min + np.arange(cs, ce, dtype=np.float64) * config.res
    y_top = config.y_min + np.arange(rs, re, dtype=np.float64) * config.res
    left, top = np.meshgrid(x_left, y_top)
    right, bottom = left + config.res, top + config.res
    local = _segment_rect_distance_sq(sx, sy, ex, ey, left, top, right, bottom) <= (radius + eps) ** 2
    mask[layer_idx, rs:re, cs:ce] |= local

    local_r, local_c = np.nonzero(local[:-1, :-1] & local[1:, 1:])
    r, c = rs + local_r, cs + local_c
    corner_x = config.x_min + (c + 1) * config.res
    corner_y = config.y_min + (r + 1) * config.res
    connected = _point_segment_distance_sq(corner_x, corner_y, sx, sy, ex, ey) < (radius - eps) ** 2
    raster.connect_down_right[layer_idx, r[connected], c[connected]] = True

    local_r, local_c = np.nonzero(local[:-1, 1:] & local[1:, :-1])
    r, c = rs + local_r, cs + local_c
    corner_x = config.x_min + (c + 1) * config.res
    corner_y = config.y_min + (r + 1) * config.res
    connected = _point_segment_distance_sq(corner_x, corner_y, sx, sy, ex, ey) < (radius - eps) ** 2
    raster.connect_down_left[layer_idx, r[connected], c[connected]] = True

    # Test the complete shared edge against the track capsule.
    local_r, local_c = np.nonzero(local[:, :-1] & local[:, 1:])
    r, c = rs + local_r, cs + local_c
    bx = config.x_min + (c + 1) * config.res
    by0 = config.y_min + r * config.res
    by1 = by0 + config.res
    right_hits = _segment_vertical_boundary_hit(
        sx, sy, ex, ey, bx, by0, by1, radius + eps
    )
    raster.connect_right[layer_idx, r, c] |= right_hits

    local_r, local_c = np.nonzero(local[:-1, :] & local[1:, :])
    r, c = rs + local_r, cs + local_c
    by = config.y_min + (r + 1) * config.res
    bx0 = config.x_min + c * config.res
    bx1 = bx0 + config.res
    down_hits = _segment_horizontal_boundary_hit(
        sx, sy, ex, ey, by, bx0, bx1, radius + eps
    )
    raster.connect_down[layer_idx, r, c] |= down_hits


def _segment_vertical_boundary_hit(sx, sy, ex, ey, x, y0, y1, radius):
    """Return whether a track capsule crosses vertical cell boundaries."""
    vx, vy = ex - sx, ey - sy
    intersects = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y0)), dtype=bool)
    if abs(vx) > 1e-15:
        t = (x - sx) / vx
        y = sy + t * vy
        intersects = (t >= 0.0) & (t <= 1.0) & (y >= y0) & (y <= y1)
    d2 = np.minimum(
        _point_segment_distance_sq(x, y0, sx, sy, ex, ey),
        _point_segment_distance_sq(x, y1, sx, sy, ex, ey),
    )
    for px, py in ((sx, sy), (ex, ey)):
        dy = np.maximum(np.maximum(y0 - py, 0.0), py - y1)
        d2 = np.minimum(d2, (x - px) ** 2 + dy ** 2)
    return intersects | (d2 <= radius * radius)


def _segment_horizontal_boundary_hit(sx, sy, ex, ey, y, x0, x1, radius):
    """Return whether a track capsule crosses horizontal cell boundaries."""
    vx, vy = ex - sx, ey - sy
    intersects = np.zeros(np.broadcast_shapes(np.shape(x0), np.shape(y)), dtype=bool)
    if abs(vy) > 1e-15:
        t = (y - sy) / vy
        x = sx + t * vx
        intersects = (t >= 0.0) & (t <= 1.0) & (x >= x0) & (x <= x1)
    d2 = np.minimum(
        _point_segment_distance_sq(x0, y, sx, sy, ex, ey),
        _point_segment_distance_sq(x1, y, sx, sy, ex, ey),
    )
    for px, py in ((sx, sy), (ex, ey)):
        dx = np.maximum(np.maximum(x0 - px, 0.0), px - x1)
        d2 = np.minimum(d2, dx ** 2 + (y - py) ** 2)
    return intersects | (d2 <= radius * radius)


def _point_segment_distance_sq(px, py, sx, sy, ex, ey):
    """Squared distance from a point to a finite segment."""
    vx, vy = ex - sx, ey - sy
    length_sq = vx * vx + vy * vy
    if length_sq == 0.0:
        return (px - sx) ** 2 + (py - sy) ** 2
    t = np.clip(((px - sx) * vx + (py - sy) * vy) / length_sq, 0.0, 1.0)
    return (px - (sx + t * vx)) ** 2 + (py - (sy + t * vy)) ** 2


def _segment_rect_distance_sq(sx, sy, ex, ey, left, top, right, bottom):
    """Squared distance between a segment and an axis-aligned cell rectangle."""
    dx, dy = ex - sx, ey - sy
    t_min = np.zeros_like(left)
    t_max = np.ones_like(left)
    intersects = np.ones_like(left, dtype=bool)
    for origin, delta, low, high in ((sx, dx, left, right), (sy, dy, top, bottom)):
        if abs(delta) < 1e-15:
            intersects &= (origin >= low) & (origin <= high)
        else:
            t0, t1 = (low - origin) / delta, (high - origin) / delta
            t_min = np.maximum(t_min, np.minimum(t0, t1))
            t_max = np.minimum(t_max, np.maximum(t0, t1))
    intersects &= t_min <= t_max

    def point_rect_distance_sq(px, py):
        return np.maximum(np.maximum(left - px, 0.0), px - right) ** 2 + np.maximum(
            np.maximum(top - py, 0.0), py - bottom
        ) ** 2

    distance_sq = np.minimum(
        point_rect_distance_sq(sx, sy), point_rect_distance_sq(ex, ey)
    )
    for x, y in ((left, top), (right, top), (left, bottom), (right, bottom)):
        distance_sq = np.minimum(
            distance_sq, _point_segment_distance_sq(x, y, sx, sy, ex, ey)
        )
    return np.where(intersects, 0.0, distance_sq)


def _fill_zone(mask: np.ndarray, layer_idx: int, lid: int, zone: Any, config: ElectricalConfig):
    """Rasterize a filled copper zone with KiCad hit-testing where available."""
    bbox = zone.GetBoundingBox()
    rs, re, cs, ce = _bbox_indices(bbox, config)
    if rs >= re or cs >= ce:
        return

    x_values = np.asarray(
        (
            config.x_min
            + (np.arange(cs, ce, dtype=np.float64) + 0.5) * config.res
        ) * 1e6,
        dtype=np.int64,
    )
    y_values = np.asarray(
        (
            config.y_min
            + (np.arange(rs, re, dtype=np.float64) + 0.5) * config.res
        ) * 1e6,
        dtype=np.int64,
    )
    zone_mask = np.zeros((re - rs, ce - cs), dtype=bool)
    if _fill_zone_mask_polygons(
        zone_mask, None, x_values, y_values, lid, zone
    ):
        mask[layer_idx, rs:re, cs:ce] |= zone_mask
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
