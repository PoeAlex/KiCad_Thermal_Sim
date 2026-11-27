"""
Electrical DC solver for trace and copper-area Joule heating.

This module builds a net-isolated resistor network on the thermal grid. It
solves the copper potential for configured pad currents and converts the edge
losses into a thermal heat-source vector.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

import pcbnew

from .geometry_mapper import (
    GeometryCancelledError,
    _ProgressEmitter,
    _arc_parameters,
    _extract_pth_drill_mm,
    _extract_via_drill_mm,
    _fill_zone_mask_polygons,
    _is_rule_or_keepout_zone,
    _mark_centerline_points,
    _pad_orientation_radians,
    _pad_shape_center_mm,
    compute_via_copper_area_m2,
    get_pad_copper_layer_indices,
    get_pad_grid_mask,
    get_pad_primary_layer_id,
    normalize_via_fill,
    select_via_grid_cells,
)
from .stackup_parser import parse_stackup_from_board_file

_ELECTRICAL_ITERATIVE_NODE_THRESHOLD = 150_000
_ELECTRICAL_ITERATIVE_RTOL = 1e-8
_ELECTRICAL_ITERATIVE_MAXITER = 4000


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
class EquipotentialBus:
    """
    Multi-pad equipotential bus group sharing current naturally by copper resistance.

    Parameters
    ----------
    name : str
        Human-readable bus group name.
    net_name : str
        KiCad net name.
    net_code : int
        KiCad net code.
    pads : list of (object, str)
        List of ``(kicad_pad_obj, pad_display_name)`` belonging to the bus.
    total_current_a : float
        Total bus current in amperes. Positive enters PCB, negative leaves PCB.
    """

    name: str
    net_name: str
    net_code: int
    pads: List[Tuple[Any, str]]
    total_current_a: float


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
    via_plating_um : float, optional
        Via/PTH barrel copper plating thickness in micrometers (default 25.0).
    via_fill : str, optional
        Via fill type (``'plated'`` or ``'copper'``, default ``'plated'``).
    ambient_c : float, optional
        Reference ambient temperature in Celsius for copper TCR (default 20.0).
    gap_m : sequence of float, optional
        Dielectric thicknesses between adjacent copper layers in meters.
    equipotential_pads : bool, optional
        Contract all copper cells of each single pad into an equipotential supernode (default True).
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
    via_plating_um: float = 25.0
    via_fill: str = "plated"
    ambient_c: float = 20.0
    gap_m: Optional[Any] = None
    equipotential_pads: bool = True


def _effective_rho_cu(config: ElectricalConfig) -> float:
    """Return temperature-adjusted copper resistivity ``rho(ambient_c)`` (ohm*m)."""
    rho_eff = float(config.rho_cu) * (
        1.0 + 0.00393 * (float(config.ambient_c) - 20.0)
    )
    return max(rho_eff, 1e-20)


def _resolve_gap_m(board: Any, config: ElectricalConfig) -> List[float]:
    """Resolve adjacent-layer dielectric gaps in meters."""
    layer_count = len(config.copper_ids)
    if layer_count <= 1:
        return []
    num_gaps = layer_count - 1
    if config.gap_m is not None:
        try:
            gaps = [max(float(g), 1e-6) for g in config.gap_m]
            if len(gaps) == num_gaps:
                return gaps
            if gaps:
                return (gaps + [gaps[-1]] * num_gaps)[:num_gaps]
        except Exception:
            pass
    try:
        stack = parse_stackup_from_board_file(board)
        if isinstance(stack, dict) and not stack.get("error"):
            raw_gaps = stack.get("dielectric_gaps_mm") or []
            if len(raw_gaps) == num_gaps and all(
                isinstance(g, (int, float)) and g > 0 for g in raw_gaps
            ):
                return [max(float(g) * 1e-3, 1e-6) for g in raw_gaps]
            total_mm = stack.get("board_thickness_mm")
            if isinstance(total_mm, (int, float)) and total_mm > 0:
                fb = max(float(total_mm) * 1e-3 / num_gaps, 1e-6)
                return [fb] * num_gaps
    except Exception:
        pass
    fb = 1.6e-3 / num_gaps
    return [fb] * num_gaps


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
    bus_name: Optional[str] = None


@dataclass
class ElectricalViaDiagnostics:
    """Solved current and Joule-power diagnostics for one physical via."""

    net_name: str
    x_mm: float
    y_mm: float
    drill_mm: float
    layers: str
    current_a: float
    power_w: float


@dataclass
class _ViaRecord:
    """Internal per-via grid footprint and reference barrel conductances."""

    x_mm: float
    y_mm: float
    drill_mm: float
    layers: str
    rs: int
    re: int
    cs: int
    ce: int
    mask_v: np.ndarray
    min_l: int
    max_l: int
    g_per_gap: Tuple[float, ...]


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
    top_vias: List[ElectricalViaDiagnostics] = field(default_factory=list)
    max_via_current_a: float = 0.0
    current_density_a_mm2: Optional[np.ndarray] = None
    max_current_density_a_mm2: float = 0.0


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
    current_density_map : np.ndarray or None
        Maximum current density (A/mm^2) across solved nets with shape ``(L, R, C)``.
    """

    q_joule: np.ndarray
    net_summaries: List[ElectricalNetSummary]
    warnings: List[str]
    errors: List[str]
    current_density_map: Optional[np.ndarray] = None

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
    progress_callback: Optional[Callable[[int, int, str], Optional[bool]]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    equipotential_buses: Optional[List[EquipotentialBus]] = None,
    temperature_field: Optional[np.ndarray] = None,
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
    progress_callback : callable, optional
        Callback ``(done, total, message) -> Optional[bool]``. Returning False cancels.
    cancel_check : callable, optional
        Callable returning True to cancel.
    equipotential_buses : list of EquipotentialBus, optional
        Multi-pad equipotential bus groups sharing current naturally by copper resistance.
    temperature_field : np.ndarray, optional
        Optional 3D ``(L, R, C)`` board temperature field in Celsius for local copper TCR.

    Returns
    -------
    ElectricalResult
        Electrical diagnostics and the Joule heat vector.
    """
    tracker = _ProgressEmitter(progress_callback, cancel_check)
    tracker.check_cancel(force=True)

    layer_count = len(config.copper_ids)
    total_nodes = layer_count * config.rows * config.cols
    q_total = np.zeros(total_nodes, dtype=np.float64)
    errors: List[str] = []
    warnings: List[str] = []
    summaries: List[ElectricalNetSummary] = []

    sigma_grid: Optional[np.ndarray] = None
    if temperature_field is not None:
        t_arr = np.asarray(temperature_field, dtype=np.float64)
        if t_arr.ndim == 1 and t_arr.size == total_nodes:
            t_arr = t_arr.reshape((layer_count, config.rows, config.cols))
        elif t_arr.shape != (layer_count, config.rows, config.cols):
            raise ValueError(
                f"temperature_field shape {t_arr.shape} does not match expected "
                f"{(layer_count, config.rows, config.cols)}."
            )
        rho_local = np.maximum(
            float(config.rho_cu) * (1.0 + 0.00393 * (t_arr - 20.0)),
            1e-20,
        )
        sigma_grid = 1.0 / rho_local

    active_terms = [t for t in terminals if abs(float(t.current_a)) > 0.0]
    active_buses = [b for b in (equipotential_buses or []) if abs(float(b.total_current_a)) > 0.0]
    if not active_terms and not active_buses:
        return ElectricalResult(q_total, summaries, warnings, errors)

    terms_by_net: Dict[str, List[CurrentTerminal]] = {}
    buses_by_net: Dict[str, List[EquipotentialBus]] = {}
    net_display: Dict[str, str] = {}
    for term in active_terms:
        key = net_key_from_values(term.net_code, term.net_name)
        terms_by_net.setdefault(key, []).append(term)
        net_display[key] = term.net_name or key
        if key == "NO_NET":
            errors.append(f"{term.name}: current terminal has no KiCad net.")
    for bus in active_buses:
        key = net_key_from_values(bus.net_code, bus.net_name)
        buses_by_net.setdefault(key, []).append(bus)
        net_display.setdefault(key, bus.net_name or key)
        if bus.net_name and net_display.get(key) == key:
            net_display[key] = bus.net_name
        if key == "NO_NET":
            errors.append(f"{bus.name}: equipotential bus has no KiCad net.")
        if not bus.pads:
            errors.append(f"{bus.name}: equipotential bus has no pads.")

    relevant_nets = set(terms_by_net) | set(buses_by_net)
    for key in relevant_nets:
        terms = terms_by_net.get(key, [])
        buses = buses_by_net.get(key, [])
        total = float(sum(t.current_a for t in terms) + sum(b.total_current_a for b in buses))
        total_abs = float(sum(abs(t.current_a) for t in terms) + sum(abs(b.total_current_a) for b in buses))
        tol = max(config.balance_abs_tol, config.balance_rel_tol * total_abs)
        if abs(total) > tol:
            errors.append(
                f"Net {net_display.get(key, key)} is not current-balanced: "
                f"sum(I)={total:.9g} A, tolerance={tol:.3g} A."
            )

    if errors:
        return ElectricalResult(q_total, summaries, warnings, errors)

    mask_progress_cb = None
    if progress_callback is not None:
        def _mask_progress(done, total, msg):
            scaled = int(50 * done / max(1, total))
            tracker.emit(scaled, 100, msg, force=True)
        mask_progress_cb = _mask_progress

    (
        net_masks,
        via_masks,
        via_g_maps,
        width_scales,
        via_records,
        primitive_summaries,
        collision_count,
    ) = _build_relevant_net_bundle(
        board,
        config,
        relevant_nets,
        progress_callback=mask_progress_cb,
        cancel_check=cancel_check,
        compute_via_conductance=True,
    )
    if collision_count:
        errors.append(
            "Copper cells from multiple active nets overlap at the current "
            f"resolution ({collision_count} grid cells). Use a finer resolution."
        )
        return ElectricalResult(q_total, summaries, warnings, errors)

    ordered_nets = list(dict.fromkeys(list(terms_by_net.keys()) + list(buses_by_net.keys())))
    total_nets = max(1, len(ordered_nets))
    current_density_map: Optional[np.ndarray] = None
    for net_idx, key in enumerate(ordered_nets):
        terms = terms_by_net.get(key, [])
        buses = buses_by_net.get(key, [])
        net_label = net_display.get(key, key)
        tracker.emit(
            50 + int(50 * net_idx / total_nets),
            100,
            f"Solving electrical net {net_idx + 1} / {total_nets} ({net_label})...",
            force=True,
        )
        copper_mask = net_masks.get(key)
        if copper_mask is None or not np.any(copper_mask):
            errors.append(f"Net {net_label} has no mapped copper.")
            continue

        result = _solve_one_net(
            key,
            net_label,
            copper_mask,
            via_masks.get(key),
            terms,
            config,
            primitive_summaries.get(key, []),
            cancel_check=cancel_check,
            via_g_map=via_g_maps.get(key),
            buses=buses,
            width_scale=width_scales.get(key),
            via_records=via_records.get(key, []),
            sigma_grid=sigma_grid,
        )
        q_total += result.q_joule
        summaries.extend(result.net_summaries)
        warnings.extend(result.warnings)
        errors.extend(result.errors)
        if result.current_density_map is not None:
            if current_density_map is None:
                current_density_map = result.current_density_map.copy()
            else:
                np.maximum(current_density_map, result.current_density_map, out=current_density_map)

    tracker.emit(100, 100, "Electrical solve complete.", force=True)
    return ElectricalResult(
        q_total,
        summaries,
        warnings,
        errors,
        current_density_map=current_density_map,
    )


def _merge_duplicate_terminals(terms: List[CurrentTerminal]) -> List[CurrentTerminal]:
    """Merge multiple CurrentTerminal entries targeting the same pad object by summing currents."""
    merged: Dict[int, CurrentTerminal] = {}
    order: List[int] = []
    for term in terms:
        pid = id(term.pad)
        if pid not in merged:
            merged[pid] = CurrentTerminal(
                pad=term.pad,
                name=term.name,
                net_name=term.net_name,
                net_code=term.net_code,
                current_a=float(term.current_a),
            )
            order.append(pid)
        else:
            merged[pid].current_a += float(term.current_a)
    return [merged[pid] for pid in order]


def _solve_one_net(
    net_key: str,
    net_name: str,
    copper_mask: np.ndarray,
    via_mask: Optional[np.ndarray],
    terms: List[CurrentTerminal],
    config: ElectricalConfig,
    primitive_diagnostics: Optional[List[ElectricalPrimitiveDiagnostics]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    via_g_map: Optional[np.ndarray] = None,
    buses: Optional[List[EquipotentialBus]] = None,
    width_scale: Optional[np.ndarray] = None,
    via_records: Optional[List[_ViaRecord]] = None,
    sigma_grid: Optional[np.ndarray] = None,
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

    edge_i, edge_j, edge_g, via_edge_count = _build_net_edges(
        copper_mask,
        via_mask,
        node_ids,
        config,
        via_g_map=via_g_map,
        width_scale=width_scale,
        sigma_grid=sigma_grid,
    )

    merged_terms = _merge_duplicate_terminals(terms)
    active_buses = list(buses or [])

    parent = np.arange(node_count, dtype=np.int64)
    touched_nodes: set = set()

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def _union_nodes(nodes: np.ndarray) -> None:
        if nodes.size == 0:
            return
        for item in nodes:
            touched_nodes.add(int(item))
        if nodes.size == 1:
            return
        r0 = _find(int(nodes[0]))
        for item in nodes[1:]:
            rk = _find(int(item))
            if rk != r0:
                parent[rk] = r0

    term_entries: List[Tuple[CurrentTerminal, np.ndarray]] = []
    for term in merged_terms:
        pad_nodes = _pad_node_indices(term.pad, node_ids, config)
        if pad_nodes.size == 0:
            errors.append(f"{term.name}: no copper cell found for current injection on net {net_name}.")
            continue
        unique_pad_nodes = np.unique(pad_nodes)
        if config.equipotential_pads:
            _union_nodes(unique_pad_nodes)
        term_entries.append((term, unique_pad_nodes))

    bus_entries: List[Tuple[EquipotentialBus, List[Tuple[Any, str, np.ndarray]], np.ndarray]] = []
    for bus in active_buses:
        pad_list: List[Tuple[Any, str, np.ndarray]] = []
        all_nodes_parts: List[np.ndarray] = []
        for pad_obj, pad_label in bus.pads:
            pad_nodes = _pad_node_indices(pad_obj, node_ids, config)
            if pad_nodes.size == 0:
                errors.append(
                    f"{pad_label}: no copper cell found for equipotential bus "
                    f"{bus.name} on net {net_name}."
                )
                continue
            unique_pad_nodes = np.unique(pad_nodes)
            pad_list.append((pad_obj, str(pad_label), unique_pad_nodes))
            all_nodes_parts.append(unique_pad_nodes)
        if not all_nodes_parts:
            continue
        all_bus_nodes = np.unique(np.concatenate(all_nodes_parts))
        _union_nodes(all_bus_nodes)
        bus_entries.append((bus, pad_list, all_bus_nodes))

    if errors:
        return ElectricalResult(q_full, [], warnings, errors)

    roots = np.arange(node_count, dtype=np.int64)
    for node in touched_nodes:
        roots[node] = _find(node)
    _, supernode_of = np.unique(roots, return_inverse=True)
    super_count = int(np.max(supernode_of)) + 1 if node_count > 0 else 0

    if edge_i.size:
        su = supernode_of[edge_i]
        sv = supernode_of[edge_j]
        keep = su != sv
        c_i = su[keep]
        c_j = sv[keep]
        c_g = edge_g[keep]
    else:
        c_i = np.empty(0, dtype=np.int64)
        c_j = np.empty(0, dtype=np.int64)
        c_g = np.empty(0, dtype=np.float64)

    if c_i.size:
        adj = sp.coo_matrix(
            (
                np.ones(c_i.size * 2, dtype=np.int8),
                (np.concatenate([c_i, c_j]), np.concatenate([c_j, c_i])),
            ),
            shape=(super_count, super_count),
        ).tocsr()
        comp_count, labels = connected_components(adj, directed=False, return_labels=True)
    else:
        comp_count = super_count
        labels = np.arange(super_count, dtype=np.int64)

    rhs_super = np.zeros(super_count, dtype=np.float64)
    terminal_components = set()
    terminal_component_current: Dict[int, float] = {}
    terminal_records: List[Tuple[CurrentTerminal, np.ndarray, List[int]]] = []

    for term, unique_pad_nodes in term_entries:
        current = float(term.current_a)
        s_nodes = supernode_of[unique_pad_nodes]
        if config.equipotential_pads:
            s_id = int(s_nodes[0])
            rhs_super[s_id] += current
            comps = {int(labels[s_id])}
            terminal_components.update(comps)
            terminal_component_current[int(labels[s_id])] = (
                terminal_component_current.get(int(labels[s_id]), 0.0) + current
            )
        else:
            np.add.at(rhs_super, s_nodes, current / float(unique_pad_nodes.size))
            comps = {int(labels[int(sn)]) for sn in s_nodes}
            terminal_components.update(comps)
            for comp in comps:
                in_comp = labels[s_nodes] == comp
                terminal_component_current[comp] = terminal_component_current.get(comp, 0.0) + (
                    current * float(np.count_nonzero(in_comp)) / float(unique_pad_nodes.size)
                )
        terminal_records.append((term, unique_pad_nodes, sorted(comps)))

    bus_records: List[Tuple[EquipotentialBus, List[Tuple[Any, str, np.ndarray, List[int]]], np.ndarray]] = []
    for bus, pad_list, all_bus_nodes in bus_entries:
        bus_current = float(bus.total_current_a)
        s_bus = int(supernode_of[all_bus_nodes[0]])
        rhs_super[s_bus] += bus_current
        bus_comp = int(labels[s_bus])
        terminal_components.add(bus_comp)
        terminal_component_current[bus_comp] = (
            terminal_component_current.get(bus_comp, 0.0) + bus_current
        )
        pad_recs = []
        for pad_obj, pad_label, unique_pad_nodes in pad_list:
            p_comps = sorted({int(labels[int(supernode_of[n])]) for n in unique_pad_nodes})
            pad_recs.append((pad_obj, pad_label, unique_pad_nodes, p_comps))
        bus_records.append((bus, pad_recs, all_bus_nodes))

    if len(terminal_components) > 1:
        errors.append(
            f"Current pads on net {net_name} are not electrically connected "
            f"({len(terminal_components)} separate copper islands)."
        )

    total_abs = float(
        sum(abs(float(t.current_a)) for t in terms)
        + sum(abs(float(b.total_current_a)) for b in active_buses)
    )
    tol = max(config.balance_abs_tol, config.balance_rel_tol * total_abs)
    for comp, comp_current in terminal_component_current.items():
        if abs(comp_current) > tol:
            errors.append(
                f"Net {net_name} copper island is not current-balanced: "
                f"sum(I)={comp_current:.9g} A."
            )

    if errors:
        return ElectricalResult(q_full, [], warnings, errors)

    if c_i.size:
        rows = np.concatenate([c_i, c_j, c_i, c_j])
        cols = np.concatenate([c_i, c_j, c_j, c_i])
        data = np.concatenate([c_g, c_g, -c_g, -c_g])
        lap = sp.coo_matrix((data, (rows, cols)), shape=(super_count, super_count)).tocsr()
    else:
        lap = sp.csr_matrix((super_count, super_count), dtype=np.float64)

    potentials_super = np.zeros(super_count, dtype=np.float64)
    active_nodes = np.flatnonzero(np.abs(rhs_super) > 0.0)
    active_components = sorted({int(labels[node]) for node in active_nodes})

    for comp in active_components:
        if cancel_check is not None and cancel_check():
            raise GeometryCancelledError("Electrical solve cancelled.")
        comp_nodes = np.flatnonzero(labels == comp)
        if comp_nodes.size <= 1:
            continue
        ref = int(comp_nodes[0])
        solve_nodes = comp_nodes[comp_nodes != ref]
        try:
            sub_lap = lap[solve_nodes][:, solve_nodes]
            sub_rhs = rhs_super[solve_nodes]
            potentials_super[solve_nodes] = _solve_electrical_system(sub_lap, sub_rhs)
        except Exception as exc:
            errors.append(f"Electrical solve failed for net {net_name}: {exc}")
            return ElectricalResult(q_full, [], warnings, errors)

    potentials = potentials_super[supernode_of]

    q_nodes = np.zeros(node_count, dtype=np.float64)
    edge_current = np.empty(0, dtype=np.float64)
    if edge_i.size:
        dv = potentials[edge_i] - potentials[edge_j]
        edge_current = edge_g * dv
        p_edge = edge_current * dv
        np.add.at(q_nodes, edge_i, 0.5 * p_edge)
        np.add.at(q_nodes, edge_j, 0.5 * p_edge)

    q_full[global_indices] = q_nodes
    total_loss = float(np.sum(q_nodes))

    # Current density map (A/mm^2) across in-plane X and Y edges
    n_inplane = int(edge_i.size) - int(via_edge_count)
    i_cell_sum = np.zeros(node_count, dtype=np.float64)
    if n_inplane > 0:
        i_inplane = np.abs(edge_current[:n_inplane])
        np.add.at(i_cell_sum, edge_i[:n_inplane], 0.5 * i_inplane)
        np.add.at(i_cell_sum, edge_j[:n_inplane], 0.5 * i_inplane)

    if width_scale is not None:
        ws_flat = np.maximum(width_scale.reshape(-1)[global_indices].astype(np.float64), 1e-6)
    else:
        ws_flat = np.ones(node_count, dtype=np.float64)
    layer_of_node = global_indices // rc
    t_cu_mm_of_node = np.asarray(config.t_cu, dtype=np.float64)[layer_of_node] * 1e3
    area_mm2 = np.maximum(float(config.res) * ws_flat * t_cu_mm_of_node, 1e-18)
    j_nodes = (i_cell_sum / area_mm2).astype(np.float32)
    current_density_a_mm2 = np.zeros(copper_mask.shape, dtype=np.float32)
    current_density_a_mm2.reshape(-1)[global_indices] = j_nodes
    max_current_density_a_mm2 = float(np.max(j_nodes)) if j_nodes.size else 0.0

    merged_abs = float(
        sum(abs(float(t.current_a)) for t in merged_terms)
        + sum(abs(float(b.total_current_a)) for b in active_buses)
    )
    source_current = float(
        sum(max(float(t.current_a), 0.0) for t in merged_terms)
        + sum(max(float(b.total_current_a), 0.0) for b in active_buses)
    )
    sink_current = float(
        -sum(min(float(t.current_a), 0.0) for t in merged_terms)
        - sum(min(float(b.total_current_a), 0.0) for b in active_buses)
    )
    current_balance = float(
        sum(float(t.current_a) for t in merged_terms)
        + sum(float(b.total_current_a) for b in active_buses)
    )
    effective_current = source_current if source_current > 0.0 else 0.5 * merged_abs
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
            bus_name=None,
        ))

    for bus, pad_recs, all_bus_nodes in bus_records:
        bus_node_out = np.zeros(node_count, dtype=np.float64)
        if edge_i.size:
            in_bus = np.zeros(node_count, dtype=bool)
            in_bus[all_bus_nodes] = True
            out_mask = in_bus[edge_i] & (~in_bus[edge_j])
            in_mask = (~in_bus[edge_i]) & in_bus[edge_j]
            if np.any(out_mask):
                np.add.at(bus_node_out, edge_i[out_mask], edge_current[out_mask])
            if np.any(in_mask):
                np.add.at(bus_node_out, edge_j[in_mask], -edge_current[in_mask])

        assigned_nodes: set = set()
        for pad_obj, pad_label, unique_pad_nodes, p_comps in pad_recs:
            owned = np.array(
                [int(n) for n in unique_pad_nodes if int(n) not in assigned_nodes],
                dtype=np.int64,
            )
            assigned_nodes.update(int(n) for n in owned)
            actual_current_a = float(np.sum(bus_node_out[owned])) if owned.size else 0.0
            mean_potential = float(np.mean(potentials[unique_pad_nodes])) if unique_pad_nodes.size else None
            x_mm, y_mm = _pad_center_mm(pad_obj)
            terminal_diagnostics.append(ElectricalTerminalDiagnostics(
                name=str(pad_label),
                net_name=str(net_name),
                current_a=actual_current_a,
                layer=_pad_layer_label(pad_obj, config),
                x_mm=x_mm,
                y_mm=y_mm,
                bbox_mm=_bbox_mm(pad_obj.GetBoundingBox()),
                cell_count=int(unique_pad_nodes.size),
                component_ids=[int(comp) for comp in p_comps],
                mean_potential_v=mean_potential,
                bus_name=str(bus.name),
            ))

    # Via current & Joule power diagnostics
    via_diagnostics: List[ElectricalViaDiagnostics] = []
    if via_records:
        sigma_ref = 1.0 / _effective_rho_cu(config)
        for vr in via_records:
            gap_currents: List[float] = []
            total_via_power = 0.0
            for gap_offset, g_idx in enumerate(range(vr.min_l, vr.max_l)):
                g_ref = vr.g_per_gap[gap_offset]
                u_ids = node_ids[g_idx, vr.rs:vr.re, vr.cs:vr.ce][vr.mask_v]
                v_ids = node_ids[g_idx + 1, vr.rs:vr.re, vr.cs:vr.ce][vr.mask_v]
                valid = (u_ids >= 0) & (v_ids >= 0)
                if not np.any(valid):
                    continue
                u_v = u_ids[valid]
                v_v = v_ids[valid]
                dv = potentials[u_v] - potentials[v_v]
                if sigma_grid is not None:
                    s_g = sigma_grid[g_idx, vr.rs:vr.re, vr.cs:vr.ce][vr.mask_v][valid]
                    s_g1 = sigma_grid[g_idx + 1, vr.rs:vr.re, vr.cs:vr.ce][vr.mask_v][valid]
                    s_gap = (2.0 * s_g * s_g1) / np.maximum(s_g + s_g1, 1e-30)
                    g_pix = g_ref * (s_gap / sigma_ref)
                else:
                    g_pix = g_ref
                i_gap = float(np.sum(g_pix * dv))
                p_gap = float(np.sum(g_pix * dv * dv))
                gap_currents.append(abs(i_gap))
                total_via_power += p_gap
            via_current = max(gap_currents) if gap_currents else 0.0
            via_diagnostics.append(ElectricalViaDiagnostics(
                net_name=str(net_name),
                x_mm=float(vr.x_mm),
                y_mm=float(vr.y_mm),
                drill_mm=float(vr.drill_mm),
                layers=str(vr.layers),
                current_a=float(via_current),
                power_w=float(total_via_power),
            ))
        via_diagnostics.sort(key=lambda item: (-item.current_a, -item.power_w))
    top_vias = via_diagnostics[:15]
    max_via_current_a = float(top_vias[0].current_a) if top_vias else 0.0

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
        terminal_count=len(terminal_diagnostics),
        total_current_a=current_balance,
        total_abs_current_a=merged_abs,
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
        top_vias=top_vias,
        max_via_current_a=max_via_current_a,
        current_density_a_mm2=current_density_a_mm2,
        max_current_density_a_mm2=max_current_density_a_mm2,
    )
    if comp_count > 1:
        warnings.append(
            f"Net {net_name} has {comp_count} mapped copper islands; "
            "only islands with current terminals affect Joule heating."
        )
    return ElectricalResult(
        q_full,
        [summary],
        warnings,
        errors,
        current_density_map=current_density_a_mm2,
    )


def _relative_residual(
    matrix: sp.csr_matrix,
    solution: np.ndarray,
    rhs: np.ndarray,
    rhs_norm: Optional[float] = None,
) -> float:
    """Compute the relative 2-norm residual ``||A x - b|| / ||b||``."""
    norm_b = float(np.linalg.norm(rhs)) if rhs_norm is None else float(rhs_norm)
    if norm_b <= 0.0:
        return 0.0
    residual = matrix.dot(solution) - rhs
    return float(np.linalg.norm(residual) / max(norm_b, 1e-30))


def _solve_electrical_system(
    sub_lap: sp.spmatrix,
    sub_rhs: np.ndarray,
    *,
    iterative_threshold: Optional[int] = None,
) -> np.ndarray:
    """Solve one grounded connected-component Laplacian system ``A x = b``.

    The grounded copper-network Laplacian is symmetric positive definite.
    Solver selection order:
    1. For large systems (``N >= iterative_threshold``), Jacobi-preconditioned
       Conjugate Gradients (``spla.cg``) verified against a tight relative
       residual (``<= 1e-8``).
    2. Direct SciPy SuperLU (``spla.spsolve``) on a CSC matrix as default for
       smaller systems and safe fallback.
    """
    rhs = np.asarray(sub_rhs, dtype=np.float64).reshape(-1)
    n_nodes = int(sub_lap.shape[0])
    if n_nodes == 0:
        return np.empty(0, dtype=np.float64)

    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return np.zeros(n_nodes, dtype=np.float64)

    sub_lap_csr = (
        sub_lap if getattr(sub_lap, "format", None) == "csr" else sub_lap.tocsr()
    )

    threshold = (
        _ELECTRICAL_ITERATIVE_NODE_THRESHOLD
        if iterative_threshold is None
        else int(iterative_threshold)
    )
    if n_nodes >= threshold:
        try:
            diag = np.asarray(sub_lap_csr.diagonal(), dtype=np.float64)
            inv_diag = np.divide(
                1.0,
                diag,
                out=np.zeros_like(diag),
                where=np.abs(diag) > 1e-30,
            )
            preconditioner = spla.LinearOperator(
                sub_lap_csr.shape,
                matvec=lambda vec: inv_diag * vec,
                dtype=np.float64,
            )
            cg_rtol = min(_ELECTRICAL_ITERATIVE_RTOL * 0.1, 1e-9)
            try:
                solution, info = spla.cg(
                    sub_lap_csr,
                    rhs,
                    M=preconditioner,
                    rtol=cg_rtol,
                    atol=0.0,
                    maxiter=_ELECTRICAL_ITERATIVE_MAXITER,
                )
            except TypeError:
                solution, info = spla.cg(
                    sub_lap_csr,
                    rhs,
                    M=preconditioner,
                    tol=cg_rtol,
                    atol=0.0,
                    maxiter=_ELECTRICAL_ITERATIVE_MAXITER,
                )
            solution = np.asarray(solution, dtype=np.float64).reshape(-1)
            if (
                info == 0
                and solution.shape == (n_nodes,)
                and np.all(np.isfinite(solution))
                and _relative_residual(sub_lap_csr, solution, rhs, rhs_norm)
                <= _ELECTRICAL_ITERATIVE_RTOL
            ):
                return solution
        except Exception:
            pass

    sub_lap_csc = (
        sub_lap if getattr(sub_lap, "format", None) == "csc" else sub_lap_csr.tocsc()
    )
    return np.asarray(spla.spsolve(sub_lap_csc, rhs), dtype=np.float64).reshape(-1)


def _build_relevant_net_masks(
    board: Any,
    config: ElectricalConfig,
    relevant_nets: set,
    progress_callback: Optional[Callable[[int, int, str], Optional[bool]]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, List[ElectricalPrimitiveDiagnostics]],
    int,
]:
    """Rasterize copper geometry for only the active current nets."""
    net_masks, via_masks, _, primitive_summaries, collision_count = _build_relevant_net_data(
        board,
        config,
        relevant_nets,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        compute_via_conductance=False,
    )
    return net_masks, via_masks, primitive_summaries, collision_count


def _build_relevant_net_data(
    board: Any,
    config: ElectricalConfig,
    relevant_nets: set,
    progress_callback: Optional[Callable[[int, int, str], Optional[bool]]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    compute_via_conductance: bool = True,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, Optional[np.ndarray]],
    Dict[str, List[ElectricalPrimitiveDiagnostics]],
    int,
]:
    """Rasterize copper geometry and physical via conductances for active nets."""
    (
        net_masks,
        via_masks,
        via_g_maps,
        _width_scales,
        _via_records,
        primitive_summaries,
        collision_count,
    ) = _build_relevant_net_bundle(
        board,
        config,
        relevant_nets,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        compute_via_conductance=compute_via_conductance,
    )
    return net_masks, via_masks, via_g_maps, primitive_summaries, collision_count


def _build_relevant_net_bundle(
    board: Any,
    config: ElectricalConfig,
    relevant_nets: set,
    progress_callback: Optional[Callable[[int, int, str], Optional[bool]]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    compute_via_conductance: bool = True,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, Optional[np.ndarray]],
    Dict[str, np.ndarray],
    Dict[str, List[_ViaRecord]],
    Dict[str, List[ElectricalPrimitiveDiagnostics]],
    int,
]:
    """Rasterize copper geometry, width scales, via records, and conductances for active nets."""
    tracker = _ProgressEmitter(progress_callback, cancel_check)
    tracker.check_cancel(force=True)

    layer_count = len(config.copper_ids)
    shape = (layer_count, config.rows, config.cols)
    net_masks = {key: np.zeros(shape, dtype=bool) for key in relevant_nets}
    via_masks = {key: np.zeros((config.rows, config.cols), dtype=bool) for key in relevant_nets}
    via_g_maps: Dict[str, Optional[np.ndarray]] = {key: None for key in relevant_nets}
    width_scales: Dict[str, np.ndarray] = {
        key: np.zeros(shape, dtype=np.float32) for key in relevant_nets
    }
    via_records: Dict[str, List[_ViaRecord]] = {key: [] for key in relevant_nets}
    primitive_stats: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {
        key: {} for key in relevant_nets
    }
    lid_to_idx = {lid: idx for idx, lid in enumerate(config.copper_ids)}
    gap_m = _resolve_gap_m(board, config) if (compute_via_conductance and layer_count > 1) else []
    rho_cu_eff = _effective_rho_cu(config)
    via_plating_um = max(0.0, float(config.via_plating_um))
    via_fill = normalize_via_fill(config.via_fill)

    def accumulate_via_conductance(
        key: str,
        layer_indices: List[int],
        cx_mm: float,
        cy_mm: float,
        spread_radius_mm: float,
        a_cu_eff: float,
        mark_masks: bool = False,
        drill_mm: float = 0.0,
        layers_label: str = "",
        record_via: bool = False,
    ) -> int:
        if layer_count <= 1 or len(layer_indices) < 2:
            return 0
        rs_v, re_v, cs_v, ce_v, mask_v, n_pix = select_via_grid_cells(
            config.rows,
            config.cols,
            config.x_min,
            config.y_min,
            config.res,
            cx_mm,
            cy_mm,
            spread_radius_mm,
        )
        if n_pix <= 0:
            return 0
        added_cells = 0
        if mark_masks:
            via_slice = via_masks[key][rs_v:re_v, cs_v:ce_v]
            before_via = int(np.count_nonzero(via_slice))
            via_slice |= mask_v
            added_cells += int(np.count_nonzero(via_slice)) - before_via
        for l_idx in layer_indices:
            layer_slice = net_masks[key][l_idx, rs_v:re_v, cs_v:ce_v]
            before = int(np.count_nonzero(layer_slice))
            layer_slice |= mask_v
            width_scales[key][l_idx, rs_v:re_v, cs_v:ce_v][mask_v] = np.float32(1.0)
            if mark_masks:
                added_cells += int(np.count_nonzero(layer_slice)) - before

        if compute_via_conductance and a_cu_eff > 0.0:
            min_l = min(layer_indices)
            max_l = max(layer_indices)
            if min_l < max_l:
                g_map = via_g_maps[key]
                if g_map is None:
                    g_map = np.zeros(
                        (layer_count - 1, config.rows, config.cols),
                        dtype=np.float64,
                    )
                    via_g_maps[key] = g_map
                g_per_gap: List[float] = []
                for g in range(min_l, max_l):
                    g_cell = a_cu_eff / (rho_cu_eff * gap_m[g] * n_pix)
                    g_map[g, rs_v:re_v, cs_v:ce_v][mask_v] += g_cell
                    g_per_gap.append(float(g_cell))
                if record_via:
                    via_records[key].append(_ViaRecord(
                        x_mm=float(cx_mm),
                        y_mm=float(cy_mm),
                        drill_mm=float(drill_mm),
                        layers=str(layers_label),
                        rs=int(rs_v),
                        re=int(re_v),
                        cs=int(cs_v),
                        ce=int(ce_v),
                        mask_v=mask_v.copy(),
                        min_l=int(min_l),
                        max_l=int(max_l),
                        g_per_gap=tuple(g_per_gap),
                    ))
        return added_cells

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
        bbox_obj = bbox if bbox is not None else obj.GetBoundingBox()
        rs, re, cs, ce = _bbox_indices(bbox_obj, config)
        if rs >= re or cs >= ce:
            return 0
        mapped_cells = 0
        if as_via:
            via_slice = via_masks[key][rs:re, cs:ce]
            before_via = int(np.count_nonzero(via_slice))
            _fill_bbox_2d(via_masks[key], bbox_obj, config)
            mapped_cells += int(np.count_nonzero(via_slice)) - before_via
            for lid in layer_ids:
                layer_idx = lid_to_idx.get(lid)
                if layer_idx is not None:
                    layer_slice = net_masks[key][layer_idx, rs:re, cs:ce]
                    before = int(np.count_nonzero(layer_slice))
                    _fill_bbox_3d(net_masks[key], layer_idx, bbox_obj, config, width_scale=width_scales[key])
                    mapped_cells += int(np.count_nonzero(layer_slice)) - before
            return mapped_cells
        for lid in layer_ids:
            layer_idx = lid_to_idx.get(lid)
            if layer_idx is None:
                continue
            layer_slice = net_masks[key][layer_idx, rs:re, cs:ce]
            before = int(np.count_nonzero(layer_slice))
            if use_track_shape:
                _fill_track(net_masks[key], layer_idx, obj, config, width_scale=width_scales[key])
            else:
                _fill_bbox_3d(net_masks[key], layer_idx, bbox_obj, config, width_scale=width_scales[key])
            mapped_cells += int(np.count_nonzero(layer_slice)) - before
        return mapped_cells

    try:
        footprints = list(board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints())
    except Exception:
        footprints = []
    try:
        tracks = list(board.Tracks() if hasattr(board, "Tracks") else board.GetTracks())
    except Exception:
        tracks = []
    try:
        zones = list(board.Zones() if hasattr(board, "Zones") else board.GetZones())
    except Exception:
        zones = []

    total_fps = len(footprints)
    total_tracks = len(tracks)
    total_zones = len(zones)
    total_items = max(1, total_fps + total_tracks + total_zones)
    completed = 0

    for fp_idx, fp in enumerate(footprints):
        tracker.emit(
            completed + fp_idx,
            total_items,
            f"Mapping electrical pads ({fp_idx + 1} / {total_fps})...",
            force=(fp_idx == 0),
        )
        for pad in fp.Pads():
            key, _, _ = net_key_from_obj(pad)
            if key not in net_masks:
                continue
            bbox = pad.GetBoundingBox()
            pad_layer_indices = get_pad_copper_layer_indices(pad, config.copper_ids, lid_to_idx)
            has_pad_shape = hasattr(pad, "GetSize") and hasattr(pad, "GetShape")
            if _is_pth_pad(pad):
                target_indices = (
                    pad_layer_indices
                    if pad_layer_indices
                    else list(range(len(config.copper_ids)))
                )
                if has_pad_shape:
                    mapped_cells = 0
                    first_lid = (
                        config.copper_ids[target_indices[0]]
                        if target_indices and 0 <= target_indices[0] < len(config.copper_ids)
                        else None
                    )
                    rs, re, cs, ce, via_pad_mask = get_pad_grid_mask(
                        pad,
                        config.rows,
                        config.cols,
                        config.x_min,
                        config.y_min,
                        config.res,
                        layer_id=first_lid,
                    )
                    if rs < re and cs < ce and np.any(via_pad_mask):
                        via_slice = via_masks[key][rs:re, cs:ce]
                        before_via = int(np.count_nonzero(via_slice))
                        via_slice |= via_pad_mask
                        mapped_cells += int(np.count_nonzero(via_slice)) - before_via
                    for layer_idx in target_indices:
                        lid = config.copper_ids[layer_idx]
                        rs_l, re_l, cs_l, ce_l, pad_mask = get_pad_grid_mask(
                            pad,
                            config.rows,
                            config.cols,
                            config.x_min,
                            config.y_min,
                            config.res,
                            layer_id=lid,
                        )
                        if rs_l < re_l and cs_l < ce_l and np.any(pad_mask):
                            layer_slice = net_masks[key][layer_idx, rs_l:re_l, cs_l:ce_l]
                            before = int(np.count_nonzero(layer_slice))
                            layer_slice |= pad_mask
                            width_scales[key][layer_idx, rs_l:re_l, cs_l:ce_l][pad_mask] = np.float32(1.0)
                            mapped_cells += int(np.count_nonzero(layer_slice)) - before
                else:
                    target_lids = [config.copper_ids[idx] for idx in target_indices]
                    mapped_cells = fill_for_obj(pad, target_lids, bbox=bbox, as_via=True)
                if compute_via_conductance and layer_count > 1 and len(target_indices) >= 2:
                    if has_pad_shape:
                        try:
                            size = pad.GetSize()
                            outer_d_mm = min(float(size.x), float(size.y)) * 1e-6
                            cx_mm, cy_mm = _pad_shape_center_mm(pad)
                        except Exception:
                            outer_d_mm = min(
                                float(bbox.GetWidth()), float(bbox.GetHeight())
                            ) * 1e-6
                            cx_mm = (float(bbox.GetX()) + 0.5 * float(bbox.GetWidth())) * 1e-6
                            cy_mm = (float(bbox.GetY()) + 0.5 * float(bbox.GetHeight())) * 1e-6
                    else:
                        outer_d_mm = min(
                            float(bbox.GetWidth()), float(bbox.GetHeight())
                        ) * 1e-6
                        cx_mm = (float(bbox.GetX()) + 0.5 * float(bbox.GetWidth())) * 1e-6
                        cy_mm = (float(bbox.GetY()) + 0.5 * float(bbox.GetHeight())) * 1e-6
                    dx_mm, dy_mm = _extract_pth_drill_mm(pad, outer_d_mm)
                    a_cu_eff = compute_via_copper_area_m2(
                        dx_mm * 1e-3,
                        dy_mm * 1e-3,
                        plating_um=via_plating_um,
                        via_fill=via_fill,
                    )
                    spread_radius_mm = max(
                        0.5 * max(outer_d_mm, 0.0),
                        0.5 * max(dx_mm, dy_mm) + via_plating_um * 1e-3,
                    )
                    accumulate_via_conductance(
                        key,
                        target_indices,
                        cx_mm,
                        cy_mm,
                        spread_radius_mm,
                        a_cu_eff,
                        mark_masks=False,
                    )
                record_primitive(
                    key, "Pad", "All copper", bbox,
                    mapped_cells=max(0, mapped_cells)
                )
                record_primitive(
                    key, "Via/PTH", "All copper", bbox,
                    mapped_cells=max(0, mapped_cells)
                )
            else:
                for layer_idx in pad_layer_indices:
                    layer_id = config.copper_ids[layer_idx]
                    if has_pad_shape:
                        rs, re, cs, ce, pad_mask = get_pad_grid_mask(
                            pad,
                            config.rows,
                            config.cols,
                            config.x_min,
                            config.y_min,
                            config.res,
                            layer_id=layer_id,
                        )
                        if rs < re and cs < ce and np.any(pad_mask):
                            layer_slice = net_masks[key][layer_idx, rs:re, cs:ce]
                            before = int(np.count_nonzero(layer_slice))
                            layer_slice |= pad_mask
                            width_scales[key][layer_idx, rs:re, cs:ce][pad_mask] = np.float32(1.0)
                            mapped_cells = int(np.count_nonzero(layer_slice)) - before
                        else:
                            mapped_cells = 0
                    else:
                        mapped_cells = fill_for_obj(pad, [layer_id], bbox=bbox)
                    record_primitive(
                        key, "Pad", _layer_label(layer_id, config), bbox,
                        mapped_cells=max(0, mapped_cells)
                    )
    completed += total_fps

    for tr_idx, track in enumerate(tracks):
        tracker.emit(
            completed + tr_idx,
            total_items,
            f"Mapping electrical tracks ({tr_idx + 1} / {total_tracks})...",
            force=(tr_idx == 0),
        )
        key, _, _ = net_key_from_obj(track)
        if key not in net_masks:
            continue
        bbox = track.GetBoundingBox()
        is_via = "VIA" in str(type(track)).upper()
        if is_via:
            layer_ids = _via_layer_ids(track, config.copper_ids)
            layer_indices = sorted(
                lid_to_idx[lid] for lid in layer_ids if lid in lid_to_idx
            )
            has_pos = hasattr(track, "GetPosition")
            if has_pos:
                via_layer_id = (
                    config.copper_ids[layer_indices[0]]
                    if layer_indices and config.copper_ids
                    else None
                )
                try:
                    if via_layer_id is not None:
                        try:
                            width_iu = track.GetWidth(int(via_layer_id))
                        except TypeError:
                            width_iu = track.GetWidth()
                    else:
                        width_iu = track.GetWidth()
                    outer_d_mm = float(width_iu) * 1e-6
                    pos = track.GetPosition()
                    cx_mm = float(pos.x) * 1e-6
                    cy_mm = float(pos.y) * 1e-6
                except Exception:
                    has_pos = False
            if not has_pos:
                outer_d_mm = max(
                    float(bbox.GetWidth()), float(bbox.GetHeight())
                ) * 1e-6
                cx_mm = (float(bbox.GetX()) + 0.5 * float(bbox.GetWidth())) * 1e-6
                cy_mm = (float(bbox.GetY()) + 0.5 * float(bbox.GetHeight())) * 1e-6
                mapped_cells = fill_for_obj(track, layer_ids, bbox=bbox, as_via=True)
            drill_mm = _extract_via_drill_mm(track, outer_d_mm)
            a_cu_eff = compute_via_copper_area_m2(
                drill_mm * 1e-3,
                plating_um=via_plating_um,
                via_fill=via_fill,
            )
            spread_radius_mm = max(
                0.5 * max(outer_d_mm, 0.0),
                0.5 * drill_mm + via_plating_um * 1e-3,
            )
            layers_lbl = _layers_label(layer_ids, config)
            if has_pos:
                mapped_cells = accumulate_via_conductance(
                    key,
                    layer_indices,
                    cx_mm,
                    cy_mm,
                    spread_radius_mm,
                    a_cu_eff,
                    mark_masks=True,
                    drill_mm=drill_mm,
                    layers_label=layers_lbl,
                    record_via=True,
                )
            elif compute_via_conductance:
                accumulate_via_conductance(
                    key,
                    layer_indices,
                    cx_mm,
                    cy_mm,
                    spread_radius_mm,
                    a_cu_eff,
                    mark_masks=False,
                    drill_mm=drill_mm,
                    layers_label=layers_lbl,
                    record_via=True,
                )
            record_primitive(
                key, "Via/PTH", layers_lbl, bbox,
                mapped_cells=max(0, mapped_cells)
            )
        else:
            layer_id = track.GetLayer()
            mapped_cells = fill_for_obj(track, [layer_id], bbox=bbox, use_track_shape=True)
            record_primitive(
                key,
                "Track",
                _layer_label(layer_id, config),
                bbox,
                mapped_cells=max(0, mapped_cells),
                track_length_mm=_track_length_mm(track),
                track_width_mm=_track_width_mm(track),
            )
    completed += total_tracks

    for z_idx, zone in enumerate(zones):
        tracker.emit(
            completed + z_idx,
            total_items,
            f"Rasterizing electrical zone {z_idx + 1} / {total_zones}...",
            force=(z_idx == 0),
        )
        if _is_rule_or_keepout_zone(zone):
            continue
        key, _, _ = net_key_from_obj(zone)
        if key not in net_masks:
            continue
        if hasattr(zone, "IsFilled") and not zone.IsFilled():
            continue
        bbox = zone.GetBoundingBox()
        rs, re, cs, ce = _bbox_indices(bbox, config)
        has_bbox_overlap = rs < re and cs < ce
        zone_progress_cb = None
        if progress_callback is not None:
            def _zone_sub_progress(
                _sub_done,
                _sub_total,
                sub_msg,
                _z_i=z_idx,
                _z_base=completed + z_idx,
            ):
                tracker.emit(
                    _z_base,
                    total_items,
                    f"Rasterizing electrical zone {_z_i + 1} / {total_zones}: {sub_msg}",
                    force=True,
                )
            zone_progress_cb = _zone_sub_progress
        for lid in _zone_layer_ids(zone, config.copper_ids):
            layer_idx = lid_to_idx.get(lid)
            if layer_idx is not None:
                if has_bbox_overlap:
                    layer_slice = net_masks[key][layer_idx, rs:re, cs:ce]
                    before = int(np.count_nonzero(layer_slice))
                    _fill_zone(
                        net_masks[key],
                        layer_idx,
                        lid,
                        zone,
                        config,
                        progress_callback=zone_progress_cb,
                        cancel_check=cancel_check,
                        width_scale=width_scales[key],
                    )
                    mapped_cells = int(np.count_nonzero(layer_slice)) - before
                else:
                    mapped_cells = 0
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

    return (
        net_masks,
        via_masks,
        via_g_maps,
        width_scales,
        via_records,
        primitive_summaries,
        collision_count,
    )


def _build_net_edges(
    mask: np.ndarray,
    via_mask: Optional[np.ndarray],
    node_ids: np.ndarray,
    config: ElectricalConfig,
    via_g_map: Optional[np.ndarray] = None,
    width_scale: Optional[np.ndarray] = None,
    sigma_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build graph edges and conductances for one net mask."""
    edge_i = []
    edge_j = []
    edge_g = []
    via_edge_count = 0
    dx = config.res * 1e-3
    dy = dx
    sigma_ref = 1.0 / _effective_rho_cu(config)

    for layer_idx in range(mask.shape[0]):
        t_layer = float(config.t_cu[layer_idx])
        if sigma_grid is not None:
            sig_layer = sigma_grid[layer_idx]
        else:
            sig_layer = sigma_ref
        if width_scale is not None:
            ws_layer = np.maximum(width_scale[layer_idx].astype(np.float64), 1e-6)
        else:
            ws_layer = 1.0

        g_cell = sig_layer * t_layer * ws_layer
        uniform_cell = np.ndim(g_cell) == 0

        both = mask[layer_idx, :, :-1] & mask[layer_idx, :, 1:]
        if np.any(both):
            i_idx = node_ids[layer_idx, :, :-1][both]
            j_idx = node_ids[layer_idx, :, 1:][both]
            edge_i.append(i_idx)
            edge_j.append(j_idx)
            if uniform_cell:
                gx = float(g_cell) * (dy / dx)
                edge_g.append(np.full(i_idx.shape, gx, dtype=np.float64))
            else:
                gi = g_cell[:, :-1][both]
                gj = g_cell[:, 1:][both]
                gx_arr = (2.0 * gi * gj / np.maximum(gi + gj, 1e-30)) * (dy / dx)
                edge_g.append(gx_arr)

        both = mask[layer_idx, :-1, :] & mask[layer_idx, 1:, :]
        if np.any(both):
            i_idx = node_ids[layer_idx, :-1, :][both]
            j_idx = node_ids[layer_idx, 1:, :][both]
            edge_i.append(i_idx)
            edge_j.append(j_idx)
            if uniform_cell:
                gy = float(g_cell) * (dx / dy)
                edge_g.append(np.full(i_idx.shape, gy, dtype=np.float64))
            else:
                gi = g_cell[:-1, :][both]
                gj = g_cell[1:, :][both]
                gy_arr = (2.0 * gi * gj / np.maximum(gi + gj, 1e-30)) * (dx / dy)
                edge_g.append(gy_arr)

    if mask.shape[0] > 1:
        if via_g_map is not None:
            for layer_idx in range(mask.shape[0] - 1):
                g_layer = via_g_map[layer_idx]
                both = (g_layer > 0.0) & mask[layer_idx] & mask[layer_idx + 1]
                if np.any(both):
                    i_idx = node_ids[layer_idx][both]
                    j_idx = node_ids[layer_idx + 1][both]
                    edge_i.append(i_idx)
                    edge_j.append(j_idx)
                    g_vals = g_layer[both]
                    if sigma_grid is not None:
                        s_0 = sigma_grid[layer_idx][both]
                        s_1 = sigma_grid[layer_idx + 1][both]
                        s_gap = (2.0 * s_0 * s_1) / np.maximum(s_0 + s_1, 1e-30)
                        g_vals = g_vals * (s_gap / sigma_ref)
                    edge_g.append(g_vals)
                    via_edge_count += int(i_idx.size)
        elif via_mask is not None and np.any(via_mask):
            gz = 1.0 / max(float(config.via_resistance_ohm), 1e-12)
            for layer_idx in range(mask.shape[0] - 1):
                both = via_mask & mask[layer_idx] & mask[layer_idx + 1]
                if np.any(both):
                    i_idx = node_ids[layer_idx][both]
                    j_idx = node_ids[layer_idx + 1][both]
                    edge_i.append(i_idx)
                    edge_j.append(j_idx)
                    if sigma_grid is not None:
                        s_0 = sigma_grid[layer_idx][both]
                        s_1 = sigma_grid[layer_idx + 1][both]
                        s_gap = (2.0 * s_0 * s_1) / np.maximum(s_0 + s_1, 1e-30)
                        g_vals = gz * (s_gap / sigma_ref)
                    else:
                        g_vals = np.full(i_idx.shape, gz, dtype=np.float64)
                    edge_g.append(g_vals)
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
    lid_to_idx = {lid: idx for idx, lid in enumerate(config.copper_ids)}
    layers = get_pad_copper_layer_indices(pad, config.copper_ids, lid_to_idx)
    if not layers:
        return np.empty(0, dtype=np.int64)

    nodes = []
    if hasattr(pad, "GetSize") and hasattr(pad, "GetShape"):
        for layer_idx in layers:
            lid = config.copper_ids[layer_idx]
            rs, re, cs, ce, mask_2d = get_pad_grid_mask(
                pad,
                config.rows,
                config.cols,
                config.x_min,
                config.y_min,
                config.res,
                layer_id=lid,
            )
            if rs < re and cs < ce and np.any(mask_2d):
                sub = node_ids[layer_idx, rs:re, cs:ce][mask_2d]
                valid = sub[sub >= 0]
                if valid.size:
                    nodes.append(valid.reshape(-1))
    else:
        rs, re, cs, ce = _bbox_indices(pad.GetBoundingBox(), config)
        if rs >= re or cs >= ce:
            return np.empty(0, dtype=np.int64)
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
        angle = _pad_orientation_radians(pad)
        return _pad_shape_center_mm(pad, angle_rad=angle)
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
    """Return track centerline length in millimeters (true arc length for PCB_ARC)."""
    try:
        start = track.GetStart()
        end = track.GetEnd()
        if hasattr(track, "GetMid"):
            try:
                middle = track.GetMid()
                if middle is not None:
                    arc = _arc_parameters(start, middle, end)
                    if arc is not None:
                        _, _, radius, _, signed_span = arc
                        return float(radius * abs(signed_span))
            except Exception:
                pass
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
        return _layer_label(get_pad_primary_layer_id(pad, config.copper_ids), config)
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


def _fill_bbox_3d(
    mask: np.ndarray,
    layer_idx: int,
    bbox: Any,
    config: ElectricalConfig,
    width_scale: Optional[np.ndarray] = None,
):
    """Fill a rectangular region on one layer."""
    rs, re, cs, ce = _bbox_indices(bbox, config)
    if rs < re and cs < ce:
        mask[layer_idx, rs:re, cs:ce] = True
        if width_scale is not None:
            width_scale[layer_idx, rs:re, cs:ce] = np.float32(1.0)


def _fill_bbox_2d(mask: np.ndarray, bbox: Any, config: ElectricalConfig):
    """Fill a rectangular region in a 2D mask."""
    rs, re, cs, ce = _bbox_indices(bbox, config)
    if rs < re and cs < ce:
        mask[rs:re, cs:ce] = True


def _apply_track_patch(
    mask: np.ndarray,
    layer_idx: int,
    config: ElectricalConfig,
    rs: int,
    re: int,
    cs: int,
    ce: int,
    local_hit: np.ndarray,
    xs: Optional[np.ndarray],
    ys: Optional[np.ndarray],
    track_scale: float,
    width_scale: Optional[np.ndarray],
):
    """Commit rasterized track cells + 4-connected centerline and update width_scale."""
    rs_p, re_p, cs_p, ce_p = rs, re, cs, ce
    has_cl = xs is not None and ys is not None and xs.size > 0 and config.res > 0.0
    if has_cl:
        grid_x_max = config.x_min + config.cols * config.res
        grid_y_max = config.y_min + config.rows * config.res
        inb = (
            (xs >= config.x_min)
            & (xs <= grid_x_max)
            & (ys >= config.y_min)
            & (ys <= grid_y_max)
        )
        if np.any(inb):
            c_min = max(0, min(config.cols - 1, int(math.floor((float(np.min(xs[inb])) - config.x_min) / config.res))))
            c_max = max(0, min(config.cols - 1, int(math.floor((float(np.max(xs[inb])) - config.x_min) / config.res))))
            r_min = max(0, min(config.rows - 1, int(math.floor((float(np.min(ys[inb])) - config.y_min) / config.res))))
            r_max = max(0, min(config.rows - 1, int(math.floor((float(np.max(ys[inb])) - config.y_min) / config.res))))
            cs_p = min(cs_p, c_min)
            ce_p = min(config.cols, max(ce_p, c_max + 2))
            rs_p = min(rs_p, r_min)
            re_p = min(config.rows, max(re_p, r_max + 2))
    if rs_p >= re_p or cs_p >= ce_p:
        return
    patch = np.zeros((re_p - rs_p, ce_p - cs_p), dtype=bool)
    if rs < re and cs < ce and np.any(local_hit):
        patch[(rs - rs_p):(re - rs_p), (cs - cs_p):(ce - cs_p)] |= local_hit
    if has_cl:
        _mark_centerline_points(
            patch,
            re_p - rs_p,
            ce_p - cs_p,
            config.x_min + cs_p * config.res,
            config.y_min + rs_p * config.res,
            config.res,
            xs,
            ys,
        )
    if np.any(patch):
        mask[layer_idx, rs_p:re_p, cs_p:ce_p] |= patch
        if width_scale is not None:
            ws_slice = width_scale[layer_idx, rs_p:re_p, cs_p:ce_p]
            ws_slice[patch] = np.maximum(ws_slice[patch], np.float32(track_scale))


def _fill_track(
    mask: np.ndarray,
    layer_idx: int,
    track: Any,
    config: ElectricalConfig,
    width_scale: Optional[np.ndarray] = None,
):
    """Rasterize a straight track or circular PCB_ARC, falling back to its bounding box."""
    if not all(hasattr(track, attr) for attr in ("GetStart", "GetEnd", "GetWidth")):
        _fill_bbox_3d(mask, layer_idx, track.GetBoundingBox(), config, width_scale=width_scale)
        return
    try:
        start = track.GetStart()
        end = track.GetEnd()
        raw_width_mm = float(track.GetWidth()) * 1e-6
    except Exception:
        _fill_bbox_3d(mask, layer_idx, track.GetBoundingBox(), config, width_scale=width_scale)
        return

    if 0.0 < raw_width_mm < config.res and config.res > 0.0:
        track_scale = min(1.0, raw_width_mm / config.res)
        radius = 0.5 * raw_width_mm
    else:
        track_scale = 1.0
        radius = 0.5 * max(raw_width_mm, config.res)

    bbox = track.GetBoundingBox()
    rs, re, cs, ce = _bbox_indices(bbox, config)

    arc = None
    if hasattr(track, "GetMid"):
        try:
            middle = track.GetMid()
            if middle is not None:
                arc = _arc_parameters(start, middle, end)
        except Exception:
            arc = None

    if rs < re and cs < ce:
        y = config.y_min + (np.arange(rs, re, dtype=np.float64) + 0.5) * config.res
        x = config.x_min + (np.arange(cs, ce, dtype=np.float64) + 0.5) * config.res
        xx, yy = np.meshgrid(x, y)
    else:
        xx = np.empty((0, 0), dtype=np.float64)
        yy = np.empty((0, 0), dtype=np.float64)

    if arc is not None:
        cx, cy, arc_radius, start_angle, signed_span = arc
        if xx.size:
            dx = xx - cx
            dy = yy - cy
            radial = np.abs(np.hypot(dx, dy) - arc_radius) <= radius
            angles = np.arctan2(dy, dx)
            tau = 2.0 * np.pi
            if signed_span >= 0.0:
                angular = np.mod(angles - start_angle, tau) <= signed_span
            else:
                angular = np.mod(start_angle - angles, tau) <= (-signed_span)
            sx, sy = float(start.x) * 1e-6, float(start.y) * 1e-6
            ex, ey = float(end.x) * 1e-6, float(end.y) * 1e-6
            endcaps = (np.hypot(xx - sx, yy - sy) <= radius) | (np.hypot(xx - ex, yy - ey) <= radius)
            local_hit = (radial & angular) | endcaps
        else:
            local_hit = np.zeros((0, 0), dtype=bool)
        xs = ys = None
        if config.res > 0.0:
            arc_len = arc_radius * abs(signed_span)
            n_steps = max(1, int(math.ceil(arc_len / (0.4 * config.res))))
            theta = start_angle + np.linspace(0.0, signed_span, n_steps + 1)
            xs = cx + arc_radius * np.cos(theta)
            ys = cy + arc_radius * np.sin(theta)
        _apply_track_patch(
            mask, layer_idx, config, rs, re, cs, ce, local_hit, xs, ys, track_scale, width_scale
        )
        return

    sx, sy = float(start.x) * 1e-6, float(start.y) * 1e-6
    ex, ey = float(end.x) * 1e-6, float(end.y) * 1e-6
    vx, vy = ex - sx, ey - sy
    seg_len_sq = vx * vx + vy * vy

    if xx.size:
        if seg_len_sq <= 1e-24:
            dist = np.hypot(xx - sx, yy - sy)
        else:
            t = ((xx - sx) * vx + (yy - sy) * vy) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            px = sx + t * vx
            py = sy + t * vy
            dist = np.hypot(xx - px, yy - py)
        local_hit = dist <= radius
    else:
        local_hit = np.zeros((0, 0), dtype=bool)
    xs = ys = None
    if config.res > 0.0:
        seg_len = math.sqrt(seg_len_sq)
        n_steps = max(1, int(math.ceil(seg_len / (0.4 * config.res))))
        t_vals = np.linspace(0.0, 1.0, n_steps + 1)
        xs = sx + t_vals * vx
        ys = sy + t_vals * vy
    _apply_track_patch(
        mask, layer_idx, config, rs, re, cs, ce, local_hit, xs, ys, track_scale, width_scale
    )


def _fill_zone(
    mask: np.ndarray,
    layer_idx: int,
    lid: int,
    zone: Any,
    config: ElectricalConfig,
    progress_callback: Optional[Callable[[int, int, str], Optional[bool]]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    width_scale: Optional[np.ndarray] = None,
):
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
        zone_mask,
        None,
        x_values,
        y_values,
        lid,
        zone,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    ):
        mask[layer_idx, rs:re, cs:ce] |= zone_mask
        if width_scale is not None and np.any(zone_mask):
            width_scale[layer_idx, rs:re, cs:ce][zone_mask] = np.float32(1.0)
        return

    has_hit = hasattr(zone, "HitTestFilledArea")
    tracker = _ProgressEmitter(progress_callback, cancel_check)
    total_rows = max(1, re - rs)
    for r in range(rs, re):
        tracker.emit(
            r - rs,
            total_rows,
            f"Hit-testing electrical zone row {r - rs + 1} / {total_rows}...",
            force=(r == rs),
        )
        y_mm = config.y_min + (r + 0.5) * config.res
        y_iu = _to_iu(y_mm)
        for c in range(cs, ce):
            if not has_hit:
                mask[layer_idx, r, c] = True
                if width_scale is not None:
                    width_scale[layer_idx, r, c] = np.float32(1.0)
                continue
            x_mm = config.x_min + (c + 0.5) * config.res
            pos = pcbnew.VECTOR2I(_to_iu(x_mm), y_iu)
            try:
                if zone.HitTestFilledArea(lid, pos, 1):
                    mask[layer_idx, r, c] = True
                    if width_scale is not None:
                        width_scale[layer_idx, r, c] = np.float32(1.0)
            except TypeError:
                if zone.HitTestFilledArea(lid, pos):
                    mask[layer_idx, r, c] = True
                    if width_scale is not None:
                        width_scale[layer_idx, r, c] = np.float32(1.0)


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
    """Return layer IDs connected by a via-like object (KiCad 9/10 safe)."""
    if hasattr(via, "GetLayerSet"):
        try:
            layer_set = via.GetLayerSet()
        except Exception:
            layer_set = None
        if layer_set is not None:
            if hasattr(layer_set, "Contains"):
                try:
                    ids = [lid for lid in copper_ids if layer_set.Contains(lid)]
                    if ids:
                        return ids
                except Exception:
                    pass
            if hasattr(layer_set, "CuStack"):
                try:
                    cu_stack = {int(lid) for lid in layer_set.CuStack()}
                    ids = [lid for lid in copper_ids if lid in cu_stack]
                    if ids:
                        return ids
                except Exception:
                    pass
    if hasattr(via, "_layers"):
        return [lid for lid in getattr(via, "_layers") if lid in copper_ids]
    if hasattr(via, "GetLayerPair"):
        try:
            ids = list(via.GetLayerPair())
            if ids:
                return [lid for lid in copper_ids if min(ids) <= lid <= max(ids)]
        except Exception:
            pass
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
