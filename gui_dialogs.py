"""
GUI dialogs for ThermalSim.

This module provides the wxPython dialog for configuring thermal
simulation parameters, organized into compact native notebook tabs with
persistent board context, preflight status, and result actions.
"""

import copy
import json
import os
import wx

try:
    import wx.adv
    _HAS_WX_ADV = True
except ImportError:
    _HAS_WX_ADV = False

try:
    from .capabilities import HAS_NATIVE_CORE, HAS_NATIVE_MG
except ImportError:
    HAS_NATIVE_CORE = False
    HAS_NATIVE_MG = False


# Tooltip text for every control, keyed by internal name
TOOLTIP_TEXTS = {
    'stackup': "Read-only stackup parsed from your .kicad_pcb file.",
    'pads': "Pads selected when the dialog opened. Use Heat Sources and Current Heating to choose simulation roles.",
    'power': "Heat dissipation for power pads in Watts. Single value, comma-separated values, or PWL file path.",
    'power_pads': "Pads used for manual heat dissipation. This list is independent from current-path terminals.",
    'power_apply': "Apply the Power field to selected power-pad rows. If no row is selected, it is applied to all rows.",
    'power_list': "Apply comma-separated power values in the same order as the power-pad table.",
    'browse_pwl': "Select a Piecewise-Linear (.pwl/.csv/.txt) file for time-varying power.",
    'duration': "Total simulation time in seconds. Longer durations approach steady-state.",
    'steady_state': "Solve directly for the final steady-state temperature (much faster than transient stepping for constant heat sources).",
    'ambient': "Surrounding air temperature in \u00b0C. Typical lab conditions: 25 \u00b0C.",
    'resolution': "Desired grid cell size in mm. The compute budget may use a coarser actual size.",
    'show_all': "Show every copper layer in the heatmap (vs. only top and bottom).",
    'snapshots': "Save intermediate temperature snapshots during the simulation.",
    'snap_count': "Number of snapshots between t=0 and the final time.",
    'output_dir': "Directory for results. A timestamped subfolder is created automatically.",
    'browse_output': "Select output directory for simulation results.",
    'ignore_traces': "Exclude traces from the thermal model. Faster, slightly more conservative.",
    'limit_area': "Crop the solver to all active heat sources and complete current-net geometry.",
    'limit_dist': "Thermal margin around sources and complete current nets. Typical: 10-30 mm.",
    'enable_pad': "Model a thermal pad/heatsink on the bottom layer (shapes on User.Eco1).",
    'pad_thick': "Thermal pad thickness in mm.",
    'pad_k': "Thermal pad conductivity in W/(m\u00b7K). Silicone ~3, aluminium ~200.",
    'pad_cap': "Areal heat capacity in J/(m\u00b2\u00b7K). Set 0 for negligible thermal mass.",
    'pad_contact_factor': "Effective contact factor (0.05 to 1.0) accounting for thermal interface contact quality.",
    'pad_sink_mode': "Select a fixed boundary temperature or a dynamic 0D heatsink with finite thermal resistance and mass.",
    'pad_sink_temp_use_ambient': "Use the simulation ambient temperature as the fixed sink temperature.",
    'pad_sink_temp_c': "Fixed sink temperature in \u00b0C when not using ambient temperature.",
    'heatsink_rth_kw': "Heatsink-to-ambient thermal resistance R_th in K/W.",
    'heatsink_material': "Heatsink material preset used to auto-fill specific heat capacity c_p.",
    'heatsink_mass_g': "Heatsink mass in grams. Set 0 for an instantaneous (zero thermal mass) heatsink.",
    'heatsink_cp_j_kgk': "Specific heat capacity c_p in J/(kg\u00b7K). Editable only when material is Custom.",
    'h_conv': "Effective top-surface heat transfer coefficient (convection + radiation) in W/(m\u00b2\u00b7K). Insulated ~0-2, still air ~8-12, fan ~25-100.",
    'h_conv_bottom': "Effective bottom-surface heat transfer coefficient (convection + radiation) in W/(m\u00b2\u00b7K). Insulated ~0-2, still air ~8-12, fan ~25-100.",
    'via_plating_um': "Via and PTH barrel copper plating wall thickness in \u00b5m (IPC Class 2 standard ~25 \u00b5m; range 5-100 \u00b5m). Disabled when copper-filled.",
    'via_fill': "Via fill type: 'Plated (standard)' models hollow/resin-filled plated barrels; 'Copper-filled' models solid copper VIPPO vias.",
    'pcb_thick': "Fallback PCB thickness in mm when board stackup thickness is unavailable.",
    'grid_detail': "Compute budget used to preserve the desired cell size on large simulations.",
    'grid_node_budget': (
        "Maximum total solver nodes across all copper layers in Custom mode "
        "(up to 100 million). Compact native mode uses ~80-125 MB per 1M nodes."
    ),
    'grid_estimate': (
        "Live estimate of actual grid cell size, total solver nodes across all copper "
        "layers, expected peak RAM, and relative runtime."
    ),
    'area_estimate': (
        "Active simulation area relative to the full board bounding box. Enable "
        "'Limit area' to crop around heat sources and current nets."
    ),
    'heatsink_summary': (
        "Derived dynamic 0D heatsink parameters: thermal capacitance C_hs = m * c_p "
        "and thermal time constant tau = R_th * C_hs."
    ),
    'compute_engine': (
        "Solver engine: Auto uses direct SciPy LU below 50k nodes and Native C++/OpenMP "
        "Matrix-Free PCG above. 'Fast CPU' forces Matrix-Free; 'Legacy sparse' forces "
        "direct SciPy LU (high RAM on large grids)."
    ),
    'time_stepping': (
        "Transient BDF2 time-stepping schedule. Auto uses fast 5-phase geometric stepping "
        "in Matrix-Free mode; Uniform BDF2 uses a constant step size throughout."
    ),
    'grid_expert_limits': "Legacy expert-grid setting; loaded as a Custom compute budget.",
    'grid_max_cells': "Legacy 2D cell threshold retained for settings compatibility.",
    'grid_target_cells': "Legacy coarsening target retained for settings compatibility.",
    'capabilities': "Detected thermal solver engine and math backends.",
    'help': "Open the ThermalSim documentation in your web browser.",
    'more': "Open actions to load/save settings JSON files or view online documentation.",
    'preview': "Generate a geometry preview image (copper, vias, heatsink, heat sources) without running the solver.",
    'run': "Start the thermal simulation with the current settings.",
    'cancel': "Close the ThermalSim dialog without running a simulation.",
    'open_report': "Open the generated interactive HTML thermal report in your default browser.",
    'open_folder': "Open the output directory containing heatmaps, snapshots, and report files.",
    'load_settings': "Load simulation settings from a JSON file.",
    'save_settings': "Save the current simulation settings to a JSON file.",
    'power_add': "Add the pads currently selected in KiCad PCB Editor to the heat-source list.",
    'power_remove': "Remove the selected rows from the heat-source list.",
    'power_clear': "Remove all manual heat sources from the list.",
    'current_enable': "Enable DC current-flow simulation for Joule heating in traces, vias, and copper pours.",
    'current_balance': "Live per-flow current balance and net validation summary.",
    'current_flows': "Pad-to-pad DC current flows. Each flow defines current from From (source) pads to To (sink) pads on the same net.",
    'flow_new': "Create a new pad-to-pad current flow.",
    'flow_duplicate': "Duplicate the selected current flow and its pads.",
    'flow_delete': "Delete the selected current flow.",
    'flow_name': "Descriptive label for the selected current flow (e.g. +12V Load, VCore).",
    'flow_apply_name': "Save the edited flow name.",
    'flow_total': "Total DC current magnitude (A) flowing from From pads to To pads. Auto-calculated when either side uses Fixed per Pad.",
    'flow_from_sharing': "How current is shared across From pads: Equipotential (Auto) lets copper resistance decide; Equal Split divides evenly; Fixed per Pad uses explicit values.",
    'flow_to_sharing': "How current is shared across To pads: Equipotential (Auto) lets copper resistance decide; Equal Split divides evenly; Fixed per Pad uses explicit values.",
    'flow_add_pads': "Add the pads currently selected in KiCad PCB Editor to this side of the flow.",
    'flow_pick_pads': "Open the Pad Picker dialog to filter and select pads by net or search text.",
    'flow_remove_pads': "Remove the selected pad rows from this side of the flow.",
    'flow_per_pad': "Current magnitude (A) applied to selected pad rows (or all rows on this side if none is selected).",
    'flow_apply_pad': "Apply the Set A value to selected pad rows (or all pads on this side).",
    'flow_pad_list': "Comma-separated non-negative per-pad current magnitudes in table order, e.g. 3, 3, 3 or 6, 4, 2.",
    'flow_apply_list': "Apply the comma-separated current values to the pads on this side in table order.",
    'flow_from_pads': "Source pads where current enters the PCB copper for this flow.",
    'flow_to_pads': "Sink pads where current leaves the PCB copper for this flow.",
    'current_groups': "Pad groups used as current sources or sinks. Positive current enters the PCB; negative current leaves it.",
    'current_mode': "Choose whether to assign individual currents per pad or distribute a total current evenly.",
    'board_details': "Expand to inspect the detected PCB layer stackup and initially selected KiCad pads.",
    'geometry_pane': "Expand to filter traces or crop the simulation area around active heat sources and current nets.",
    'thermal_pad_pane': "Expand to configure a bottom-layer thermal pad / heatsink boundary (shapes on User.Eco1).",
    'solver_pane': "Expand to adjust top/bottom convection, via plating/fill, PCB thickness fallback, compute engine, and transient time stepping.",
}


CURRENT_GROUP_COLORS = [
    "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
    "#9467bd", "#17becf", "#8c564b", "#e377c2",
]


DEFAULT_GRID_MAX_CELLS = 200000
DEFAULT_GRID_TARGET_CELLS = 100000
GRID_DETAIL_LABELS = ["Fast", "Balanced", "Detailed", "Very detailed", "Custom"]
GRID_DETAIL_VALUES = ["fast", "balanced", "detailed", "very_detailed", "custom"]
GRID_DETAIL_PRESETS = {
    "fast": (1_000_000, 850_000),
    "balanced": (3_000_000, 2_550_000),
    "detailed": (8_000_000, 6_800_000),
    "very_detailed": (16_000_000, 13_600_000),
}
DEFAULT_GRID_DETAIL = "balanced"
DEFAULT_GRID_NODE_BUDGET = GRID_DETAIL_PRESETS[DEFAULT_GRID_DETAIL][0]
MAX_CUSTOM_GRID_NODES = 100_000_000

HEATSINK_MATERIAL_OPTIONS = [
    ("Aluminium (897 J/kg\u00b7K)", "aluminium", 897.0),
    ("Copper (385 J/kg\u00b7K)", "copper", 385.0),
    ("Steel (490 J/kg\u00b7K)", "steel", 490.0),
    ("Magnesium (1020 J/kg\u00b7K)", "magnesium", 1020.0),
    ("Custom", "custom", None),
]


def resolve_heatsink_cp(material, custom_cp=897.0):
    """Return specific heat c_p in J/(kg*K) for a heatsink material preset or custom value."""
    mat_key = str(material or "aluminium").strip().lower()
    for _, key, preset_cp in HEATSINK_MATERIAL_OPTIONS:
        if key == mat_key and preset_cp is not None:
            return float(preset_cp)
    try:
        return float(custom_cp)
    except (TypeError, ValueError):
        return 897.0


def _set_control_enabled(control, enabled):
    """Enable or disable a wx control safely across real wx and headless mocks."""
    if hasattr(control, "Enable"):
        control.Enable(bool(enabled))
    else:
        control._enabled = bool(enabled)


def _safe_float(value, default=0.0):
    """Parse a float and fall back to a default for UI data."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    """Parse an int and fall back to a default for UI data."""
    try:
        return int(float(value))
    except Exception:
        return int(default)


def normalize_current_mode(mode):
    """
    Normalize saved and UI current-entry modes to stable internal values.

    Parameters
    ----------
    mode : str
        Saved mode value or display label.

    Returns
    -------
    str
        Either ``per_pad`` or ``total``.
    """
    value = str(mode or "").strip().lower()
    if value in ("per_pad", "per-pad currents", "strom pro pad", "pad current", "pad currents"):
        return "per_pad"
    if value in ("total", "distribute total current", "gesamtstrom verteilen", "distributed"):
        return "total"
    return "per_pad"


def normalize_pad_descriptor(pad):
    """
    Return a serializable pad descriptor with stable keys.

    Parameters
    ----------
    pad : dict
        Partially populated pad descriptor.

    Returns
    -------
    dict
        Normalized descriptor suitable for settings persistence.
    """
    pad = dict(pad or {})
    current = _safe_float(pad.get('current_a', pad.get('current', 0.0)))
    return {
        'pad_key': str(pad.get('pad_key', pad.get('key', ''))),
        'name': str(pad.get('name', '')),
        'net_name': str(pad.get('net_name', pad.get('net', ''))),
        'net_code': int(_safe_float(pad.get('net_code', 0), 0)),
        'layer': str(pad.get('layer', '')),
        'current_a': current,
    }


def normalize_power_pad_descriptor(pad, default_power="1.0"):
    """
    Return a serializable manual-power pad descriptor.

    Parameters
    ----------
    pad : dict
        Pad descriptor from saved settings or live KiCad selection.
    default_power : str, optional
        Power value to use when the descriptor has no explicit power value.

    Returns
    -------
    dict
        Normalized power-pad descriptor suitable for settings persistence.
    """
    pad = dict(pad or {})
    power = pad.get('power', pad.get('power_str', pad.get('power_w', default_power)))
    if power is None:
        power = default_power
    return {
        'pad_key': str(pad.get('pad_key', pad.get('key', ''))),
        'name': str(pad.get('name', '')),
        'net_name': str(pad.get('net_name', pad.get('net', ''))),
        'net_code': int(_safe_float(pad.get('net_code', 0), 0)),
        'layer': str(pad.get('layer', '')),
        'power': str(power).strip(),
    }


def prepare_power_pads(pads, power_str="1.0"):
    """
    Normalize manual-power pads and apply legacy power strings when needed.

    Parameters
    ----------
    pads : list of dict
        Raw power pad descriptors.
    power_str : str, optional
        Legacy single/comma-separated power input.

    Returns
    -------
    list of dict
        Serializable power-pad descriptors.
    """
    raw_pads = list(pads or [])
    entries = [part.strip() for part in str(power_str or "").split(",") if part.strip()]
    if not entries:
        entries = ["0.0"]

    prepared = []
    for idx, raw in enumerate(raw_pads):
        has_power = any(key in raw for key in ("power", "power_str", "power_w"))
        if has_power:
            default_power = entries[0]
        elif len(entries) == 1:
            default_power = entries[0]
        elif idx < len(entries):
            default_power = entries[idx]
        else:
            default_power = "0.0"
        prepared.append(normalize_power_pad_descriptor(raw, default_power=default_power))
    return prepared


def power_pads_to_power_str(power_pads, fallback=""):
    """
    Serialize power-pad values to the legacy ``power_str`` setting.

    Parameters
    ----------
    power_pads : list of dict
        Normalized power-pad descriptors.
    fallback : str
        Value used when no power pads are configured.

    Returns
    -------
    str
        Single value if all pads share the same power, otherwise comma-separated.
    """
    values = [str(pad.get('power', '')).strip() for pad in (power_pads or [])]
    values = [value for value in values if value]
    if not values:
        return str(fallback)
    if all(value == values[0] for value in values):
        return values[0]
    return ", ".join(values)


def summarize_power_pads(power_pads):
    """Return a compact heat-source summary for the persistent header.

    Parameters
    ----------
    power_pads : list of dict
        Normalized manual-power pad descriptors.

    Returns
    -------
    str
        Source count plus total constant power, or a PWL indicator.
    """
    pads = list(power_pads or [])
    if not pads:
        return "0 heat sources"
    total = 0.0
    contains_pwl = False
    for pad in pads:
        try:
            total += float(str(pad.get('power', '')).strip())
        except (TypeError, ValueError):
            contains_pwl = True
    count_label = f"{len(pads)} heat source" + ("" if len(pads) == 1 else "s")
    if contains_pwl:
        return f"{count_label} (contains PWL)"
    return f"{count_label} / {total:.6g} W"


def prepare_current_groups(groups):
    """
    Normalize groups and apply their current distribution mode.

    Parameters
    ----------
    groups : list of dict
        Raw group dictionaries from UI or saved settings.

    Returns
    -------
    list of dict
        Serializable groups with pad currents resolved.
    """
    prepared = []
    for idx, raw in enumerate(groups or []):
        pads = [normalize_pad_descriptor(p) for p in raw.get('pads', [])]
        mode = normalize_current_mode(raw.get('mode', 'per_pad'))
        total_current = _safe_float(raw.get('total_current_a', raw.get('total_current', 0.0)))
        if mode == 'total' and pads:
            per_pad = total_current / float(len(pads))
            for pad in pads:
                pad['current_a'] = per_pad
        elif mode == 'per_pad':
            total_current = float(sum(_safe_float(p.get('current_a')) for p in pads))
        prepared.append({
            'name': str(raw.get('name') or f"Group {idx + 1}"),
            'color': str(raw.get('color') or CURRENT_GROUP_COLORS[idx % len(CURRENT_GROUP_COLORS)]),
            'mode': mode,
            'total_current_a': total_current,
            'pads': pads,
        })
    return prepared


def summarize_current_groups(groups):
    """
    Build user-readable group and net balance summaries.

    Parameters
    ----------
    groups : list of dict
        Current groups.

    Returns
    -------
    tuple
        (group_rows, balance_rows) where each row is a tuple of strings.
    """
    prepared = prepare_current_groups(groups)
    group_rows = []
    net_totals = {}
    for group in prepared:
        nets = sorted({p.get('net_name') or "(no net)" for p in group.get('pads', [])})
        if not nets:
            net_label = "-"
        elif len(nets) == 1:
            net_label = nets[0]
        else:
            net_label = "Mixed nets: " + ", ".join(nets)
        total_current = float(sum(p.get('current_a', 0.0) for p in group.get('pads', [])))
        group_rows.append((
            group.get('name', ''),
            net_label,
            str(len(group.get('pads', []))),
            f"{total_current:.6g} A",
        ))
        for pad in group.get('pads', []):
            net = pad.get('net_name') or "(no net)"
            net_totals[net] = net_totals.get(net, 0.0) + float(pad.get('current_a', 0.0))
    balance_rows = []
    for net, total in sorted(net_totals.items()):
        status = "OK" if abs(total) <= max(1e-9, 1e-6 * abs(total)) else "Needs balance"
        balance_rows.append((net, f"{total:.9g} A", status))
    return group_rows, balance_rows


SHARING_MODE_OPTIONS = [
    ("Equipotential (Auto)", "equipotential"),
    ("Equal Split", "equal"),
    ("Fixed per Pad", "fixed"),
]
SHARING_MODE_LABELS = [label for label, _ in SHARING_MODE_OPTIONS]
SHARING_MODE_KEYS = [key for _, key in SHARING_MODE_OPTIONS]


def normalize_sharing_mode(mode):
    """
    Normalize a flow-side sharing mode to ``equipotential``, ``equal``, or ``fixed``.

    Parameters
    ----------
    mode : str
        Saved sharing mode or UI display label.

    Returns
    -------
    str
        One of ``"equipotential"``, ``"equal"``, or ``"fixed"``.
    """
    value = str(mode or "").strip().lower()
    if value in (
        "equipotential", "equipotential (auto)", "auto", "auto (cu)", "copper", "cu", "bus"
    ):
        return "equipotential"
    if value in ("equal", "equal split", "equal_split", "split", "even", "total", "distributed"):
        return "equal"
    if value in ("fixed", "fixed per pad", "fixed_per_pad", "per_pad", "per-pad", "manual"):
        return "fixed"
    return "equipotential"


def current_groups_to_pad_currents(groups):
    """
    Aggregate signed per-pad currents across legacy current groups.

    Parameters
    ----------
    groups : list of dict
        Legacy current-group dictionaries.

    Returns
    -------
    dict
        Mapping from ``pad_key`` (or ``name``) to summed signed current in Amperes.
    """
    totals = {}
    for group in prepare_current_groups(groups):
        for pad in group.get("pads", []):
            key = pad.get("pad_key") or pad.get("name") or ""
            if not key:
                continue
            totals[key] = totals.get(key, 0.0) + float(pad.get("current_a", 0.0))
    return totals


def _normalize_flow_pad(pad):
    """Normalize a pad descriptor inside a flow, enforcing non-negative magnitude."""
    desc = normalize_pad_descriptor(pad)
    desc["current_a"] = abs(_safe_float(desc.get("current_a", 0.0), 0.0))
    return desc


def _flow_pad_nets(flow):
    """Return sorted distinct non-empty net names across both sides of a flow."""
    all_pads = list(flow.get("from_pads", [])) + list(flow.get("to_pads", []))
    return sorted({p.get("net_name", "") for p in all_pads if p.get("net_name")})


def prepare_current_flows(flows):
    """
    Normalize pad-to-pad current flows and auto-balance sharing modes.

    Parameters
    ----------
    flows : list of dict
        Raw current-flow dictionaries from UI or saved settings.

    Returns
    -------
    list of dict
        Serializable flow dictionaries matching the canonical v0.5.0 schema.
    """
    prepared = []
    for idx, raw in enumerate(flows or []):
        if not isinstance(raw, dict):
            continue
        from_pads = [_normalize_flow_pad(p) for p in raw.get("from_pads", [])]
        to_pads = [_normalize_flow_pad(p) for p in raw.get("to_pads", [])]
        from_sharing = normalize_sharing_mode(raw.get("from_sharing", "equipotential"))
        to_sharing = normalize_sharing_mode(raw.get("to_sharing", "fixed"))
        total_current_a = abs(
            _safe_float(raw.get("total_current_a", raw.get("total_current", 0.0)), 0.0)
        )

        if to_sharing == "fixed" and from_sharing != "fixed":
            total_current_a = float(sum(p["current_a"] for p in to_pads))
        elif from_sharing == "fixed" and to_sharing != "fixed":
            total_current_a = float(sum(p["current_a"] for p in from_pads))
        elif from_sharing == "fixed" and to_sharing == "fixed":
            sum_from = float(sum(p["current_a"] for p in from_pads))
            sum_to = float(sum(p["current_a"] for p in to_pads))
            total_current_a = sum_from if from_pads else sum_to

        if from_sharing == "equal" and from_pads:
            per_pad = total_current_a / float(len(from_pads))
            for pad in from_pads:
                pad["current_a"] = per_pad
        elif from_sharing == "equipotential":
            if len(from_pads) == 1:
                from_pads[0]["current_a"] = total_current_a
            else:
                for pad in from_pads:
                    pad["current_a"] = 0.0

        if to_sharing == "equal" and to_pads:
            per_pad = total_current_a / float(len(to_pads))
            for pad in to_pads:
                pad["current_a"] = per_pad
        elif to_sharing == "equipotential":
            if len(to_pads) == 1:
                to_pads[0]["current_a"] = total_current_a
            else:
                for pad in to_pads:
                    pad["current_a"] = 0.0

        all_pads = from_pads + to_pads
        net_name = str(raw.get("net_name", "") or "")
        net_code = _safe_int(raw.get("net_code", 0), 0)
        non_empty_nets = [p for p in all_pads if p.get("net_name")]
        if non_empty_nets:
            unique_nets = {p["net_name"] for p in non_empty_nets}
            if len(unique_nets) == 1:
                net_name = non_empty_nets[0]["net_name"]
                if non_empty_nets[0]["net_code"]:
                    net_code = int(non_empty_nets[0]["net_code"])
            elif not net_name:
                net_name = non_empty_nets[0]["net_name"]
                net_code = int(non_empty_nets[0]["net_code"])

        prepared.append({
            "name": str(raw.get("name") or f"Flow {idx + 1}"),
            "net_name": net_name,
            "net_code": net_code,
            "total_current_a": float(total_current_a),
            "from_sharing": from_sharing,
            "to_sharing": to_sharing,
            "from_pads": from_pads,
            "to_pads": to_pads,
        })
    return prepared


def convert_current_groups_to_flows(groups):
    """
    Losslessly convert legacy ``current_groups`` into ``current_flows`` per net.

    Positive-current pads become ``from_pads`` and negative-current pads become
    ``to_pads`` (with non-negative magnitudes), grouped by net name.

    Parameters
    ----------
    groups : list of dict
        Legacy current-group dictionaries.

    Returns
    -------
    list of dict
        Normalized pad-to-pad flow dictionaries.
    """
    prepared_groups = prepare_current_groups(groups)
    if not prepared_groups:
        return []

    buckets = {}
    order = []
    for group in prepared_groups:
        gname = str(group.get("name") or "").strip()
        for pad in group.get("pads", []):
            net_name = str(pad.get("net_name") or "")
            net_code = _safe_int(pad.get("net_code", 0), 0)
            if net_name not in buckets:
                buckets[net_name] = {
                    "net_name": net_name,
                    "net_code": net_code,
                    "group_names": [],
                    "from_pads": [],
                    "to_pads": [],
                    "zero_pads": [],
                }
                order.append(net_name)
            bucket = buckets[net_name]
            if net_code and not bucket["net_code"]:
                bucket["net_code"] = net_code
            if gname and gname not in bucket["group_names"]:
                bucket["group_names"].append(gname)

            current = _safe_float(pad.get("current_a", 0.0), 0.0)
            norm_pad = _normalize_flow_pad(pad)
            if current > 0.0:
                norm_pad["current_a"] = float(current)
                bucket["from_pads"].append(norm_pad)
            elif current < 0.0:
                norm_pad["current_a"] = float(abs(current))
                bucket["to_pads"].append(norm_pad)
            else:
                norm_pad["current_a"] = 0.0
                bucket["zero_pads"].append(norm_pad)

    flows = []
    for idx, net_name in enumerate(order):
        bucket = buckets[net_name]
        from_pads = bucket["from_pads"]
        to_pads = bucket["to_pads"]
        if not from_pads and not to_pads and bucket["zero_pads"]:
            from_pads = bucket["zero_pads"]
        if len(bucket["group_names"]) == 1:
            flow_name = bucket["group_names"][0]
        elif net_name:
            flow_name = f"{net_name} Flow"
        else:
            flow_name = f"Flow {idx + 1}"
        sum_from = sum(p["current_a"] for p in from_pads)
        sum_to = sum(p["current_a"] for p in to_pads)
        flows.append({
            "name": flow_name,
            "net_name": net_name,
            "net_code": bucket["net_code"],
            "total_current_a": sum_from if from_pads else sum_to,
            "from_sharing": "fixed",
            "to_sharing": "fixed",
            "from_pads": from_pads,
            "to_pads": to_pads,
        })
    return prepare_current_flows(flows)


def summarize_current_flows(flows):
    """
    Return a compact summary of configured current flows (e.g. ``"2 flows / 15 A"``).

    Parameters
    ----------
    flows : list of dict
        Current-flow dictionaries.

    Returns
    -------
    str
        Compact human-readable summary string.
    """
    prepared = prepare_current_flows(flows)
    if not prepared:
        return "0 flows"
    total_a = sum(float(flow.get("total_current_a", 0.0)) for flow in prepared)
    count = len(prepared)
    noun = "flow" if count == 1 else "flows"
    return f"{count} {noun} / {total_a:.6g} A"


def validate_current_flows(flows):
    """
    Validate pad-to-pad current flows and return ``(errors, warnings)``.

    Checks for missing From/To pads, unassigned nets, mismatched nets within a
    flow, fixed/fixed current mismatches, zero-current flows, and duplicate pads
    appearing on both sides of the same flow.

    Parameters
    ----------
    flows : list of dict
        Current-flow dictionaries.

    Returns
    -------
    tuple of (list of str, list of str)
        ``(errors, warnings)`` describing any flow balance or configuration issues.
    """
    prepared = prepare_current_flows(flows)
    errors = []
    warnings = []
    for flow in prepared:
        name = flow.get("name", "Flow")
        from_pads = flow.get("from_pads", [])
        to_pads = flow.get("to_pads", [])
        all_pads = from_pads + to_pads
        nets = _flow_pad_nets(flow)
        net_label = flow.get("net_name") or (nets[0] if len(nets) == 1 else "(no net)")

        if not from_pads:
            errors.append(f"{name}: missing From pads.")
        if not to_pads:
            errors.append(f"{name}: missing To pads.")

        if all_pads and any(not p.get("net_name") for p in all_pads):
            errors.append(f"{name}: one or more pads have no net assigned.")
        if len(nets) > 1:
            errors.append(f"{name}: mismatched nets ({', '.join(nets)}).")

        if flow.get("from_sharing") == "fixed" and flow.get("to_sharing") == "fixed":
            sum_from = sum(float(p.get("current_a", 0.0)) for p in from_pads)
            sum_to = sum(float(p.get("current_a", 0.0)) for p in to_pads)
            tol = max(1e-9, 1e-6 * max(abs(sum_from), abs(sum_to)))
            if abs(sum_from - sum_to) > tol:
                errors.append(
                    f"{name} ({net_label}): From ({sum_from:.6g} A) != To ({sum_to:.6g} A)."
                )

        if from_pads and to_pads:
            from_ids = {
                p.get("pad_key") or p.get("name")
                for p in from_pads if (p.get("pad_key") or p.get("name"))
            }
            overlap = [
                p.get("name") or p.get("pad_key")
                for p in to_pads
                if (p.get("pad_key") or p.get("name")) in from_ids
            ]
            if overlap:
                warnings.append(
                    f"{name} ({net_label}): pad in both From and To ({', '.join(overlap)})."
                )
            if float(flow.get("total_current_a", 0.0)) <= 1e-12:
                warnings.append(f"{name} ({net_label}): total current is 0 A.")

    return errors, warnings


def _format_flow_balance_lines(flows):
    """Build per-flow live status lines for the Current Heating balance box."""
    prepared = prepare_current_flows(flows)
    if not prepared:
        return ["No current flows configured."]
    lines = []
    for flow in prepared:
        from_pads = flow.get("from_pads", [])
        to_pads = flow.get("to_pads", [])
        all_pads = from_pads + to_pads
        nets = _flow_pad_nets(flow)
        if len(nets) > 1:
            net_label = f"Mixed ({', '.join(nets)})"
        elif nets:
            net_label = nets[0]
        else:
            net_label = flow.get("net_name") or "(no net)"
        total_a = float(flow.get("total_current_a", 0.0))

        if not from_pads or not to_pads:
            status = "Needs From/To pads"
        elif any(not p.get("net_name") for p in all_pads):
            status = "No net"
        elif len(nets) > 1:
            status = "Mismatched nets"
        elif flow.get("from_sharing") == "fixed" and flow.get("to_sharing") == "fixed":
            sum_from = sum(float(p.get("current_a", 0.0)) for p in from_pads)
            sum_to = sum(float(p.get("current_a", 0.0)) for p in to_pads)
            tol = max(1e-9, 1e-6 * max(abs(sum_from), abs(sum_to)))
            if abs(sum_from - sum_to) > tol:
                status = f"Mismatch (From {sum_from:.6g} A != To {sum_to:.6g} A)"
            elif total_a <= 1e-12:
                status = "0 A"
            else:
                status = "OK"
        elif total_a <= 1e-12:
            status = "0 A"
        else:
            status = "OK"
        lines.append(
            f"{net_label}: {total_a:.6g} A ({len(from_pads)} -> {len(to_pads)} pads, {status})"
        )
    return lines


class PadPickerDialog(wx.Dialog):
    """
    Modal dialog allowing engineers to filter and pick board pads by net or search text.

    Parameters
    ----------
    parent : wx.Window or None
        Parent window.
    board_pads : list of dict
        Available copper pad descriptors from the board.
    initial_net : str, optional
        Net name to pre-select in the filter dropdown.
    title : str, optional
        Dialog window title.
    """

    def __init__(self, parent, board_pads, initial_net="", title="Pick Pads from Board"):
        dialog_style = (
            getattr(wx, "DEFAULT_DIALOG_STYLE", 0)
            | getattr(wx, "RESIZE_BORDER", 0)
        )
        super().__init__(parent, title=title, style=dialog_style)

        seen_keys = set()
        self.all_pads = []
        for raw in (board_pads or []):
            desc = _normalize_flow_pad(raw)
            ident = desc["pad_key"] or (desc["name"], desc["net_name"], desc["layer"])
            if ident in seen_keys:
                continue
            seen_keys.add(ident)
            self.all_pads.append(desc)

        self.all_pads.sort(key=lambda p: (p.get("net_name", ""), p.get("name", "")))
        self.filtered_pads = []

        nets = sorted({p.get("net_name") or "(no net)" for p in self.all_pads})
        self._net_choices = ["All Nets"] + nets

        sizer = wx.BoxSizer(wx.VERTICAL)

        filter_row = wx.BoxSizer(wx.HORIZONTAL)
        filter_row.Add(wx.StaticText(self, label="Net:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.net_choice = wx.Choice(self, choices=self._net_choices)
        if initial_net and initial_net in nets:
            self.net_choice.SetSelection(self._net_choices.index(initial_net))
        else:
            self.net_choice.SetSelection(0)
        self.net_choice.Bind(wx.EVT_CHOICE, self._on_filter_changed)
        filter_row.Add(self.net_choice, 1, wx.EXPAND | wx.RIGHT, 8)

        filter_row.Add(wx.StaticText(self, label="Search:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.search_input = wx.TextCtrl(self, value="")
        evt_text = getattr(wx, "EVT_TEXT", None)
        if evt_text is not None:
            self.search_input.Bind(evt_text, self._on_filter_changed)
        filter_row.Add(self.search_input, 1, wx.EXPAND)
        sizer.Add(filter_row, 0, wx.EXPAND | wx.ALL, 8)

        self.pad_list = wx.ListCtrl(self, style=getattr(wx, "LC_REPORT", 0))
        for idx, (col_title, width) in enumerate([
            ("Pad", 180), ("Net", 140), ("Layer", 80),
        ]):
            self.pad_list.InsertColumn(idx, col_title, width=width)
        self.pad_list.SetMinSize((440, 260))
        sizer.Add(self.pad_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        btn_all = wx.Button(self, label="Select All")
        btn_all.Bind(wx.EVT_BUTTON, self._on_select_all)
        btn_row.Add(btn_all, 0, wx.RIGHT, 6)
        btn_row.AddStretchSpacer()
        btn_ok = wx.Button(self, wx.ID_OK, label="Add Selected")
        btn_ok.SetDefault()
        btn_ok.Bind(wx.EVT_BUTTON, self._on_ok)
        btn_cancel = wx.Button(self, wx.ID_CANCEL, label="Cancel")
        btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
        btn_row.Add(btn_ok, 0, wx.RIGHT, 6)
        btn_row.Add(btn_cancel, 0)
        sizer.Add(btn_row, 0, wx.EXPAND | wx.ALL, 8)

        self.SetSizer(sizer)
        self.SetSize((480, 380))
        self.SetMinSize((420, 320))
        self.Center()
        self._apply_filter()

    def get_selected_net(self):
        """Return the active net filter, or empty string when 'All Nets' is selected."""
        idx = self.net_choice.GetSelection()
        if idx <= 0 or idx >= len(self._net_choices):
            return ""
        return self._net_choices[idx]

    def set_net_filter(self, net_name):
        """Select a net filter programmatically and refresh the table."""
        if net_name and net_name in self._net_choices:
            self.net_choice.SetSelection(self._net_choices.index(net_name))
        else:
            self.net_choice.SetSelection(0)
        self._apply_filter()

    def set_search_text(self, text):
        """Set search filter text programmatically and refresh the table."""
        self.search_input.SetValue(str(text or ""))
        self._apply_filter()

    def _on_filter_changed(self, event):
        """Refresh visible pads when net or search filter changes."""
        self._apply_filter()

    def _apply_filter(self):
        """Filter ``all_pads`` by net and search query and update ``pad_list``."""
        selected_net = self.get_selected_net()
        query = self.search_input.GetValue().strip().lower()
        filtered = []
        for pad in self.all_pads:
            pad_net = pad.get("net_name") or "(no net)"
            if selected_net and pad_net != selected_net:
                continue
            if query:
                haystack = f"{pad.get('name', '')} {pad_net} {pad.get('layer', '')}".lower()
                if query not in haystack:
                    continue
            filtered.append(pad)
        self.filtered_pads = filtered
        try:
            self.pad_list.DeleteAllItems()
            for row_idx, pad in enumerate(self.filtered_pads):
                self.pad_list.InsertItem(row_idx, pad.get("name", ""))
                self.pad_list.SetItem(row_idx, 1, pad.get("net_name") or "(no net)")
                self.pad_list.SetItem(row_idx, 2, pad.get("layer", ""))
        except Exception:
            pass

    def _on_select_all(self, event):
        """Select all currently filtered pad rows."""
        for row_idx in range(len(self.filtered_pads)):
            try:
                self.pad_list.Select(row_idx)
            except Exception:
                pass

    def _on_ok(self, event):
        """Confirm pad selection."""
        try:
            self.EndModal(wx.ID_OK)
        except Exception:
            pass

    def _on_cancel(self, event):
        """Cancel pad selection."""
        try:
            self.EndModal(wx.ID_CANCEL)
        except Exception:
            pass

    def get_selected_pads(self):
        """Return copies of all currently selected filtered pad descriptors."""
        indices = []
        try:
            idx = self.pad_list.GetFirstSelected()
            while idx != -1:
                indices.append(idx)
                idx = self.pad_list.GetNextItem(
                    idx, getattr(wx, "LIST_NEXT_ALL", 0), getattr(wx, "LIST_STATE_SELECTED", 0)
                )
        except Exception:
            pass
        return [
            copy.deepcopy(self.filtered_pads[i])
            for i in indices
            if 0 <= i < len(self.filtered_pads)
        ]


class SettingsDialog(wx.Dialog):
    """
    Dialog for configuring thermal simulation parameters.

    Organized into four notebook tabs for overview, heat sources,
    current heating, and advanced settings.

    Parameters
    ----------
    parent : wx.Window or None
        Parent window for the dialog.
    selected_count : int
        Number of selected pads.
    suggested_res : float
        Suggested grid resolution in mm.
    layer_names : list of str
        Names of copper layers.
    preview_callback : callable, optional
        Function to call when Preview button is clicked.
        Signature: callback(settings_dict, layer_names).
    stackup_details : str, optional
        Formatted stackup information to display.
    pad_names : list of str, optional
        Names of selected pads with net info.
    default_output_dir : str, optional
        Default output directory path.
    defaults : dict, optional
        Default values to pre-fill in the dialog.
    defer_initial_preflight : bool, optional
        Schedule the first board scan after the dialog becomes visible.

    Attributes
    ----------
    layer_names : list of str
        Stored layer names for preview callback.
    preview_callback : callable or None
        Stored preview callback function.
    """

    def __init__(
        self,
        parent,
        selected_count,
        suggested_res,
        layer_names,
        preview_callback=None,
        selection_provider=None,
        run_callback=None,
        close_callback=None,
        preflight_callback=None,
        load_settings_callback=None,
        save_settings_callback=None,
        stackup_details="",
        pad_names=None,
        initial_power_pads=None,
        default_output_dir="",
        defaults=None,
        board_name="",
        board_size_mm=None,
        defer_initial_preflight=False,
        stackup_thickness_mm=None,
        board_pads=None,
    ):
        dialog_style = (
            getattr(wx, "DEFAULT_DIALOG_STYLE", 0)
            | getattr(wx, "RESIZE_BORDER", 0)
        )
        super().__init__(parent, title="Thermal Sim", style=dialog_style)

        self.layer_names = layer_names
        self.board_name = str(board_name or "Unsaved board")
        self.board_size_mm = tuple(board_size_mm or ())
        try:
            stack_th = float(stackup_thickness_mm) if stackup_thickness_mm is not None else None
        except (TypeError, ValueError):
            stack_th = None
        self.stackup_thickness_mm = stack_th if (stack_th is not None and stack_th > 0.0) else None
        self.preview_callback = preview_callback
        self.selection_provider = selection_provider
        self.run_callback = run_callback
        self.close_callback = close_callback
        self.preflight_callback = preflight_callback
        self.load_settings_callback = load_settings_callback
        self.save_settings_callback = save_settings_callback
        self.current_groups = []
        self.current_group_index = -1
        self.current_flows = []
        self.current_flow_index = -1
        self._current_flows_edited = False
        self.board_pads = [_normalize_flow_pad(p) for p in (board_pads or [])]
        self.power_pads = []
        self.initial_power_pads = [
            normalize_power_pad_descriptor(pad) for pad in (initial_power_pads or [])
        ]
        self._power_pads_edited = False
        self.last_report_path = None
        self.last_run_dir = None
        self.last_run_status = "idle"
        self._preflight_debounce_ms = 150
        self._preflight_timer = None
        self._init_preflight_timer()

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Persistent board and simulation context.
        header = wx.BoxSizer(wx.VERTICAL)
        self.lbl_board_name = wx.StaticText(self, label=self.board_name)
        title_font = self.lbl_board_name.GetFont()
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.lbl_board_name.SetFont(title_font)
        header.Add(self.lbl_board_name, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        size_text = ""
        if len(self.board_size_mm) == 2:
            size_text = f"{self.board_size_mm[0]:.1f} x {self.board_size_mm[1]:.1f} mm / "
        self.lbl_board_meta = wx.StaticText(
            self,
            label=f"{size_text}{len(layer_names)} copper layers",
        )
        header.Add(self.lbl_board_meta, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)
        self.lbl_context = wx.StaticText(
            self,
            label="0 heat sources / Current heating off",
        )
        header.Add(self.lbl_context, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, 12)
        main_sizer.Add(header, 0, wx.EXPAND)
        main_sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 12)

        # --- Notebook ---
        self.notebook = wx.Notebook(self)

        # Tab 1: Simulation
        self.tab_sim = wx.Panel(self.notebook)
        self._build_simulation_tab(
            self.tab_sim, layer_names, stackup_details,
            pad_names, suggested_res, default_output_dir
        )
        self.notebook.AddPage(self.tab_sim, "Overview")

        # Tab 2: Power pads
        self.tab_power = wx.Panel(self.notebook)
        self._build_power_tab(self.tab_power)
        self.notebook.AddPage(self.tab_power, "Heat Sources")

        # Tab 3: Current heating
        self.tab_current = wx.Panel(self.notebook)
        self._build_current_tab(self.tab_current)
        self.notebook.AddPage(self.tab_current, "Current Heating")

        # Tab 4: Advanced
        self.tab_adv = wx.Panel(self.notebook)
        self._build_advanced_tab(self.tab_adv)
        self.notebook.AddPage(self.tab_adv, "Advanced")

        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)

        # Compact completion summary, hidden until a run finishes.
        self.result_panel = wx.Panel(self)
        result_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.lbl_result = wx.StaticText(self.result_panel, label="")
        result_sizer.Add(self.lbl_result, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.btn_open_report = wx.Button(self.result_panel, label="Open Report")
        self.btn_open_report.SetToolTip(TOOLTIP_TEXTS['open_report'])
        self.btn_open_report.Bind(wx.EVT_BUTTON, self._on_open_report)
        self.btn_open_report.Enable(False)
        result_sizer.Add(self.btn_open_report, 0, wx.RIGHT, 6)
        self.btn_open_folder = wx.Button(self.result_panel, label="Open Folder")
        self.btn_open_folder.SetToolTip(TOOLTIP_TEXTS['open_folder'])
        self.btn_open_folder.Bind(wx.EVT_BUTTON, self._on_open_folder)
        self.btn_open_folder.Enable(False)
        result_sizer.Add(self.btn_open_folder, 0)
        self.result_panel.SetSizer(result_sizer)
        self.result_panel.Show(False)
        main_sizer.Add(self.result_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        main_sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        # Keep status text and actions in separate child panels.  Native
        # StaticText controls may otherwise paint beyond their horizontal
        # sizer allocation and cover adjacent buttons when a warning is long.
        self.footer_status_panel = wx.Panel(self)
        status_sizer = wx.BoxSizer(wx.VERTICAL)
        self.lbl_preflight_status = wx.StaticText(
            self.footer_status_panel, label="Checking setup..."
        )
        status_font = self.lbl_preflight_status.GetFont()
        status_font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.lbl_preflight_status.SetFont(status_font)
        status_sizer.Add(self.lbl_preflight_status, 0, wx.EXPAND)
        self.lbl_preflight = wx.StaticText(
            self.footer_status_panel, label="Checking simulation setup..."
        )
        status_sizer.Add(self.lbl_preflight, 0, wx.EXPAND | wx.TOP, 2)
        self.footer_status_panel.SetSizer(status_sizer)
        main_sizer.Add(
            self.footer_status_panel,
            0,
            wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP,
            12,
        )

        self.footer_actions_panel = wx.Panel(self)
        footer = wx.BoxSizer(wx.HORIZONTAL)
        footer.AddStretchSpacer()

        self.btn_more = wx.Button(self.footer_actions_panel, label="More")
        self.btn_more.SetToolTip(TOOLTIP_TEXTS['more'])
        self.btn_more.Bind(wx.EVT_BUTTON, self._on_more)
        footer.Add(self.btn_more, 0, wx.ALL, 4)

        self.btn_preview = wx.Button(self.footer_actions_panel, label="Preview")
        self.btn_preview.Bind(wx.EVT_BUTTON, self._on_preview)
        self.btn_preview.SetToolTip(TOOLTIP_TEXTS['preview'])
        footer.Add(self.btn_preview, 0, wx.ALL, 4)

        self.btn_run = wx.Button(self.footer_actions_panel, label="Run Simulation")
        self.btn_run.SetToolTip(TOOLTIP_TEXTS['run'])
        self.btn_run.Bind(wx.EVT_BUTTON, self._on_run)
        self.btn_cancel = wx.Button(self.footer_actions_panel, label="Close")
        self.btn_cancel.SetToolTip(TOOLTIP_TEXTS['cancel'])
        self.btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
        footer.Add(self.btn_run, 0, wx.ALL, 4)
        footer.Add(self.btn_cancel, 0, wx.ALL, 4)
        # Do not make Run the dialog default button.  Spin controls accept
        # typed values and users commonly confirm them with Enter; starting a
        # simulation from that key press is surprising and potentially costly.

        try:
            self.Bind(wx.EVT_CLOSE, self._on_cancel)
        except Exception:
            pass

        self.footer_actions_panel.SetSizer(footer)
        main_sizer.Add(
            self.footer_actions_panel,
            0,
            wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
            8,
        )

        self.SetSizer(main_sizer)
        self.SetSize((820, 720))
        self.SetMinSize((760, 640))
        self.Center()

        if defaults:
            self._apply_defaults(defaults)
        else:
            self.power_pads = prepare_power_pads(self.initial_power_pads, self.power_input.GetValue())
            self._render_power_pads()
            self._render_current_flows()
        self._bind_estimate_controls()
        self._refresh_context_summary()
        if defer_initial_preflight:
            self._set_footer_status(
                "Dialog ready", "Checking board and simulation settings..."
            )
            try:
                wx.CallAfter(self._refresh_preflight)
            except Exception:
                self._refresh_preflight()
        else:
            self._refresh_preflight()

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_simulation_tab(self, panel, layer_names, stackup_details,
                              pad_names, suggested_res, default_output_dir):
        """Build the compact Overview tab contents."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Board detail stays available without dominating the page.
        self.board_details = wx.CollapsiblePane(panel, label="Board details")
        self.board_details.SetToolTip(TOOLTIP_TEXTS['board_details'])
        details_panel = self.board_details.GetPane()
        details_sizer = wx.BoxSizer(wx.VERTICAL)
        self.txt_stackup = wx.TextCtrl(
            details_panel, value=stackup_details or "No stackup details available.",
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
        )
        self.txt_stackup.SetMinSize((-1, 80))
        self.txt_stackup.SetToolTip(TOOLTIP_TEXTS['stackup'])
        details_sizer.Add(self.txt_stackup, 0, wx.EXPAND | wx.BOTTOM, 6)
        pad_lines = pad_names if isinstance(pad_names, (list, tuple)) else []
        pad_text = "\n".join(str(x) for x in pad_lines)
        self.txt_pads = wx.TextCtrl(
            details_panel, value=pad_text or "No pads were selected when the dialog opened.",
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
        )
        self.txt_pads.SetMinSize((-1, 60))
        self.txt_pads.SetToolTip(TOOLTIP_TEXTS['pads'])
        details_sizer.Add(self.txt_pads, 0, wx.EXPAND)
        details_panel.SetSizer(details_sizer)

        # --- Parameters ---
        box_params = wx.StaticBoxSizer(wx.VERTICAL, panel, "Simulation")
        params_parent = box_params.GetStaticBox()

        # Duration
        self.time_input = self._add_spin_field(
            box_params, params_parent, "Duration (s)", 20.0,
            min_val=0.1, max_val=3600.0, inc=1.0, digits=1,
            tooltip_key='duration'
        )

        self.chk_steady_state = wx.CheckBox(
            params_parent, label="Steady-state (final temperature)"
        )
        self.chk_steady_state.SetValue(False)
        self.chk_steady_state.SetToolTip(TOOLTIP_TEXTS['steady_state'])
        self.chk_steady_state.Bind(wx.EVT_CHECKBOX, self._on_steady_state_toggle)
        box_params.Add(self.chk_steady_state, 0, wx.ALL, 3)

        # Ambient Temperature
        self.amb_input = self._add_spin_field(
            box_params, params_parent, "Ambient (\u00b0C)", 25.0,
            min_val=-40.0, max_val=200.0, inc=1.0, digits=1,
            tooltip_key='ambient'
        )

        # Desired physical resolution and layer-aware compute budget.
        self.res_input = self._add_spin_field(
            box_params, params_parent, "Target cell size (mm)", suggested_res,
            min_val=0.05, max_val=10.0, inc=0.05, digits=2,
            tooltip_key='resolution'
        )

        detail_row = wx.BoxSizer(wx.HORIZONTAL)
        detail_row.Add(
            wx.StaticText(params_parent, label="Compute budget", size=(160, -1)),
            0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5,
        )
        self.grid_detail_choice = wx.Choice(params_parent, choices=GRID_DETAIL_LABELS)
        self.grid_detail_choice.SetSelection(1)
        self.grid_detail_choice.SetToolTip(TOOLTIP_TEXTS['grid_detail'])
        self.grid_detail_choice.Bind(wx.EVT_CHOICE, self._on_grid_detail_changed)
        detail_row.Add(self.grid_detail_choice, 1, wx.EXPAND)
        box_params.Add(detail_row, 0, wx.EXPAND | wx.ALL, 2)

        self.grid_node_budget_input = self._add_int_spin_field(
            box_params, params_parent, "Maximum solver nodes", DEFAULT_GRID_NODE_BUDGET,
            min_val=50000, max_val=MAX_CUSTOM_GRID_NODES,
            tooltip_key='grid_node_budget',
        )
        self.grid_node_budget_input.Enable(False)
        self.lbl_grid_estimate = wx.StaticText(
            params_parent,
            label="Estimate will update from the active board and sources.",
        )
        self.lbl_grid_estimate.SetToolTip(TOOLTIP_TEXTS['grid_estimate'])
        box_params.Add(self.lbl_grid_estimate, 0, wx.EXPAND | wx.ALL, 3)

        sizer.Add(box_params, 0, wx.EXPAND | wx.ALL, 5)

        # --- Output ---
        box_out = wx.StaticBoxSizer(wx.VERTICAL, panel, "Output")
        output_parent = box_out.GetStaticBox()

        output_options = wx.BoxSizer(wx.HORIZONTAL)
        self.chk_all_layers = wx.CheckBox(output_parent, label="Show all copper layers")
        self.chk_all_layers.SetValue(True)
        self.chk_all_layers.SetToolTip(TOOLTIP_TEXTS['show_all'])
        output_options.Add(self.chk_all_layers, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 18)

        self.chk_snapshots = wx.CheckBox(output_parent, label="Save snapshots")
        self.chk_snapshots.SetValue(False)
        self.chk_snapshots.SetToolTip(TOOLTIP_TEXTS['snapshots'])
        output_options.Add(self.chk_snapshots, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        output_options.Add(
            wx.StaticText(output_parent, label="Count"),
            0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5,
        )
        self.snap_count_input = wx.SpinCtrl(
            output_parent, value="5", min=1, max=50,
            style=getattr(wx, "SP_ARROW_KEYS", 0),
        )
        self.snap_count_input.SetToolTip(TOOLTIP_TEXTS['snap_count'])
        self.snap_count_input.Enable(False)
        output_options.Add(self.snap_count_input, 0, wx.ALIGN_CENTER_VERTICAL)
        box_out.Add(output_options, 0, wx.EXPAND | wx.ALL, 3)
        self.chk_snapshots.Bind(wx.EVT_CHECKBOX, self._on_snapshots_toggle)

        # Output folder
        row_path = wx.BoxSizer(wx.HORIZONTAL)
        lbl_path = wx.StaticText(output_parent, label="Output folder", size=(160, -1))
        self.output_dir_input = wx.TextCtrl(output_parent, value=default_output_dir)
        self.output_dir_input.SetToolTip(TOOLTIP_TEXTS['output_dir'])
        btn_browse = wx.Button(output_parent, label="Browse...")
        btn_browse.Bind(wx.EVT_BUTTON, self._on_browse_output)
        btn_browse.SetToolTip(TOOLTIP_TEXTS['browse_output'])
        row_path.Add(lbl_path, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        row_path.Add(self.output_dir_input, 1, wx.EXPAND | wx.RIGHT, 5)
        row_path.Add(btn_browse, 0)
        box_out.Add(row_path, 0, wx.EXPAND | wx.ALL, 3)

        sizer.Add(box_out, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        sizer.Add(self.board_details, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        panel.SetSizer(sizer)

    def _build_power_tab(self, panel):
        """Build the manual power-pad tab."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        help_text = wx.StaticText(
            panel,
            label="Configure manual heat sources here. Current-heating terminals are selected separately."
        )
        sizer.Add(help_text, 0, wx.EXPAND | wx.ALL, 5)

        edit_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Heat Sources")
        edit_parent = edit_box.GetStaticBox()
        row_pwr = wx.BoxSizer(wx.HORIZONTAL)
        row_pwr.Add(wx.StaticText(edit_parent, label="Power / PWL", size=(105, -1)), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.power_input = wx.TextCtrl(edit_parent, value="1.0")
        self.power_input.SetToolTip(TOOLTIP_TEXTS['power'])
        row_pwr.Add(self.power_input, 1, wx.EXPAND | wx.RIGHT, 5)
        btn_browse_pwl = wx.Button(edit_parent, label="Browse...")
        btn_browse_pwl.Bind(wx.EVT_BUTTON, self._on_browse_pwl)
        btn_browse_pwl.SetToolTip(TOOLTIP_TEXTS['browse_pwl'])
        row_pwr.Add(btn_browse_pwl, 0, wx.RIGHT, 5)
        btn_apply = wx.Button(edit_parent, label="Apply")
        btn_apply.Bind(wx.EVT_BUTTON, self._on_power_apply_value)
        btn_apply.SetToolTip(TOOLTIP_TEXTS['power_apply'])
        row_pwr.Add(btn_apply, 0)
        edit_box.Add(row_pwr, 0, wx.EXPAND | wx.ALL, 3)

        edit_buttons = wx.BoxSizer(wx.HORIZONTAL)
        btn_add = wx.Button(edit_parent, label="Add Selected")
        btn_add.SetToolTip(TOOLTIP_TEXTS['power_add'])
        btn_add.Bind(wx.EVT_BUTTON, self._on_power_add_selection)
        edit_buttons.Add(btn_add, 0, wx.ALL, 2)
        btn_remove = wx.Button(edit_parent, label="Remove")
        btn_remove.SetToolTip(TOOLTIP_TEXTS['power_remove'])
        btn_remove.Bind(wx.EVT_BUTTON, self._on_power_remove_pads)
        edit_buttons.Add(btn_remove, 0, wx.ALL, 2)
        btn_clear = wx.Button(edit_parent, label="Clear")
        btn_clear.SetToolTip(TOOLTIP_TEXTS['power_clear'])
        btn_clear.Bind(wx.EVT_BUTTON, self._on_power_clear_pads)
        edit_buttons.Add(btn_clear, 0, wx.ALL, 2)
        edit_box.Add(edit_buttons, 0, wx.EXPAND | wx.ALL, 3)

        self.power_pad_list = wx.ListCtrl(
            edit_parent,
            style=getattr(wx, "LC_REPORT", 0)
        )
        for idx, (title, width) in enumerate([
            ("Pad", 210), ("Net", 150), ("Layer", 80), ("Power W/PWL", 180),
        ]):
            self.power_pad_list.InsertColumn(idx, title, width=width)
        self.power_pad_list.SetMinSize((-1, 300))
        self.power_pad_list.SetToolTip(TOOLTIP_TEXTS['power_pads'])
        activated_event = getattr(wx, "EVT_LIST_ITEM_ACTIVATED", None)
        if activated_event is not None:
            self.power_pad_list.Bind(activated_event, self._on_power_edit_row)
        edit_box.Add(self.power_pad_list, 1, wx.EXPAND | wx.ALL, 3)
        self.lbl_heat_summary = wx.StaticText(edit_parent, label="0 heat sources")
        edit_box.Add(self.lbl_heat_summary, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        sizer.Add(edit_box, 1, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(sizer)

    def _build_advanced_tab(self, panel):
        """Build the Advanced tab contents."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Geometry Filters ---
        self.geometry_pane = wx.CollapsiblePane(panel, label="Geometry")
        self.geometry_pane.SetToolTip(TOOLTIP_TEXTS['geometry_pane'])
        self.geometry_pane.Expand()
        filter_panel = self.geometry_pane.GetPane()
        box_filter = wx.BoxSizer(wx.VERTICAL)

        self.chk_ignore_traces = wx.CheckBox(filter_panel, label="Ignore traces")
        self.chk_ignore_traces.SetValue(False)
        self.chk_ignore_traces.SetToolTip(TOOLTIP_TEXTS['ignore_traces'])
        box_filter.Add(self.chk_ignore_traces, 0, wx.ALL, 3)

        self.chk_limit_area = wx.CheckBox(
            filter_panel, label="Limit to active sources and current paths"
        )
        self.chk_limit_area.SetValue(False)
        self.chk_limit_area.SetToolTip(TOOLTIP_TEXTS['limit_area'])
        box_filter.Add(self.chk_limit_area, 0, wx.ALL, 3)

        self.pad_dist_input = self._add_spin_field(
            box_filter, filter_panel, "Thermal margin (mm)", 20.0,
            min_val=0.0, max_val=200.0, inc=5.0, digits=1,
            tooltip_key='limit_dist'
        )
        self.pad_dist_input.Enable(False)
        self.chk_limit_area.Bind(wx.EVT_CHECKBOX, self._on_limit_area_toggle)
        self.lbl_area_estimate = wx.StaticText(
            filter_panel, label="Area: full board"
        )
        self.lbl_area_estimate.SetToolTip(TOOLTIP_TEXTS['area_estimate'])
        box_filter.Add(self.lbl_area_estimate, 0, wx.EXPAND | wx.ALL, 3)

        filter_panel.SetSizer(box_filter)
        sizer.Add(self.geometry_pane, 0, wx.EXPAND | wx.ALL, 5)

        # --- Thermal Pad ---
        self.thermal_pad_pane = wx.CollapsiblePane(panel, label="Thermal Pad (User.Eco1)")
        self.thermal_pad_pane.SetToolTip(TOOLTIP_TEXTS['thermal_pad_pane'])
        pad_panel = self.thermal_pad_pane.GetPane()
        box_pad = wx.BoxSizer(wx.VERTICAL)

        self.chk_heatsink = wx.CheckBox(pad_panel, label="Enable pad simulation")
        self.chk_heatsink.SetValue(False)
        self.chk_heatsink.SetToolTip(TOOLTIP_TEXTS['enable_pad'])
        self.chk_heatsink.Bind(wx.EVT_CHECKBOX, self._on_heatsink_toggle)
        box_pad.Add(self.chk_heatsink, 0, wx.ALL, 3)

        self.pad_thick = self._add_spin_field(
            box_pad, pad_panel, "Pad thickness (mm)", 1.0,
            min_val=0.1, max_val=50.0, inc=0.5, digits=2,
            tooltip_key='pad_thick'
        )

        self.pad_k = self._add_spin_field(
            box_pad, pad_panel, "Conductivity (W/mK)", 3.0,
            min_val=0.01, max_val=500.0, inc=1.0, digits=1,
            tooltip_key='pad_k'
        )

        self.pad_cap = self._add_spin_field(
            box_pad, pad_panel, "Heat capacity (J/m\u00b2K)", 0.0,
            min_val=0.0, max_val=100000.0, inc=100.0, digits=0,
            tooltip_key='pad_cap'
        )
        self.pad_contact_factor = self._add_spin_field(
            box_pad, pad_panel, "Contact factor", 0.2,
            min_val=0.05, max_val=1.0, inc=0.05, digits=2,
            tooltip_key='pad_contact_factor'
        )

        choice_event = getattr(wx, "EVT_CHOICE", None)
        spin_double_event = getattr(wx, "EVT_SPINCTRLDOUBLE", None)
        focus_event = getattr(wx, "EVT_KILL_FOCUS", None)

        sink_mode_row = wx.BoxSizer(wx.HORIZONTAL)
        sink_mode_row.Add(
            wx.StaticText(pad_panel, label="Sink model", size=(160, -1)),
            0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5,
        )
        self.pad_sink_mode_choice = wx.Choice(
            pad_panel,
            choices=["Fixed temperature", "Dynamic heatsink (0D)"],
        )
        self.pad_sink_mode_choice.SetSelection(0)
        self.pad_sink_mode_choice.SetToolTip(TOOLTIP_TEXTS['pad_sink_mode'])
        if choice_event is not None:
            self.pad_sink_mode_choice.Bind(choice_event, self._on_pad_sink_mode_changed)
        sink_mode_row.Add(self.pad_sink_mode_choice, 1, wx.EXPAND)
        box_pad.Add(sink_mode_row, 0, wx.EXPAND | wx.ALL, 2)

        self.chk_pad_sink_ambient = wx.CheckBox(
            pad_panel, label="Use ambient temperature"
        )
        self.chk_pad_sink_ambient.SetValue(True)
        self.chk_pad_sink_ambient.SetToolTip(TOOLTIP_TEXTS['pad_sink_temp_use_ambient'])
        self.chk_pad_sink_ambient.Bind(wx.EVT_CHECKBOX, self._on_pad_sink_ambient_toggle)
        box_pad.Add(self.chk_pad_sink_ambient, 0, wx.ALL, 3)

        self.pad_sink_temp_input = self._add_spin_field(
            box_pad, pad_panel, "Fixed sink temp (C)", 25.0,
            min_val=-50.0, max_val=300.0, inc=1.0, digits=1,
            tooltip_key='pad_sink_temp_c'
        )

        self.heatsink_rth_input = self._add_spin_field(
            box_pad, pad_panel, "Heatsink R_th (K/W)", 5.0,
            min_val=0.05, max_val=500.0, inc=0.5, digits=2,
            tooltip_key='heatsink_rth_kw'
        )

        material_row = wx.BoxSizer(wx.HORIZONTAL)
        material_row.Add(
            wx.StaticText(pad_panel, label="Heatsink material", size=(160, -1)),
            0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5,
        )
        self.heatsink_material_choice = wx.Choice(
            pad_panel,
            choices=[label for label, _, _ in HEATSINK_MATERIAL_OPTIONS],
        )
        self.heatsink_material_choice.SetSelection(0)
        self.heatsink_material_choice.SetToolTip(TOOLTIP_TEXTS['heatsink_material'])
        if choice_event is not None:
            self.heatsink_material_choice.Bind(choice_event, self._on_heatsink_material_changed)
        material_row.Add(self.heatsink_material_choice, 1, wx.EXPAND)
        box_pad.Add(material_row, 0, wx.EXPAND | wx.ALL, 2)

        self.heatsink_mass_input = self._add_spin_field(
            box_pad, pad_panel, "Heatsink mass (g)", 50.0,
            min_val=0.0, max_val=50000.0, inc=5.0, digits=1,
            tooltip_key='heatsink_mass_g'
        )

        self.heatsink_cp_input = self._add_spin_field(
            box_pad, pad_panel, "Specific heat c_p (J/kg\u00b7K)", 897.0,
            min_val=10.0, max_val=5000.0, inc=10.0, digits=1,
            tooltip_key='heatsink_cp_j_kgk'
        )

        for ctrl in (self.heatsink_rth_input, self.heatsink_mass_input, self.heatsink_cp_input):
            if spin_double_event is not None and hasattr(ctrl, "Bind"):
                ctrl.Bind(spin_double_event, self._on_heatsink_param_changed)
            if focus_event is not None and hasattr(ctrl, "Bind"):
                ctrl.Bind(focus_event, self._on_heatsink_param_changed)

        self.lbl_heatsink_summary = wx.StaticText(pad_panel, label="")
        self.lbl_heatsink_summary.SetToolTip(TOOLTIP_TEXTS['heatsink_summary'])
        box_pad.Add(self.lbl_heatsink_summary, 0, wx.EXPAND | wx.ALL, 3)

        self._thermal_pad_controls = [
            self.pad_thick,
            self.pad_k,
            self.pad_cap,
            self.pad_contact_factor,
        ]
        self._apply_thermal_pad_controls()

        pad_panel.SetSizer(box_pad)
        sizer.Add(self.thermal_pad_pane, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # --- Solver ---
        self.solver_pane = wx.CollapsiblePane(panel, label="Solver")
        self.solver_pane.SetToolTip(TOOLTIP_TEXTS['solver_pane'])
        solver_panel = self.solver_pane.GetPane()
        box_solver = wx.BoxSizer(wx.VERTICAL)

        self.h_conv_input = self._add_spin_field(
            box_solver, solver_panel, "Convection top (W/m\u00b2K)", 10.0,
            min_val=0.0, max_val=200.0, inc=1.0, digits=1,
            tooltip_key='h_conv'
        )

        self.h_conv_bottom_input = self._add_spin_field(
            box_solver, solver_panel, "Convection bottom (W/m\u00b2K)", 10.0,
            min_val=0.0, max_val=200.0, inc=1.0, digits=1,
            tooltip_key='h_conv_bottom'
        )

        self.via_plating_input = self._add_spin_field(
            box_solver, solver_panel, "Via plating (\u00b5m)", 25.0,
            min_val=5.0, max_val=100.0, inc=5.0, digits=1,
            tooltip_key='via_plating_um'
        )

        via_fill_row = wx.BoxSizer(wx.HORIZONTAL)
        via_fill_row.Add(
            wx.StaticText(solver_panel, label="Via fill", size=(165, -1)),
            0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5
        )
        self.via_fill_choice = wx.Choice(
            solver_panel,
            choices=["Plated (standard)", "Copper-filled"],
        )
        self.via_fill_choice.SetSelection(0)
        self.via_fill_choice.SetToolTip(TOOLTIP_TEXTS['via_fill'])
        if choice_event is not None:
            self.via_fill_choice.Bind(choice_event, self._on_via_fill_changed)
        via_fill_row.Add(self.via_fill_choice, 1, wx.EXPAND)
        box_solver.Add(via_fill_row, 0, wx.EXPAND | wx.ALL, 3)

        self.thick_input = self._add_spin_field(
            box_solver, solver_panel, "PCB thickness (mm)", 1.6,
            min_val=0.1, max_val=10.0, inc=0.1, digits=2,
            tooltip_key='pcb_thick'
        )
        if self.stackup_thickness_mm is not None:
            self.thick_input.SetValue(self.stackup_thickness_mm)
            self.thick_input.Enable(False)
            self.thick_input.SetToolTip(
                "Auto-detected from the PCB stackup (only editable as fallback when stackup has no thickness)."
            )
        self._apply_via_fill_controls()

        engine_row = wx.BoxSizer(wx.HORIZONTAL)
        engine_row.Add(
            wx.StaticText(solver_panel, label="Compute engine", size=(165, -1)),
            0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5
        )
        self.compute_engine_choice = wx.Choice(
            solver_panel,
            choices=["Auto", "Fast CPU (matrix-free)", "Legacy sparse"],
        )
        self.compute_engine_choice.SetSelection(0)
        self.compute_engine_choice.SetToolTip(TOOLTIP_TEXTS['compute_engine'])
        engine_row.Add(self.compute_engine_choice, 1, wx.EXPAND)
        box_solver.Add(engine_row, 0, wx.EXPAND | wx.ALL, 3)

        stepping_row = wx.BoxSizer(wx.HORIZONTAL)
        stepping_row.Add(wx.StaticText(solver_panel, label="Time stepping", size=(165, -1)), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.time_stepping_choice = wx.Choice(
            solver_panel,
            choices=["Auto", "Legacy 3-phase", "2-phase", "Uniform BDF2"],
        )
        self.time_stepping_choice.SetSelection(0)
        self.time_stepping_choice.SetToolTip(TOOLTIP_TEXTS['time_stepping'])
        stepping_row.Add(self.time_stepping_choice, 1, wx.EXPAND)
        box_solver.Add(stepping_row, 0, wx.EXPAND | wx.ALL, 3)

        solver_panel.SetSizer(box_solver)
        sizer.Add(self.solver_pane, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # --- Capabilities (read-only) ---
        if HAS_NATIVE_MG:
            engine_str = "Engine: Native C++/OpenMP + SciPy"
        elif HAS_NATIVE_CORE:
            engine_str = "Engine: Native C++ + SciPy"
        else:
            engine_str = "Engine: NumPy + SciPy"
        self.lbl_capabilities = wx.StaticText(panel, label=engine_str)
        self.lbl_capabilities.SetToolTip(TOOLTIP_TEXTS['capabilities'])
        sizer.Add(self.lbl_capabilities, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        pane_event = getattr(wx, "EVT_COLLAPSIBLEPANE_CHANGED", None)
        if pane_event is not None:
            for pane in (self.geometry_pane, self.thermal_pad_pane, self.solver_pane):
                pane.Bind(pane_event, self._on_advanced_pane_changed)

        panel.SetSizer(sizer)

    def _build_current_tab(self, panel):
        """Build the Pad-to-Pad Current Flows / Joule-heating tab."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.chk_current_enabled = wx.CheckBox(panel, label="Enable current heating")
        self.chk_current_enabled.SetValue(False)
        self.chk_current_enabled.SetToolTip(TOOLTIP_TEXTS['current_enable'])
        self.chk_current_enabled.Bind(wx.EVT_CHECKBOX, self._on_current_enabled_toggle)
        sizer.Add(self.chk_current_enabled, 0, wx.ALL, 5)

        help_text = wx.StaticText(
            panel,
            label="Define pad-to-pad DC current flows (From -> To) on each net."
        )
        sizer.Add(help_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        balance_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Flow Balance")
        balance_parent = balance_box.GetStaticBox()
        self.current_balance_text = wx.TextCtrl(
            balance_parent, value="", style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
        )
        self.current_balance_text.SetMinSize((-1, 44))
        self.current_balance_text.SetToolTip(TOOLTIP_TEXTS['current_balance'])
        balance_box.Add(self.current_balance_text, 0, wx.EXPAND | wx.ALL, 3)
        sizer.Add(balance_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        current_content = wx.BoxSizer(wx.HORIZONTAL)

        flow_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Current Flows")
        flow_parent = flow_box.GetStaticBox()
        self.current_flow_list = wx.ListCtrl(
            flow_parent,
            style=getattr(wx, "LC_REPORT", 0) | getattr(wx, "LC_SINGLE_SEL", 0)
        )
        for idx, (title, width) in enumerate([
            ("Name", 70), ("Net", 65), ("From->To", 60), ("Current", 55),
        ]):
            self.current_flow_list.InsertColumn(idx, title, width=width)
        self.current_flow_list.SetToolTip(TOOLTIP_TEXTS['current_flows'])
        self.current_flow_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_current_flow_selected)
        self.current_flow_list.SetMinSize((255, -1))
        flow_box.Add(self.current_flow_list, 1, wx.EXPAND | wx.ALL, 3)

        flow_buttons = wx.BoxSizer(wx.HORIZONTAL)
        btn_new = wx.Button(flow_parent, label="New Flow")
        btn_new.SetToolTip(TOOLTIP_TEXTS['flow_new'])
        btn_new.Bind(wx.EVT_BUTTON, self._on_flow_new)
        flow_buttons.Add(btn_new, 0, wx.ALL, 2)
        btn_duplicate = wx.Button(flow_parent, label="Duplicate")
        btn_duplicate.SetToolTip(TOOLTIP_TEXTS['flow_duplicate'])
        btn_duplicate.Bind(wx.EVT_BUTTON, self._on_flow_duplicate)
        flow_buttons.Add(btn_duplicate, 0, wx.ALL, 2)
        btn_delete = wx.Button(flow_parent, label="Delete")
        btn_delete.SetToolTip(TOOLTIP_TEXTS['flow_delete'])
        btn_delete.Bind(wx.EVT_BUTTON, self._on_flow_delete)
        flow_buttons.Add(btn_delete, 0, wx.ALL, 2)
        flow_box.Add(flow_buttons, 0, wx.EXPAND | wx.ALL, 3)
        current_content.Add(flow_box, 0, wx.EXPAND | wx.RIGHT, 5)

        edit_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Selected Flow")
        selected_parent = edit_box.GetStaticBox()

        row_top = wx.BoxSizer(wx.HORIZONTAL)
        row_top.Add(wx.StaticText(selected_parent, label="Name:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.flow_name_input = wx.TextCtrl(selected_parent, value="")
        self.flow_name_input.SetToolTip(TOOLTIP_TEXTS['flow_name'])
        row_top.Add(self.flow_name_input, 1, wx.EXPAND | wx.RIGHT, 4)
        btn_apply_name = wx.Button(selected_parent, label="Apply")
        btn_apply_name.SetToolTip(TOOLTIP_TEXTS['flow_apply_name'])
        btn_apply_name.Bind(wx.EVT_BUTTON, self._on_flow_apply_fields)
        row_top.Add(btn_apply_name, 0, wx.RIGHT, 8)

        row_top.Add(wx.StaticText(selected_parent, label="Total (A):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.flow_total_input = wx.SpinCtrlDouble(
            selected_parent, value="0.0", min=0.0, max=10000.0, inc=0.5
        )
        self.flow_total_input.SetDigits(3)
        self.flow_total_input.SetMinSize((88, -1))
        self.flow_total_input.SetToolTip(TOOLTIP_TEXTS['flow_total'])
        self.flow_total_input.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_flow_total_changed)
        row_top.Add(self.flow_total_input, 0)
        edit_box.Add(row_top, 0, wx.EXPAND | wx.ALL, 3)

        self._flow_side_widgets = {}
        from_box = self._build_flow_side_box(
            selected_parent, "from", "From (Source Pads)", default_sharing_idx=0
        )
        edit_box.Add(from_box, 1, wx.EXPAND | wx.ALL, 2)

        to_box = self._build_flow_side_box(
            selected_parent, "to", "To (Sink Pads)", default_sharing_idx=2
        )
        edit_box.Add(to_box, 1, wx.EXPAND | wx.ALL, 2)

        current_content.Add(edit_box, 1, wx.EXPAND)
        sizer.Add(current_content, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        panel.SetSizer(sizer)

    def _build_flow_side_box(self, parent, side, title, default_sharing_idx=0):
        """Build a compact From or To pad-side editor box."""
        side_box = wx.StaticBoxSizer(wx.VERTICAL, parent, title)
        box_parent = side_box.GetStaticBox()

        top_row = wx.BoxSizer(wx.HORIZONTAL)
        top_row.Add(wx.StaticText(box_parent, label="Sharing:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        sharing_choice = wx.Choice(box_parent, choices=SHARING_MODE_LABELS)
        sharing_choice.SetSelection(default_sharing_idx)
        sharing_choice.SetToolTip(TOOLTIP_TEXTS[f'flow_{side}_sharing'])
        sharing_choice.Bind(wx.EVT_CHOICE, lambda evt, s=side: self._on_flow_sharing_changed(s))
        top_row.Add(sharing_choice, 0, wx.RIGHT, 6)

        btn_add = wx.Button(box_parent, label="Add Selected")
        btn_add.SetToolTip(TOOLTIP_TEXTS['flow_add_pads'])
        btn_add.Bind(wx.EVT_BUTTON, lambda evt, s=side: self._on_flow_add_selection(s))
        top_row.Add(btn_add, 0, wx.RIGHT, 3)

        btn_pick = wx.Button(box_parent, label="Pick...")
        btn_pick.SetToolTip(TOOLTIP_TEXTS['flow_pick_pads'])
        btn_pick.Bind(wx.EVT_BUTTON, lambda evt, s=side: self._on_flow_pick_pads(s))
        top_row.Add(btn_pick, 0, wx.RIGHT, 3)

        btn_remove = wx.Button(box_parent, label="Remove")
        btn_remove.SetToolTip(TOOLTIP_TEXTS['flow_remove_pads'])
        btn_remove.Bind(wx.EVT_BUTTON, lambda evt, s=side: self._on_flow_remove_pads(s))
        top_row.Add(btn_remove, 0)
        side_box.Add(top_row, 0, wx.EXPAND | wx.ALL, 2)

        fixed_row = wx.BoxSizer(wx.HORIZONTAL)
        fixed_row.Add(wx.StaticText(box_parent, label="Set A:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        pad_value_input = wx.SpinCtrlDouble(
            box_parent, value="0.0", min=0.0, max=10000.0, inc=0.1
        )
        pad_value_input.SetDigits(3)
        pad_value_input.SetMinSize((76, -1))
        pad_value_input.SetToolTip(TOOLTIP_TEXTS['flow_per_pad'])
        fixed_row.Add(pad_value_input, 0, wx.RIGHT, 3)

        btn_set_pad = wx.Button(box_parent, label="Set")
        btn_set_pad.SetToolTip(TOOLTIP_TEXTS['flow_apply_pad'])
        btn_set_pad.Bind(wx.EVT_BUTTON, lambda evt, s=side: self._on_flow_apply_pad_current(s))
        fixed_row.Add(btn_set_pad, 0, wx.RIGHT, 6)

        fixed_row.Add(wx.StaticText(box_parent, label="List:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        pad_list_input = wx.TextCtrl(box_parent, value="")
        pad_list_input.SetToolTip(TOOLTIP_TEXTS['flow_pad_list'])
        fixed_row.Add(pad_list_input, 1, wx.EXPAND | wx.RIGHT, 3)

        btn_apply_list = wx.Button(box_parent, label="Apply")
        btn_apply_list.SetToolTip(TOOLTIP_TEXTS['flow_apply_list'])
        btn_apply_list.Bind(wx.EVT_BUTTON, lambda evt, s=side: self._on_flow_apply_pad_current_list(s))
        fixed_row.Add(btn_apply_list, 0)
        side_box.Add(fixed_row, 0, wx.EXPAND | wx.ALL, 2)

        pad_list = wx.ListCtrl(box_parent, style=getattr(wx, "LC_REPORT", 0))
        for idx, (col_title, width) in enumerate([
            ("Pad", 160), ("Layer", 70), ("Current A", 95),
        ]):
            pad_list.InsertColumn(idx, col_title, width=width)
        pad_list.SetMinSize((-1, 82))
        pad_list.SetToolTip(TOOLTIP_TEXTS[f'flow_{side}_pads'])
        side_box.Add(pad_list, 1, wx.EXPAND | wx.ALL, 2)

        setattr(self, f"flow_{side}_sharing", sharing_choice)
        setattr(self, f"flow_{side}_fixed_row", fixed_row)
        setattr(self, f"flow_{side}_pad_value_input", pad_value_input)
        setattr(self, f"flow_{side}_pad_list_input", pad_list_input)
        setattr(self, f"flow_{side}_list", pad_list)
        self._flow_side_widgets[side] = {
            "sharing": sharing_choice,
            "fixed_row": fixed_row,
            "pad_value": pad_value_input,
            "pad_list_input": pad_list_input,
            "list_ctrl": pad_list,
        }
        return side_box

    # ------------------------------------------------------------------
    # Helper: add spinner fields
    # ------------------------------------------------------------------

    def _add_spin_field(self, sizer, parent, label_text, default_val,
                        min_val=0.0, max_val=1000.0, inc=1.0, digits=1,
                        tooltip_key=None):
        """
        Add a labeled SpinCtrlDouble field to a sizer.

        Parameters
        ----------
        sizer : wx.Sizer
            Parent sizer to add the field to.
        parent : wx.Window
            Parent window for the controls.
        label_text : str
            Label text for the field.
        default_val : float
            Default value for the spinner.
        min_val : float
            Minimum allowed value.
        max_val : float
            Maximum allowed value.
        inc : float
            Increment per spinner click.
        digits : int
            Number of decimal places to display.
        tooltip_key : str, optional
            Key into TOOLTIP_TEXTS for this control.

        Returns
        -------
        wx.SpinCtrlDouble
            The created spinner control.
        """
        row = wx.BoxSizer(wx.HORIZONTAL)
        lbl = wx.StaticText(parent, label=label_text, size=(160, -1))
        spin = wx.SpinCtrlDouble(
            parent, value=str(default_val),
            min=min_val, max=max_val, inc=inc
        )
        spin.SetDigits(digits)
        if tooltip_key and tooltip_key in TOOLTIP_TEXTS:
            spin.SetToolTip(TOOLTIP_TEXTS[tooltip_key])
        row.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        row.Add(spin, 1, wx.EXPAND)
        sizer.Add(row, 0, wx.EXPAND | wx.ALL, 2)
        spin._thermal_row_sizer = row
        return spin

    def _add_int_spin_field(self, sizer, parent, label_text, default_val,
                            min_val=0, max_val=100, tooltip_key=None):
        """
        Add a labeled SpinCtrl (integer) field to a sizer.

        Parameters
        ----------
        sizer : wx.Sizer
            Parent sizer to add the field to.
        parent : wx.Window
            Parent window for the controls.
        label_text : str
            Label text for the field.
        default_val : int
            Default value for the spinner.
        min_val : int
            Minimum allowed value.
        max_val : int
            Maximum allowed value.
        tooltip_key : str, optional
            Key into TOOLTIP_TEXTS for this control.

        Returns
        -------
        wx.SpinCtrl
            The created spinner control.
        """
        row = wx.BoxSizer(wx.HORIZONTAL)
        lbl = wx.StaticText(parent, label=label_text, size=(160, -1))
        spin = wx.SpinCtrl(
            parent, value=str(default_val),
            min=min_val, max=max_val
        )
        if tooltip_key and tooltip_key in TOOLTIP_TEXTS:
            spin.SetToolTip(TOOLTIP_TEXTS[tooltip_key])
        row.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        row.Add(spin, 1, wx.EXPAND)
        sizer.Add(row, 0, wx.EXPAND | wx.ALL, 2)
        spin._thermal_row_sizer = row
        return spin

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _bind_estimate_controls(self):
        """Bind controls that materially change the live cost estimate."""
        bindings = (
            (self.res_input, getattr(wx, "EVT_SPINCTRLDOUBLE", None)),
            (self.pad_dist_input, getattr(wx, "EVT_SPINCTRLDOUBLE", None)),
            (self.grid_node_budget_input, getattr(wx, "EVT_SPINCTRL", None)),
        )
        for control, event_type in bindings:
            if event_type is not None and hasattr(control, "Bind"):
                control.Bind(event_type, self._on_estimate_input_changed)
            focus_event = getattr(wx, "EVT_KILL_FOCUS", None)
            if focus_event is not None and hasattr(control, "Bind"):
                control.Bind(focus_event, self._on_estimate_input_changed)

    def _settings_file_start_dir(self):
        """Return the preferred directory for settings file dialogs."""
        try:
            output_dir = self.output_dir_input.GetValue().strip()
            if output_dir and os.path.isdir(output_dir):
                return output_dir
        except Exception:
            pass
        return os.path.dirname(__file__)

    def _load_settings_file(self, path):
        """Load settings from a JSON file when no external callback is configured."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_settings_file(self, settings, path):
        """Save settings to a JSON file when no external callback is configured."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, sort_keys=True)
            return True
        except Exception:
            return False

    def _on_load_settings(self, event):
        """Handle manual settings import from a JSON file."""
        dlg = wx.FileDialog(
            self,
            message="Load ThermalSim settings",
            defaultDir=self._settings_file_start_dir(),
            wildcard="JSON files (*.json)|*.json|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        finally:
            dlg.Destroy()

        loader = self.load_settings_callback or self._load_settings_file
        settings = loader(path)
        if not isinstance(settings, dict) or not settings:
            wx.MessageBox("No valid ThermalSim settings found in this file.", "ThermalSim")
            return

        self._apply_defaults(settings)
        wx.MessageBox("Settings loaded.", "ThermalSim")

    def _on_save_settings(self, event):
        """Handle manual settings export to a JSON file."""
        settings = self.get_values()
        if not settings:
            wx.MessageBox("Invalid simulation settings. Settings were not saved.", "ThermalSim")
            return

        save_style = getattr(wx, "FD_SAVE", 0) | getattr(wx, "FD_OVERWRITE_PROMPT", 0)
        dlg = wx.FileDialog(
            self,
            message="Save ThermalSim settings",
            defaultDir=self._settings_file_start_dir(),
            defaultFile="thermal_sim_settings.json",
            wildcard="JSON files (*.json)|*.json|All files (*.*)|*.*",
            style=save_style,
        )
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        finally:
            dlg.Destroy()

        if path and not os.path.splitext(path)[1]:
            path += ".json"

        saver = self.save_settings_callback or self._save_settings_file
        if saver(settings, path):
            wx.MessageBox("Settings saved.", "ThermalSim")
        else:
            wx.MessageBox("Settings could not be saved.", "ThermalSim")

    def _set_footer_status(self, title, message):
        """Update and wrap the footer status without touching its action row."""
        title = str(title or "")
        message = str(message or "")
        self.lbl_preflight_status.SetLabel(title)
        self.lbl_preflight.SetLabel(message)
        try:
            self.lbl_preflight.SetToolTip(message)
        except Exception:
            pass
        try:
            width = self.footer_status_panel.GetClientSize().GetWidth() - 24
        except Exception:
            width = 700
        if width <= 0:
            width = 700
        try:
            self.lbl_preflight.Wrap(max(240, int(width)))
        except Exception:
            pass
        try:
            self.footer_status_panel.Layout()
            self.Layout()
        except Exception:
            pass

    def _flush_ui_updates(self):
        """Force immediate repaint and pending UI event processing safely."""
        for method_name in ("Refresh", "Update"):
            method = getattr(self, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
        try:
            get_app = getattr(wx, "GetApp", None)
            app = get_app() if callable(get_app) else None
            if app is not None and hasattr(app, "Yield"):
                try:
                    app.Yield(True)
                except TypeError:
                    app.Yield()
        except Exception:
            pass

    def _on_preview(self, event):
        """Handle Preview button click."""
        if self.preview_callback:
            settings = self.get_values()
            if settings and self._refresh_preflight(settings):
                self._set_footer_status(
                    "Building preview...",
                    "Extracting and rasterizing board geometry.",
                )
                self.btn_preview.Enable(False)
                self._flush_ui_updates()
                try:
                    output_path = self.preview_callback(settings, self.layer_names)
                    if output_path:
                        self._set_footer_status(
                            "Preview ready", os.path.basename(output_path)
                        )
                    elif self.lbl_preflight.GetLabel() != "Preview cancelled.":
                        self._set_footer_status(
                            "Preview failed", "No preview image was created."
                        )
                except Exception:
                    self._set_footer_status(
                        "Preview failed",
                        "The geometry preview could not be created.",
                    )
                    wx.MessageBox("Geometry preview failed.", "ThermalSim")
                finally:
                    self.btn_preview.Enable(True)
                    self._flush_ui_updates()

    def _on_run(self, event):
        """Handle Run button click for modal and modeless workflows."""
        settings = self.get_values()
        if not settings:
            wx.MessageBox("Invalid simulation settings.", "ThermalSim")
            return
        if not self._refresh_preflight(settings):
            return
        if self.run_callback:
            self.set_run_state("running")
            self.run_callback(settings)
        else:
            try:
                self.EndModal(wx.ID_OK)
            except Exception:
                pass

    def _init_preflight_timer(self):
        """Initialize a wx.Timer for debouncing preflight updates when available."""
        timer_cls = getattr(wx, "Timer", None)
        timer_event = getattr(wx, "EVT_TIMER", None)
        if timer_cls is None or timer_event is None:
            self._preflight_timer = None
            return
        try:
            timer = timer_cls(self)
            if hasattr(timer, "StartOnce") or hasattr(timer, "Start"):
                self.Bind(timer_event, self._on_preflight_timer, timer)
                self._preflight_timer = timer
            else:
                self._preflight_timer = None
        except Exception:
            self._preflight_timer = None

    def _stop_preflight_timer(self):
        """Stop any pending debounced preflight timer."""
        timer = getattr(self, "_preflight_timer", None)
        if timer is not None:
            try:
                timer.Stop()
            except Exception:
                pass

    def _schedule_preflight_refresh(self):
        """Debounce preflight refresh via wx.Timer or run synchronously if unavailable."""
        timer = getattr(self, "_preflight_timer", None)
        if timer is not None:
            try:
                if hasattr(timer, "Stop"):
                    timer.Stop()
                if hasattr(timer, "StartOnce"):
                    started = timer.StartOnce(self._preflight_debounce_ms)
                    if started is not False:
                        return
                elif hasattr(timer, "Start"):
                    one_shot = getattr(wx, "TIMER_ONE_SHOT", True)
                    started = timer.Start(self._preflight_debounce_ms, one_shot)
                    if started is not False:
                        return
            except Exception:
                pass
        self._refresh_preflight()

    def _on_preflight_timer(self, event):
        """Execute the debounced preflight refresh."""
        self._refresh_preflight()

    def _refresh_preflight(self, settings=None):
        """Refresh the permanent readiness summary and Run availability."""
        self._stop_preflight_timer()
        settings = settings if settings is not None else self.get_values()
        if settings is None:
            self._set_footer_status(
                "Blocked", "One or more numeric settings are invalid."
            )
            self.btn_run.Enable(False)
            return False
        if not self.preflight_callback:
            self._set_footer_status("Ready", "Settings are valid.")
            self.btn_run.Enable(True)
            return True
        try:
            result = self.preflight_callback(settings)
            grid = getattr(result, "grid", None)
            area = getattr(result, "area", None)
            details = []
            if grid is not None:
                details.append(
                    f"{grid.actual_res_mm:.3f} mm, {grid.rows} x {grid.cols} x "
                    f"{grid.layer_count} = {grid.nodes:,} nodes, {grid.runtime_class}"
                )
                estimate = (
                    f"Actual {grid.actual_res_mm:.3f} mm / {grid.nodes:,} nodes / "
                    f"features about {grid.feature_min_mm:.2f}-{grid.feature_max_mm:.2f} mm\n"
                    f"Estimated memory {grid.memory_mb_low:,}-{grid.memory_mb_high:,} MB / "
                    f"runtime {grid.runtime_class.lower()}"
                )
                if grid.auto_coarsened:
                    estimate += f" / requested {grid.requested_res_mm:.3f} mm"
                self.lbl_grid_estimate.SetLabel(estimate)
            if area is not None:
                percent = int(round(area.area_fraction * 100.0))
                nets = len(area.active_net_names)
                area_text = (
                    f"Area: {area.width_mm:.1f} x {area.height_mm:.1f} mm "
                    f"({percent}% of board)"
                )
                if nets:
                    area_text += f" / {nets} complete current net{'s' if nets != 1 else ''}"
                self.lbl_area_estimate.SetLabel(area_text)
                if details:
                    details[0] += f" / area {percent}%"
            messages = list(getattr(result, "errors", []) or getattr(result, "warnings", []))
            if messages:
                details.append(messages[0])
            label = details[0] if details else "Settings checked."
            if len(details) > 1:
                label += "\n" + details[1]
            self._set_footer_status(str(result.status), label)
            self.btn_run.Enable(bool(result.ready))
            return bool(result.ready)
        except Exception:
            self._set_footer_status(
                "Blocked", "Preflight could not be completed."
            )
            self.btn_run.Enable(False)
            return False

    def set_run_state(self, status, message=""):
        """Render a durable running, cancelled, or failed state."""
        self.last_run_status = str(status or "idle")
        labels = {
            "running": ("Running", "Thermal simulation is in progress..."),
            "cancelled": ("Cancelled", "The simulation was cancelled."),
            "failed": ("Failed", "The simulation did not complete."),
        }
        title, default_message = labels.get(self.last_run_status, (self.last_run_status.title(), ""))
        self._set_footer_status(title, message or default_message)
        self.btn_run.Enable(self.last_run_status != "running")
        self._flush_ui_updates()

    def set_artifacts(self, report_path, run_dir, elapsed_s=None, max_temp_c=None):
        """Expose completion summary and actions for a successful run."""
        self.last_report_path = report_path
        self.last_run_dir = run_dir
        self.last_run_status = "success"
        summary = ["Completed"]
        if max_temp_c is not None:
            summary.append(f"max {float(max_temp_c):.1f} °C")
        if elapsed_s is not None:
            summary.append(f"{float(elapsed_s):.1f} s")
        label = " / ".join(summary)
        self.lbl_result.SetLabel(label)
        self._set_footer_status("Completed", "Simulation results are ready.")
        self.btn_open_report.Enable(bool(report_path and os.path.isfile(report_path)))
        self.btn_open_folder.Enable(bool(run_dir and os.path.isdir(run_dir)))
        self.result_panel.Show(True)
        try:
            self.Layout()
        except Exception:
            pass

    def _on_open_report(self, event):
        if self.last_report_path:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(self.last_report_path))

    def _on_open_folder(self, event):
        if not self.last_run_dir:
            return
        import subprocess
        import sys
        path = os.path.abspath(self.last_run_dir)
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _on_cancel(self, event):
        """Handle Cancel button click."""
        self._stop_preflight_timer()
        self._preflight_timer = None
        if self.close_callback:
            self.close_callback()
        try:
            self.Destroy()
        except Exception:
            try:
                self.EndModal(wx.ID_CANCEL)
            except Exception:
                pass

    def _on_limit_area_toggle(self, event):
        """Handle Limit Area checkbox toggle."""
        self.pad_dist_input.Enable(self.chk_limit_area.GetValue())
        if self.chk_limit_area.GetValue():
            self.geometry_pane.Expand()
        self._refresh_preflight()

    def _apply_steady_state_controls(self):
        """Enable or disable transient-only controls based on steady-state mode."""
        steady = self.chk_steady_state.GetValue()
        self.time_input.Enable(not steady)
        self.chk_snapshots.Enable(not steady)
        self.snap_count_input.Enable((not steady) and self.chk_snapshots.GetValue())

    def _on_steady_state_toggle(self, event):
        """Handle Steady-state checkbox toggle."""
        self._apply_steady_state_controls()
        self._refresh_preflight()

    def _on_snapshots_toggle(self, event):
        """Handle Snapshots checkbox toggle."""
        self.snap_count_input.Enable(
            self.chk_snapshots.GetValue() and not self.chk_steady_state.GetValue()
        )
        self._refresh_preflight()

    def _apply_via_fill_controls(self):
        """Disable via plating thickness when copper-filled vias are selected."""
        is_copper_filled = self.via_fill_choice.GetSelection() == 1
        self.via_plating_input.Enable(not is_copper_filled)

    def _on_via_fill_changed(self, event):
        """Handle Via fill selection changes."""
        self._apply_via_fill_controls()
        self._refresh_preflight()

    def _selected_heatsink_material(self):
        """Return (material_key, preset_cp) for the active material selection."""
        idx = self.heatsink_material_choice.GetSelection()
        if 0 <= idx < len(HEATSINK_MATERIAL_OPTIONS):
            _, key, preset_cp = HEATSINK_MATERIAL_OPTIONS[idx]
            return key, preset_cp
        return "aluminium", 897.0

    def _apply_thermal_pad_controls(self):
        """Update enable states, preset c_p, and live heatsink summary."""
        enabled = bool(self.chk_heatsink.GetValue())
        is_heatsink_mode = self.pad_sink_mode_choice.GetSelection() == 1
        use_ambient = bool(self.chk_pad_sink_ambient.GetValue())
        mat_key, preset_cp = self._selected_heatsink_material()

        for control in self._thermal_pad_controls:
            control.Enable(enabled)
        _set_control_enabled(self.pad_sink_mode_choice, enabled)

        self.chk_pad_sink_ambient.Enable(enabled and not is_heatsink_mode)
        self.pad_sink_temp_input.Enable(enabled and (not is_heatsink_mode) and (not use_ambient))

        if preset_cp is not None:
            self.heatsink_cp_input.SetValue(float(preset_cp))

        self.heatsink_rth_input.Enable(enabled and is_heatsink_mode)
        _set_control_enabled(self.heatsink_material_choice, enabled and is_heatsink_mode)
        self.heatsink_mass_input.Enable(enabled and is_heatsink_mode)
        self.heatsink_cp_input.Enable(enabled and is_heatsink_mode and mat_key == "custom")

        if not is_heatsink_mode:
            summary = "Fixed sink temperature (infinite thermal mass)"
        else:
            mass_g = max(0.0, float(self.heatsink_mass_input.GetValue()))
            cp = float(self.heatsink_cp_input.GetValue())
            rth = float(self.heatsink_rth_input.GetValue())
            if mass_g <= 0.0:
                summary = "C_hs = 0.0 J/K (instantaneous heatsink, tau = 0 s)"
            else:
                c_hs = (mass_g / 1000.0) * cp
                tau = rth * c_hs
                summary = f"C_hs = {c_hs:.1f} J/K  \u00b7  Time constant tau = {tau:.1f} s"
        if hasattr(self.lbl_heatsink_summary, "SetLabel"):
            self.lbl_heatsink_summary.SetLabel(summary)

    def _on_heatsink_toggle(self, event):
        """Enable thermal-interface fields only when their model is active."""
        self._apply_thermal_pad_controls()
        if self.chk_heatsink.GetValue():
            self.thermal_pad_pane.Expand()
        self._refresh_preflight()

    def _on_pad_sink_mode_changed(self, event):
        """Handle sink mode selection change."""
        self._apply_thermal_pad_controls()

    def _on_pad_sink_ambient_toggle(self, event):
        """Handle fixed sink ambient checkbox toggle."""
        self._apply_thermal_pad_controls()

    def _on_heatsink_material_changed(self, event):
        """Handle heatsink material preset change."""
        self._apply_thermal_pad_controls()

    def _on_heatsink_param_changed(self, event):
        """Update live heatsink capacity and time-constant readout."""
        self._apply_thermal_pad_controls()
        try:
            event.Skip()
        except Exception:
            pass

    def _on_advanced_pane_changed(self, event):
        """Relayout the Advanced tab when a native pane is expanded."""
        try:
            self.tab_adv.Layout()
        except Exception:
            pass

    def _on_grid_detail_changed(self, event):
        """Enable the explicit node budget only for Custom detail mode."""
        custom = self.grid_detail_choice.GetSelection() == len(GRID_DETAIL_VALUES) - 1
        self.grid_node_budget_input.Enable(custom)
        self._refresh_preflight()

    def _on_estimate_input_changed(self, event):
        """Refresh area and grid estimates after a numeric input changes."""
        self._schedule_preflight_refresh()
        try:
            event.Skip()
        except Exception:
            pass

    def _on_current_enabled_toggle(self, event):
        """Refresh summaries when current simulation is toggled."""
        self._render_current_flows()

    def _apply_flow_mode_visibility(self):
        """Enable/show controls based on selected flow sharing modes."""
        flow = self._selected_current_flow()
        has_flow = flow is not None
        from_sharing = flow.get("from_sharing", "equipotential") if has_flow else "equipotential"
        to_sharing = flow.get("to_sharing", "fixed") if has_flow else "fixed"
        total_editable = has_flow and (from_sharing != "fixed") and (to_sharing != "fixed")
        self.flow_total_input.Enable(total_editable)
        self.flow_name_input.Enable(has_flow)
        _set_control_enabled(self.flow_from_sharing, has_flow)
        _set_control_enabled(self.flow_to_sharing, has_flow)
        for side, sharing in (("from", from_sharing), ("to", to_sharing)):
            widgets = self._flow_side_widgets[side]
            widgets["fixed_row"].ShowItems(has_flow and sharing == "fixed")
        try:
            self.tab_current.Layout()
        except Exception:
            pass

    def _make_empty_flow(self, idx=None):
        """Return a default empty flow dictionary."""
        if idx is None:
            idx = len(self.current_flows)
        return {
            "name": f"Flow {idx + 1}",
            "net_name": "",
            "net_code": 0,
            "total_current_a": 0.0,
            "from_sharing": "equipotential",
            "to_sharing": "fixed",
            "from_pads": [],
            "to_pads": [],
        }

    def _on_flow_new(self, event):
        """Create a new pad-to-pad current flow."""
        self._sync_current_flow_from_fields()
        self.current_flows.append(self._make_empty_flow())
        self.current_flow_index = len(self.current_flows) - 1
        self._current_flows_edited = True
        self.chk_current_enabled.SetValue(True)
        self._render_current_flows()

    def _on_flow_duplicate(self, event):
        """Duplicate the selected current flow."""
        self._sync_current_flow_from_fields()
        flow = self._selected_current_flow()
        if flow is None:
            return
        duplicate = copy.deepcopy(flow)
        duplicate["name"] = f"{flow.get('name', 'Flow')} Copy"
        self.current_flows.append(duplicate)
        self.current_flow_index = len(self.current_flows) - 1
        self._current_flows_edited = True
        self._render_current_flows()

    def _on_flow_delete(self, event):
        """Delete the selected current flow."""
        self._sync_current_flow_from_fields()
        if 0 <= self.current_flow_index < len(self.current_flows):
            del self.current_flows[self.current_flow_index]
        self.current_flow_index = min(self.current_flow_index, len(self.current_flows) - 1)
        self._current_flows_edited = True
        if not self.current_flows:
            self.current_groups = []
        self._render_current_flows()

    def _on_current_flow_selected(self, event):
        """Handle current-flow list selection changes."""
        self._sync_current_flow_from_fields()
        try:
            self.current_flow_index = event.GetIndex()
        except Exception:
            try:
                self.current_flow_index = self.current_flow_list.GetFirstSelected()
            except Exception:
                self.current_flow_index = -1
        self._render_current_flow_editor()

    def _on_flow_apply_fields(self, event):
        """Apply edited flow name and sharing/total settings."""
        self._sync_current_flow_from_fields()
        self._current_flows_edited = True
        self._render_current_flows()

    def _on_flow_total_changed(self, event):
        """Update flow total current when user edits the Total (A) spinner."""
        self._sync_current_flow_from_fields()
        self._current_flows_edited = True
        self._render_current_flows()
        try:
            event.Skip()
        except Exception:
            pass

    def _on_flow_sharing_changed(self, side):
        """Handle sharing mode change on From or To side."""
        self._sync_current_flow_from_fields()
        self._current_flows_edited = True
        self._render_current_flows()

    def _append_pads_to_flow_side(self, flow, side, pads):
        """Append normalized unique pad descriptors to ``from_pads`` or ``to_pads``."""
        key = f"{side}_pads"
        target = flow.setdefault(key, [])
        existing = {p.get("pad_key") or p.get("name") for p in target if (p.get("pad_key") or p.get("name"))}
        for pad in (pads or []):
            desc = _normalize_flow_pad(pad)
            desc["current_a"] = 0.0
            ident = desc.get("pad_key") or desc.get("name")
            if ident and ident in existing:
                continue
            target.append(desc)
            if ident:
                existing.add(ident)

    def _on_flow_add_selection(self, side):
        """Add live KiCad selected pads to the From or To side of the selected flow."""
        self._sync_current_flow_from_fields()
        pads = self._selection_descriptors()
        if not pads:
            wx.MessageBox("Select one or more pads in KiCad first.", "ThermalSim")
            return
        flow = self._selected_current_flow()
        if flow is None:
            flow = self._make_empty_flow()
            self.current_flows.append(flow)
            self.current_flow_index = len(self.current_flows) - 1
            self.chk_current_enabled.SetValue(True)
        self._append_pads_to_flow_side(flow, side, pads)
        self._current_flows_edited = True
        self._render_current_flows()

    def _on_flow_pick_pads(self, side):
        """Open ``PadPickerDialog`` to add pads from any net on the board."""
        self._sync_current_flow_from_fields()
        if not self.board_pads:
            wx.MessageBox(
                "No board pad list available. Select pads in KiCad and click 'Add Selected'.",
                "ThermalSim",
            )
            return
        flow = self._selected_current_flow()
        initial_net = ""
        if flow is not None:
            nets = _flow_pad_nets(flow)
            initial_net = flow.get("net_name") or (nets[0] if nets else "")
        picker = PadPickerDialog(self, self.board_pads, initial_net=initial_net)
        try:
            if picker.ShowModal() != wx.ID_OK:
                return
            picked = picker.get_selected_pads()
        finally:
            picker.Destroy()
        if not picked:
            return
        if flow is None:
            flow = self._make_empty_flow()
            self.current_flows.append(flow)
            self.current_flow_index = len(self.current_flows) - 1
            self.chk_current_enabled.SetValue(True)
        self._append_pads_to_flow_side(flow, side, picked)
        self._current_flows_edited = True
        self._render_current_flows()

    def _on_flow_remove_pads(self, side):
        """Remove selected pad rows from the From or To side of the flow."""
        self._sync_current_flow_from_fields()
        flow = self._selected_current_flow()
        if flow is None:
            return
        indices = set(self._selected_flow_pad_indices(side))
        if not indices:
            return
        key = f"{side}_pads"
        flow[key] = [
            pad for idx, pad in enumerate(flow.get(key, []))
            if idx not in indices
        ]
        self._current_flows_edited = True
        self._render_current_flows()

    def _on_flow_apply_pad_current(self, side):
        """Apply the ``Set A`` value to selected rows (or all rows) on one side."""
        flow = self._selected_current_flow()
        if flow is None:
            return
        widgets = self._flow_side_widgets[side]
        flow[f"{side}_sharing"] = "fixed"
        widgets["sharing"].SetSelection(SHARING_MODE_KEYS.index("fixed"))
        current = abs(float(widgets["pad_value"].GetValue()))
        pads = flow.get(f"{side}_pads", [])
        indices = self._selected_flow_pad_indices(side)
        if not indices:
            indices = list(range(len(pads)))
        for idx in indices:
            if 0 <= idx < len(pads):
                pads[idx]["current_a"] = current
        self._current_flows_edited = True
        self._render_current_flows()

    def _on_flow_apply_pad_current_list(self, side):
        """Apply comma-separated non-negative currents to pads on one side."""
        flow = self._selected_current_flow()
        if flow is None:
            return
        widgets = self._flow_side_widgets[side]
        text = widgets["pad_list_input"].GetValue().strip()
        if not text:
            return
        try:
            values = [
                float(part.strip())
                for part in text.replace(";", ",").split(",")
                if part.strip()
            ]
        except ValueError:
            wx.MessageBox(
                "Enter comma-separated numeric currents, e.g. 3, 3, 3.",
                "ThermalSim",
            )
            return
        if any(v < 0.0 for v in values):
            wx.MessageBox(
                "Enter non-negative current magnitudes (flow direction is From -> To).",
                "ThermalSim",
            )
            return
        pads = flow.get(f"{side}_pads", [])
        if len(values) == 1 and pads:
            for pad in pads:
                pad["current_a"] = values[0]
        elif len(values) == len(pads):
            for idx, value in enumerate(values):
                pads[idx]["current_a"] = value
        else:
            wx.MessageBox(
                f"Current list has {len(values)} values, but this side has {len(pads)} pads.",
                "ThermalSim",
            )
            return
        flow[f"{side}_sharing"] = "fixed"
        widgets["sharing"].SetSelection(SHARING_MODE_KEYS.index("fixed"))
        self._current_flows_edited = True
        self._render_current_flows()

    def _on_browse_pwl(self, event):
        """Handle Browse PWL button click to select a PWL file."""
        start_dir = os.path.dirname(__file__)
        dlg = wx.FileDialog(
            self,
            "Select PWL Power Profile",
            defaultDir=start_dir,
            wildcard="PWL files (*.pwl;*.txt;*.csv)|*.pwl;*.txt;*.csv|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            current = self.power_input.GetValue().strip()
            if current:
                self.power_input.SetValue(current + ", " + path)
            else:
                self.power_input.SetValue(path)
        dlg.Destroy()

    def _on_browse_output(self, event):
        """Handle Browse button click for output directory."""
        start_dir = self.output_dir_input.GetValue()
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = os.path.dirname(__file__)
        dlg = wx.DirDialog(
            self,
            "Select Output Folder",
            defaultPath=start_dir,
            style=wx.DD_DEFAULT_STYLE
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.output_dir_input.SetValue(dlg.GetPath())
        dlg.Destroy()

    def _on_help(self, event):
        """Handle Help button click (fallback when wx.adv unavailable)."""
        import webbrowser
        webbrowser.open("https://github.com/PoeAlex/KiCad_Thermal_Sim#readme")

    def _on_more(self, event):
        """Open the compact secondary-actions menu."""
        menu = wx.Menu()
        load_item = menu.Append(wx.ID_ANY, "Load Settings...")
        save_item = menu.Append(wx.ID_ANY, "Save Settings...")
        menu.AppendSeparator()
        help_item = menu.Append(wx.ID_ANY, "Help")
        self.Bind(wx.EVT_MENU, self._on_load_settings, load_item)
        self.Bind(wx.EVT_MENU, self._on_save_settings, save_item)
        self.Bind(wx.EVT_MENU, self._on_help, help_item)
        try:
            self.PopupMenu(menu)
        finally:
            menu.Destroy()

    def _refresh_context_summary(self):
        """Refresh persistent heat-source and current-flow context."""
        power_text = summarize_power_pads(self.power_pads)
        if not self.chk_current_enabled.GetValue():
            current_text = "Current heating off"
        elif self.current_flows:
            errors, _ = validate_current_flows(self.current_flows)
            flow_summary = summarize_current_flows(self.current_flows)
            if errors:
                current_text = f"Current heating on / {flow_summary} / needs attention"
            else:
                current_text = f"Current heating on / {flow_summary}"
        elif self.current_groups:
            _, balance_rows = summarize_current_groups(self.current_groups)
            if any(status != "OK" for _, _, status in balance_rows):
                current_text = "Current heating on / needs balance"
            else:
                current_text = "Current heating on / balanced"
        else:
            current_text = "Current heating on / no flows"
        self.lbl_context.SetLabel(f"{power_text} / {current_text}")
        if hasattr(self, "lbl_heat_summary"):
            self.lbl_heat_summary.SetLabel(power_text)

    # ------------------------------------------------------------------
    # Power-pad tab helpers
    # ------------------------------------------------------------------

    def _selected_power_pad_indices(self):
        """Return selected row indices in the manual power-pad table."""
        indices = []
        try:
            idx = self.power_pad_list.GetFirstSelected()
            while idx != -1:
                indices.append(idx)
                idx = self.power_pad_list.GetNextItem(idx, getattr(wx, "LIST_NEXT_ALL", 0), getattr(wx, "LIST_STATE_SELECTED", 0))
        except Exception:
            pass
        return indices

    def _on_power_add_selection(self, event):
        """Add the live KiCad pad selection to the manual power-pad list."""
        pads = self._selection_descriptors()
        if not pads:
            wx.MessageBox("Select one or more pads in KiCad first.", "ThermalSim")
            return
        power_value = self.power_input.GetValue().strip() or "0.0"
        existing = {pad.get('pad_key') for pad in self.power_pads}
        for pad in pads:
            descriptor = normalize_power_pad_descriptor(pad, default_power=power_value)
            if descriptor['pad_key'] not in existing:
                self.power_pads.append(descriptor)
                existing.add(descriptor['pad_key'])
        self._power_pads_edited = True
        self._render_power_pads()

    def _on_power_remove_pads(self, event):
        """Remove selected rows from the manual power-pad list."""
        indices = set(self._selected_power_pad_indices())
        if not indices:
            return
        self.power_pads = [
            pad for idx, pad in enumerate(self.power_pads)
            if idx not in indices
        ]
        self._power_pads_edited = True
        self._render_power_pads()

    def _on_power_clear_pads(self, event):
        """Clear all manual power pads."""
        self.power_pads = []
        self._power_pads_edited = True
        self._render_power_pads()

    def _on_power_apply_value(self, event):
        """Apply the Power W/PWL field to selected rows or all power pads."""
        value = self.power_input.GetValue().strip()
        if not value:
            return
        indices = self._selected_power_pad_indices()
        list_values = [part.strip() for part in value.split(",") if part.strip()]
        if not indices and len(list_values) > 1:
            self._on_power_apply_list(event)
            return
        if not indices:
            indices = list(range(len(self.power_pads)))
        for idx in indices:
            if 0 <= idx < len(self.power_pads):
                self.power_pads[idx]['power'] = value
        self._power_pads_edited = True
        self._render_power_pads()

    def _on_power_apply_list(self, event):
        """Apply comma-separated power values to all power pads in table order."""
        text = self.power_input.GetValue().strip()
        if not text:
            return
        values = [part.strip() for part in text.split(",") if part.strip()]
        if not values:
            return
        if not self.power_pads:
            return
        if len(values) == 1:
            for pad in self.power_pads:
                pad['power'] = values[0]
        elif len(values) == len(self.power_pads):
            for idx, value in enumerate(values):
                self.power_pads[idx]['power'] = value
        else:
            wx.MessageBox(
                f"Power list has {len(values)} values, but the table has {len(self.power_pads)} pads.",
                "ThermalSim"
            )
            return
        self._power_pads_edited = True
        self._render_power_pads()

    def _on_power_edit_row(self, event):
        """Edit a heat-source value by activating its table row."""
        try:
            index = int(event.GetIndex())
        except Exception:
            return
        if not (0 <= index < len(self.power_pads)):
            return
        current = str(self.power_pads[index].get('power', ''))
        dialog = wx.TextEntryDialog(
            self,
            "Enter constant power in W or a PWL file path:",
            "Edit Heat Source",
            current,
        )
        try:
            if dialog.ShowModal() != wx.ID_OK:
                return
            value = dialog.GetValue().strip()
        finally:
            dialog.Destroy()
        if value:
            self.power_pads[index]['power'] = value
            self._power_pads_edited = True
            self._render_power_pads()

    def _render_power_pads(self):
        """Refresh the manual power-pad table."""
        self.power_pads = prepare_power_pads(self.power_pads, self.power_input.GetValue())
        try:
            self.power_pad_list.DeleteAllItems()
            for row_idx, pad in enumerate(self.power_pads):
                values = [
                    pad.get('name', ''),
                    pad.get('net_name') or "(no net)",
                    pad.get('layer', ''),
                    pad.get('power', ''),
                ]
                self.power_pad_list.InsertItem(row_idx, values[0])
                for col_idx, value in enumerate(values[1:], start=1):
                    self.power_pad_list.SetItem(row_idx, col_idx, value)
        except Exception:
            pass
        if hasattr(self, "lbl_context"):
            self._refresh_context_summary()
        if hasattr(self, "lbl_preflight"):
            self._refresh_preflight()

    # ------------------------------------------------------------------
    # Current-path tab helpers
    # ------------------------------------------------------------------

    def _selection_descriptors(self):
        """Return live pad descriptors from the host plugin."""
        if not self.selection_provider:
            return []
        try:
            pads = self.selection_provider()
        except Exception:
            pads = []
        return [normalize_pad_descriptor(pad) for pad in (pads or [])]

    def _selected_current_flow(self):
        """Return the selected current flow or None."""
        if 0 <= self.current_flow_index < len(self.current_flows):
            return self.current_flows[self.current_flow_index]
        return None

    def _selected_flow_pad_indices(self, side):
        """Return selected row indices in ``flow_from_list`` or ``flow_to_list``."""
        list_ctrl = self._flow_side_widgets[side]["list_ctrl"]
        indices = []
        try:
            idx = list_ctrl.GetFirstSelected()
            while idx != -1:
                indices.append(idx)
                idx = list_ctrl.GetNextItem(
                    idx, getattr(wx, "LIST_NEXT_ALL", 0), getattr(wx, "LIST_STATE_SELECTED", 0)
                )
        except Exception:
            pass
        return indices

    def _sync_current_flow_from_fields(self):
        """Copy editor fields into the selected current flow."""
        flow = self._selected_current_flow()
        if flow is None:
            return
        flow["name"] = self.flow_name_input.GetValue().strip() or flow.get("name", "Flow")
        from_sel = self.flow_from_sharing.GetSelection()
        to_sel = self.flow_to_sharing.GetSelection()
        from_sharing = (
            SHARING_MODE_KEYS[from_sel]
            if 0 <= from_sel < len(SHARING_MODE_KEYS)
            else "equipotential"
        )
        to_sharing = (
            SHARING_MODE_KEYS[to_sel]
            if 0 <= to_sel < len(SHARING_MODE_KEYS)
            else "fixed"
        )
        flow["from_sharing"] = from_sharing
        flow["to_sharing"] = to_sharing
        if from_sharing != "fixed" and to_sharing != "fixed":
            flow["total_current_a"] = abs(float(self.flow_total_input.GetValue()))

    def _render_current_flows(self):
        """Refresh the Current Flows table, balance status, and flow editor."""
        self.current_flows = prepare_current_flows(self.current_flows)
        if self.current_flows and not (0 <= self.current_flow_index < len(self.current_flows)):
            self.current_flow_index = 0
        try:
            self.current_flow_list.DeleteAllItems()
            for row_idx, flow in enumerate(self.current_flows):
                nets = _flow_pad_nets(flow)
                if len(nets) > 1:
                    net_label = "Mixed"
                elif nets:
                    net_label = nets[0]
                else:
                    net_label = flow.get("net_name") or "-"
                from_count = len(flow.get("from_pads", []))
                to_count = len(flow.get("to_pads", []))
                total_a = float(flow.get("total_current_a", 0.0))
                self.current_flow_list.InsertItem(row_idx, flow.get("name", ""))
                self.current_flow_list.SetItem(row_idx, 1, net_label)
                self.current_flow_list.SetItem(row_idx, 2, f"{from_count} -> {to_count}")
                self.current_flow_list.SetItem(row_idx, 3, f"{total_a:.6g} A")
            if 0 <= self.current_flow_index < len(self.current_flows):
                self.current_flow_list.Select(self.current_flow_index)
        except Exception:
            pass

        lines = _format_flow_balance_lines(self.current_flows)
        _, warnings = validate_current_flows(self.current_flows)
        for warning in warnings:
            lines.append(f"Warning: {warning}")
        self.current_balance_text.SetValue("\n".join(lines))
        self._render_current_flow_editor()
        if hasattr(self, "lbl_context"):
            self._refresh_context_summary()
        if hasattr(self, "lbl_preflight"):
            self._refresh_preflight()

    def _render_current_flow_editor(self):
        """Refresh right-hand flow editor fields and From/To pad tables."""
        flow = self._selected_current_flow()
        if flow is None:
            self.flow_name_input.SetValue("")
            self.flow_total_input.SetValue(0.0)
            self.flow_from_sharing.SetSelection(0)
            self.flow_to_sharing.SetSelection(2)
            for side in ("from", "to"):
                widgets = self._flow_side_widgets[side]
                widgets["pad_list_input"].SetValue("")
                try:
                    widgets["list_ctrl"].DeleteAllItems()
                except Exception:
                    pass
            self._apply_flow_mode_visibility()
            return

        self.flow_name_input.SetValue(str(flow.get("name", "")))
        self.flow_total_input.SetValue(float(flow.get("total_current_a", 0.0)))
        from_sharing = normalize_sharing_mode(flow.get("from_sharing", "equipotential"))
        to_sharing = normalize_sharing_mode(flow.get("to_sharing", "fixed"))
        self.flow_from_sharing.SetSelection(SHARING_MODE_KEYS.index(from_sharing))
        self.flow_to_sharing.SetSelection(SHARING_MODE_KEYS.index(to_sharing))

        for side, sharing in (("from", from_sharing), ("to", to_sharing)):
            widgets = self._flow_side_widgets[side]
            pads = flow.get(f"{side}_pads", [])
            if sharing == "fixed":
                widgets["pad_list_input"].SetValue(
                    ", ".join(f"{float(p.get('current_a', 0.0)):.6g}" for p in pads)
                )
            else:
                widgets["pad_list_input"].SetValue("")
            try:
                list_ctrl = widgets["list_ctrl"]
                list_ctrl.DeleteAllItems()
                for row_idx, pad in enumerate(pads):
                    current_label = (
                        "Auto (Cu)"
                        if sharing == "equipotential"
                        else f"{float(pad.get('current_a', 0.0)):.6g}"
                    )
                    list_ctrl.InsertItem(row_idx, pad.get("name", ""))
                    list_ctrl.SetItem(row_idx, 1, pad.get("layer", ""))
                    list_ctrl.SetItem(row_idx, 2, current_label)
            except Exception:
                pass
        self._apply_flow_mode_visibility()

    # ------------------------------------------------------------------
    # Settings I/O
    # ------------------------------------------------------------------

    def get_values(self):
        """
        Extract all settings from the dialog.

        Returns
        -------
        dict or None
            Dictionary of all settings if parsing succeeds, None if
            any value fails to parse.

        Notes
        -----
        The returned dictionary contains:
        - power_str : str
        - power_pads : list
        - time : float
        - steady_state : bool
        - amb : float
        - thick : float
        - res : float
        - show_all : bool
        - snapshots : bool
        - snap_count : int
        - output_dir : str
        - ignore_traces : bool
        - ignore_polygons : bool (always False, disabled feature)
        - limit_area : bool
        - pad_dist_mm : float
        - area_mode : str
        - area_margin_mm : float
        - use_heatsink : bool
        - pad_th : float
        - pad_k : float
        - pad_cap_areal : float
        - pad_sink_mode : str
        - pad_contact_factor : float
        - pad_sink_temp_use_ambient : bool
        - pad_sink_temp_c : float
        - heatsink_rth_kw : float
        - heatsink_material : str
        - heatsink_mass_g : float
        - heatsink_cp_j_kgk : float
        - h_conv : float
        - grid_expert_limits : bool
        - grid_max_cells : int
        - grid_target_cells : int
        - grid_detail_level : str
        - grid_node_budget : int
        - compute_engine : str
        """
        try:
            self._sync_current_flow_from_fields()
            if not self._power_pads_edited:
                legacy_pads = []
                for pad in (self.power_pads or self.initial_power_pads):
                    item = dict(pad)
                    item.pop('power', None)
                    item.pop('power_str', None)
                    item.pop('power_w', None)
                    legacy_pads.append(item)
                self.power_pads = prepare_power_pads(legacy_pads, self.power_input.GetValue())
            power_pads = prepare_power_pads(self.power_pads, self.power_input.GetValue())
            current_flows = prepare_current_flows(self.current_flows)
            current_groups = (
                [] if self._current_flows_edited else prepare_current_groups(self.current_groups)
            )
            power_str = power_pads_to_power_str(power_pads, self.power_input.GetValue())
            detail_idx = self.grid_detail_choice.GetSelection()
            if 0 <= detail_idx < len(GRID_DETAIL_VALUES):
                grid_detail_level = GRID_DETAIL_VALUES[detail_idx]
            else:
                grid_detail_level = "balanced"
            grid_expert_limits = grid_detail_level == "custom"
            if grid_expert_limits:
                grid_node_budget = int(self.grid_node_budget_input.GetValue())
                if (
                    grid_node_budget < 50000
                    or grid_node_budget > MAX_CUSTOM_GRID_NODES
                ):
                    raise ValueError
                target_nodes = max(25_000, int(grid_node_budget * 0.85))
            else:
                grid_node_budget, target_nodes = GRID_DETAIL_PRESETS[grid_detail_level]
            layer_count = max(1, len(self.layer_names))
            grid_max_cells = max(1000, grid_node_budget // layer_count)
            grid_target_cells = max(1000, min(grid_max_cells, target_nodes // layer_count))
            stepping_modes = ["auto", "multi_phase", "two_phase", "uniform"]
            stepping_idx = self.time_stepping_choice.GetSelection()
            time_stepping = stepping_modes[stepping_idx] if 0 <= stepping_idx < len(stepping_modes) else "auto"
            engine_modes = ["auto", "fast_native", "legacy"]
            engine_idx = self.compute_engine_choice.GetSelection()
            compute_engine = (
                engine_modes[engine_idx]
                if 0 <= engine_idx < len(engine_modes)
                else "auto"
            )
            mat_key, preset_cp = self._selected_heatsink_material()
            pad_sink_mode = (
                "heatsink"
                if self.pad_sink_mode_choice.GetSelection() == 1
                else "fixed"
            )
            heatsink_cp = (
                float(preset_cp)
                if preset_cp is not None
                else float(self.heatsink_cp_input.GetValue())
            )
            return {
                'power_str': power_str,
                'power_pads': power_pads,
                'time': float(self.time_input.GetValue()),
                'steady_state': self.chk_steady_state.GetValue(),
                'amb': float(self.amb_input.GetValue()),
                'thick': float(self.thick_input.GetValue()),
                'res': float(self.res_input.GetValue()),
                'show_all': self.chk_all_layers.GetValue(),
                'snapshots': self.chk_snapshots.GetValue(),
                'snap_count': int(self.snap_count_input.GetValue()),
                'output_dir': self.output_dir_input.GetValue().strip(),
                'ignore_traces': self.chk_ignore_traces.GetValue(),
                'ignore_polygons': False,  # Disabled by request
                'limit_area': self.chk_limit_area.GetValue(),
                'pad_dist_mm': float(self.pad_dist_input.GetValue()),
                'area_mode': 'active' if self.chk_limit_area.GetValue() else 'full',
                'area_margin_mm': float(self.pad_dist_input.GetValue()),
                'use_heatsink': self.chk_heatsink.GetValue(),
                'pad_th': float(self.pad_thick.GetValue()),
                'pad_k': float(self.pad_k.GetValue()),
                'pad_cap_areal': float(self.pad_cap.GetValue()),
                'pad_sink_mode': pad_sink_mode,
                'pad_contact_factor': float(self.pad_contact_factor.GetValue()),
                'pad_sink_temp_use_ambient': bool(self.chk_pad_sink_ambient.GetValue()),
                'pad_sink_temp_c': float(self.pad_sink_temp_input.GetValue()),
                'heatsink_rth_kw': float(self.heatsink_rth_input.GetValue()),
                'heatsink_material': mat_key,
                'heatsink_mass_g': float(self.heatsink_mass_input.GetValue()),
                'heatsink_cp_j_kgk': heatsink_cp,
                'h_conv': float(self.h_conv_input.GetValue()),
                'h_conv_bottom': float(self.h_conv_bottom_input.GetValue()),
                'via_plating_um': float(self.via_plating_input.GetValue()),
                'via_fill': 'copper' if self.via_fill_choice.GetSelection() == 1 else 'plated',
                'grid_expert_limits': grid_expert_limits,
                'grid_max_cells': grid_max_cells,
                'grid_target_cells': grid_target_cells,
                'grid_detail_level': grid_detail_level,
                'grid_node_budget': grid_node_budget,
                'compute_engine': compute_engine,
                'mesh_mode': 'uniform',
                'time_stepping': time_stepping,
                'current_enabled': self.chk_current_enabled.GetValue(),
                'current_flows': current_flows,
                'current_groups': current_groups,
            }
        except ValueError:
            return None

    def _apply_defaults(self, defaults):
        """
        Apply default values to dialog controls.

        Parameters
        ----------
        defaults : dict
            Dictionary of default values to apply.
        """
        try:
            self.power_input.SetValue(
                str(defaults.get('power_str', self.power_input.GetValue()))
            )

            if 'time' in defaults:
                self.time_input.SetValue(float(defaults['time']))
            if 'amb' in defaults:
                self.amb_input.SetValue(float(defaults['amb']))
            if 'thick' in defaults and self.stackup_thickness_mm is None:
                self.thick_input.SetValue(float(defaults['thick']))
            elif self.stackup_thickness_mm is not None:
                self.thick_input.SetValue(self.stackup_thickness_mm)
                self.thick_input.Enable(False)
            if 'res' in defaults:
                self.res_input.SetValue(float(defaults['res']))

            self.chk_all_layers.SetValue(
                bool(defaults.get('show_all', self.chk_all_layers.GetValue()))
            )
            self.chk_snapshots.SetValue(
                bool(defaults.get('snapshots', self.chk_snapshots.GetValue()))
            )
            if 'snap_count' in defaults:
                self.snap_count_input.SetValue(int(defaults['snap_count']))
            self.chk_steady_state.SetValue(
                bool(defaults.get('steady_state', False))
            )
            self._apply_steady_state_controls()

            out_dir = defaults.get('output_dir')
            if out_dir:
                self.output_dir_input.SetValue(str(out_dir))

            self.chk_ignore_traces.SetValue(
                bool(defaults.get('ignore_traces', self.chk_ignore_traces.GetValue()))
            )
            area_mode = str(defaults.get('area_mode', '') or '').lower()
            area_limited = (
                area_mode == 'active'
                if area_mode
                else bool(defaults.get('limit_area', self.chk_limit_area.GetValue()))
            )
            self.chk_limit_area.SetValue(area_limited)
            margin = defaults.get('area_margin_mm', defaults.get('pad_dist_mm'))
            if margin is not None:
                self.pad_dist_input.SetValue(float(margin))
            self.pad_dist_input.Enable(self.chk_limit_area.GetValue())

            self.chk_heatsink.SetValue(
                bool(defaults.get('use_heatsink', self.chk_heatsink.GetValue()))
            )
            if 'pad_th' in defaults:
                self.pad_thick.SetValue(float(defaults['pad_th']))
            if 'pad_k' in defaults:
                self.pad_k.SetValue(float(defaults['pad_k']))
            if 'pad_cap_areal' in defaults:
                self.pad_cap.SetValue(float(defaults['pad_cap_areal']))

            contact_factor = defaults.get('pad_contact_factor', 0.2)
            self.pad_contact_factor.SetValue(
                0.2 if contact_factor is None else float(contact_factor)
            )
            sink_mode = str(defaults.get('pad_sink_mode', 'fixed') or 'fixed').strip().lower()
            self.pad_sink_mode_choice.SetSelection(1 if sink_mode == 'heatsink' else 0)
            self.chk_pad_sink_ambient.SetValue(
                bool(defaults.get('pad_sink_temp_use_ambient', True))
            )
            sink_temp = defaults.get('pad_sink_temp_c', 25.0)
            self.pad_sink_temp_input.SetValue(
                25.0 if sink_temp is None else float(sink_temp)
            )
            rth_kw = defaults.get('heatsink_rth_kw', 5.0)
            self.heatsink_rth_input.SetValue(
                5.0 if rth_kw is None else float(rth_kw)
            )
            mat_key = str(defaults.get('heatsink_material', 'aluminium') or 'aluminium').strip().lower()
            mat_idx = next(
                (i for i, (_, key, _) in enumerate(HEATSINK_MATERIAL_OPTIONS) if key == mat_key),
                0,
            )
            self.heatsink_material_choice.SetSelection(mat_idx)
            mass_g = defaults.get('heatsink_mass_g', 50.0)
            self.heatsink_mass_input.SetValue(
                50.0 if mass_g is None else float(mass_g)
            )
            custom_cp = defaults.get('heatsink_cp_j_kgk', 897.0)
            self.heatsink_cp_input.SetValue(
                897.0 if custom_cp is None else float(custom_cp)
            )
            self._apply_thermal_pad_controls()

            if 'h_conv' in defaults and defaults['h_conv'] is not None:
                self.h_conv_input.SetValue(float(defaults['h_conv']))
            if 'h_conv_bottom' in defaults and defaults['h_conv_bottom'] is not None:
                self.h_conv_bottom_input.SetValue(float(defaults['h_conv_bottom']))
            else:
                self.h_conv_bottom_input.SetValue(float(self.h_conv_input.GetValue()))

            via_plating_val = defaults.get('via_plating_um', 25.0)
            self.via_plating_input.SetValue(
                25.0 if via_plating_val is None else float(via_plating_val)
            )
            via_fill_norm = (
                str(defaults.get('via_fill', 'plated') or 'plated')
                .strip()
                .lower()
                .replace('-', '_')
            )
            self.via_fill_choice.SetSelection(
                1 if via_fill_norm in {'copper', 'copper_filled'} else 0
            )
            self._apply_via_fill_controls()

            compute_engine = str(defaults.get('compute_engine', 'auto')).lower()
            compute_engine_idx = {
                'auto': 0,
                'fast_native': 1,
                'legacy': 2,
            }.get(compute_engine, 0)
            self.compute_engine_choice.SetSelection(compute_engine_idx)

            stepping = str(defaults.get('time_stepping', 'auto')).lower()
            stepping_idx = {'auto': 0, 'multi_phase': 1, 'two_phase': 2, 'uniform': 3}.get(stepping, 0)
            self.time_stepping_choice.SetSelection(stepping_idx)

            detail = str(defaults.get('grid_detail_level', '') or '').lower()
            detail = detail.replace(' ', '_').replace('-', '_')
            if detail not in GRID_DETAIL_VALUES:
                detail = 'custom' if defaults.get('grid_expert_limits') else 'balanced'
            self.grid_detail_choice.SetSelection(GRID_DETAIL_VALUES.index(detail))
            if detail == 'custom':
                node_budget = defaults.get('grid_node_budget')
                if node_budget is None:
                    legacy_cells = _safe_int(
                        defaults.get('grid_max_cells'), DEFAULT_GRID_MAX_CELLS
                    )
                    node_budget = legacy_cells * max(1, len(self.layer_names))
                node_budget = min(MAX_CUSTOM_GRID_NODES, max(50000, _safe_int(
                    node_budget, DEFAULT_GRID_NODE_BUDGET
                )))
                self.grid_node_budget_input.SetValue(node_budget)
            else:
                self.grid_node_budget_input.SetValue(GRID_DETAIL_PRESETS[detail][0])
            self.grid_node_budget_input.Enable(detail == 'custom')

            if 'power_pads' in defaults:
                self.power_pads = prepare_power_pads(defaults.get('power_pads', []), self.power_input.GetValue())
                self._power_pads_edited = True
            else:
                self.power_pads = prepare_power_pads(self.initial_power_pads, self.power_input.GetValue())
                self._power_pads_edited = False
            self._render_power_pads()

            self.chk_current_enabled.SetValue(
                bool(defaults.get('current_enabled', self.chk_current_enabled.GetValue()))
            )
            if 'current_flows' in defaults:
                self.current_flows = prepare_current_flows(defaults.get('current_flows', []))
                self.current_groups = prepare_current_groups(defaults.get('current_groups', []))
                self._current_flows_edited = False
            elif defaults.get('current_groups'):
                legacy_groups = prepare_current_groups(defaults.get('current_groups', []))
                self.current_groups = legacy_groups
                self.current_flows = convert_current_groups_to_flows(legacy_groups)
                self._current_flows_edited = False
            else:
                self.current_flows = []
                self.current_groups = []
                self._current_flows_edited = False
            self.current_flow_index = 0 if self.current_flows else -1
            self.current_group_index = 0 if self.current_groups else -1
            self._render_current_flows()
            if self.chk_heatsink.GetValue():
                self.thermal_pad_pane.Expand()
            h_conv_val = float(
                defaults.get('h_conv', 10.0)
                if defaults.get('h_conv') is not None
                else 10.0
            )
            raw_h_bot = defaults.get('h_conv_bottom')
            h_bot_val = h_conv_val if raw_h_bot is None else float(raw_h_bot)
            via_plating_val = float(
                defaults.get('via_plating_um', 25.0)
                if defaults.get('via_plating_um') is not None
                else 25.0
            )
            via_fill_norm = (
                str(defaults.get('via_fill', 'plated') or 'plated')
                .strip()
                .lower()
                .replace('-', '_')
            )
            if (
                str(defaults.get('compute_engine', 'auto') or 'auto').lower() != 'auto'
                or str(defaults.get('time_stepping', 'auto') or 'auto').lower() != 'auto'
                or h_conv_val != 10.0
                or h_bot_val != 10.0
                or via_plating_val != 25.0
                or via_fill_norm in {'copper', 'copper_filled'}
                or (
                    self.stackup_thickness_mm is None
                    and float(defaults.get('thick', 1.6)) != 1.6
                )
            ):
                self.solver_pane.Expand()
        except Exception:
            pass
