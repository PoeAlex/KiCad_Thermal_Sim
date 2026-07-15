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
    from .capabilities import HAS_PARDISO
except ImportError:
    HAS_PARDISO = False


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
    'ambient': "Surrounding air temperature in \u00b0C. Typical lab conditions: 25 \u00b0C.",
    'resolution': "Grid cell size in mm. Smaller = finer detail but quadratically slower.",
    'show_all': "Show every copper layer in the heatmap (vs. only top and bottom).",
    'snapshots': "Save intermediate temperature snapshots during the simulation.",
    'snap_count': "Number of snapshots between t=0 and the final time.",
    'output_dir': "Directory for results. A timestamped subfolder is created automatically.",
    'browse_output': "Select output directory for simulation results.",
    'ignore_traces': "Exclude traces from the thermal model. Faster, slightly more conservative.",
    'limit_area': "Restrict simulation to the region around selected pads.",
    'limit_dist': "Radius in mm around pads when Limit Area is enabled. Typical: 20-40 mm.",
    'enable_pad': "Model a thermal pad/heatsink on the bottom layer (shapes on User.Eco1).",
    'pad_thick': "Thermal pad thickness in mm.",
    'pad_k': "Thermal pad conductivity in W/(m\u00b7K). Silicone ~3, aluminium ~200.",
    'pad_cap': "Areal heat capacity in J/(m\u00b2\u00b7K). Set 0 for negligible thermal mass.",
    'h_conv': "Convection coefficient in W/(m\u00b2\u00b7K). Still air ~5-10, light fan ~25, forced ~50-100.",
    'pcb_thick': "PCB thickness override in mm. Usually auto-detected from stackup.",
    'grid_expert_limits': "Enable expert control over automatic grid coarsening limits.",
    'grid_max_cells': "Estimated cell count above which the grid is automatically coarsened.",
    'grid_target_cells': "Target cell count used when automatic grid coarsening is applied.",
    'capabilities': "Detected solver backends. PyPardiso accelerates large grids significantly.",
    'help': "Open the ThermalSim documentation in your web browser.",
    'preview': "Generate a geometry preview image without running the simulation.",
    'load_settings': "Load simulation settings from a JSON file.",
    'save_settings': "Save the current simulation settings to a JSON file.",
    'current_enable': "Enable DC current-flow simulation for Joule heating in traces and copper pours.",
    'current_groups': "Pad groups used as current sources or sinks. Positive current enters the PCB; negative current leaves it.",
    'current_total': "Total current for the selected group. In distribution mode it is split evenly across all pads.",
    'current_per_pad': "Current value applied to selected pad rows in per-pad mode. If no row is selected, it is applied to all pads in the group.",
    'current_pad_list': "Comma-separated per-pad currents in the same order as the pad table, for example: +6, -4, -2.",
}


CURRENT_GROUP_COLORS = [
    "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
    "#9467bd", "#17becf", "#8c564b", "#e377c2",
]


DEFAULT_GRID_MAX_CELLS = 200000
DEFAULT_GRID_TARGET_CELLS = 100000


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
    ):
        dialog_style = (
            getattr(wx, "DEFAULT_DIALOG_STYLE", 0)
            | getattr(wx, "RESIZE_BORDER", 0)
        )
        super().__init__(parent, title="Thermal Sim", style=dialog_style)

        self.layer_names = layer_names
        self.board_name = str(board_name or "Unsaved board")
        self.board_size_mm = tuple(board_size_mm or ())
        self.preview_callback = preview_callback
        self.selection_provider = selection_provider
        self.run_callback = run_callback
        self.close_callback = close_callback
        self.preflight_callback = preflight_callback
        self.load_settings_callback = load_settings_callback
        self.save_settings_callback = save_settings_callback
        self.current_groups = []
        self.current_group_index = -1
        self.power_pads = []
        self.initial_power_pads = [
            normalize_power_pad_descriptor(pad) for pad in (initial_power_pads or [])
        ]
        self._power_pads_edited = False
        self.last_report_path = None
        self.last_run_dir = None
        self.last_run_status = "idle"

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
        self.btn_open_report.Bind(wx.EVT_BUTTON, self._on_open_report)
        self.btn_open_report.Enable(False)
        result_sizer.Add(self.btn_open_report, 0, wx.RIGHT, 6)
        self.btn_open_folder = wx.Button(self.result_panel, label="Open Folder")
        self.btn_open_folder.Bind(wx.EVT_BUTTON, self._on_open_folder)
        self.btn_open_folder.Enable(False)
        result_sizer.Add(self.btn_open_folder, 0)
        self.result_panel.SetSizer(result_sizer)
        self.result_panel.Show(False)
        main_sizer.Add(self.result_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        main_sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        # Fixed status and action footer.
        footer = wx.BoxSizer(wx.HORIZONTAL)
        status_sizer = wx.BoxSizer(wx.VERTICAL)
        self.lbl_preflight_status = wx.StaticText(self, label="Checking setup...")
        status_font = self.lbl_preflight_status.GetFont()
        status_font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.lbl_preflight_status.SetFont(status_font)
        status_sizer.Add(self.lbl_preflight_status, 0, wx.EXPAND)
        self.lbl_preflight = wx.StaticText(self, label="Checking simulation setup...")
        status_sizer.Add(self.lbl_preflight, 0, wx.EXPAND | wx.TOP, 2)
        footer.Add(status_sizer, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 12)

        self.btn_more = wx.Button(self, label="More")
        self.btn_more.Bind(wx.EVT_BUTTON, self._on_more)
        footer.Add(self.btn_more, 0, wx.ALL, 4)

        self.btn_preview = wx.Button(self, label="Preview")
        self.btn_preview.Bind(wx.EVT_BUTTON, self._on_preview)
        self.btn_preview.SetToolTip(TOOLTIP_TEXTS['preview'])
        footer.Add(self.btn_preview, 0, wx.ALL, 4)

        self.btn_run = wx.Button(self, label="Run Simulation")
        self.btn_run.Bind(wx.EVT_BUTTON, self._on_run)
        self.btn_cancel = wx.Button(self, label="Close")
        self.btn_cancel.Bind(wx.EVT_BUTTON, self._on_cancel)
        footer.Add(self.btn_run, 0, wx.ALL, 4)
        footer.Add(self.btn_cancel, 0, wx.ALL, 4)
        try:
            self.btn_run.SetDefault()
        except Exception:
            pass

        try:
            self.Bind(wx.EVT_CLOSE, self._on_cancel)
        except Exception:
            pass

        main_sizer.Add(footer, 0, wx.EXPAND | wx.ALL, 8)

        self.SetSizer(main_sizer)
        self.SetSize((820, 720))
        self.SetMinSize((760, 640))
        self.Center()

        if defaults:
            self._apply_defaults(defaults)
        else:
            self.power_pads = prepare_power_pads(self.initial_power_pads, self.power_input.GetValue())
            self._render_power_pads()
            self._render_current_groups()
        self._refresh_context_summary()
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

        # Ambient Temperature
        self.amb_input = self._add_spin_field(
            box_params, params_parent, "Ambient (\u00b0C)", 25.0,
            min_val=-40.0, max_val=200.0, inc=1.0, digits=1,
            tooltip_key='ambient'
        )

        # Resolution
        self.res_input = self._add_spin_field(
            box_params, params_parent, "Resolution (mm)", suggested_res,
            min_val=0.05, max_val=10.0, inc=0.05, digits=2,
            tooltip_key='resolution'
        )

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
        btn_add.Bind(wx.EVT_BUTTON, self._on_power_add_selection)
        edit_buttons.Add(btn_add, 0, wx.ALL, 2)
        btn_remove = wx.Button(edit_parent, label="Remove")
        btn_remove.Bind(wx.EVT_BUTTON, self._on_power_remove_pads)
        edit_buttons.Add(btn_remove, 0, wx.ALL, 2)
        btn_clear = wx.Button(edit_parent, label="Clear")
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
        self.geometry_pane.Expand()
        filter_panel = self.geometry_pane.GetPane()
        box_filter = wx.BoxSizer(wx.VERTICAL)

        self.chk_ignore_traces = wx.CheckBox(filter_panel, label="Ignore traces")
        self.chk_ignore_traces.SetValue(False)
        self.chk_ignore_traces.SetToolTip(TOOLTIP_TEXTS['ignore_traces'])
        box_filter.Add(self.chk_ignore_traces, 0, wx.ALL, 3)

        self.chk_limit_area = wx.CheckBox(filter_panel, label="Limit area to pads")
        self.chk_limit_area.SetValue(False)
        self.chk_limit_area.SetToolTip(TOOLTIP_TEXTS['limit_area'])
        box_filter.Add(self.chk_limit_area, 0, wx.ALL, 3)

        self.pad_dist_input = self._add_spin_field(
            box_filter, filter_panel, "Limit distance (mm)", 30.0,
            min_val=1.0, max_val=200.0, inc=5.0, digits=1,
            tooltip_key='limit_dist'
        )
        self.pad_dist_input.Enable(False)
        self.chk_limit_area.Bind(wx.EVT_CHECKBOX, self._on_limit_area_toggle)

        filter_panel.SetSizer(box_filter)
        sizer.Add(self.geometry_pane, 0, wx.EXPAND | wx.ALL, 5)

        # --- Thermal Pad ---
        self.thermal_pad_pane = wx.CollapsiblePane(panel, label="Thermal Pad (User.Eco1)")
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
        self._thermal_pad_controls = [self.pad_thick, self.pad_k, self.pad_cap]
        for control in self._thermal_pad_controls:
            control.Enable(False)

        pad_panel.SetSizer(box_pad)
        sizer.Add(self.thermal_pad_pane, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # --- Solver ---
        self.solver_pane = wx.CollapsiblePane(panel, label="Solver")
        solver_panel = self.solver_pane.GetPane()
        box_solver = wx.BoxSizer(wx.VERTICAL)

        self.h_conv_input = self._add_spin_field(
            box_solver, solver_panel, "Convection h (W/m\u00b2K)", 10.0,
            min_val=1.0, max_val=200.0, inc=1.0, digits=1,
            tooltip_key='h_conv'
        )

        self.thick_input = self._add_spin_field(
            box_solver, solver_panel, "PCB thickness (mm)", 1.6,
            min_val=0.1, max_val=10.0, inc=0.1, digits=2,
            tooltip_key='pcb_thick'
        )

        backend_row = wx.BoxSizer(wx.HORIZONTAL)
        backend_row.Add(wx.StaticText(solver_panel, label="Linear solver", size=(165, -1)), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        backend_choices = ["Auto"]
        if HAS_PARDISO:
            backend_choices.append("PyPardiso")
        backend_choices.append("SciPy / SuperLU")
        self.solver_backend_choice = wx.Choice(solver_panel, choices=backend_choices)
        self.solver_backend_choice.SetSelection(0)
        backend_row.Add(self.solver_backend_choice, 1, wx.EXPAND)
        box_solver.Add(backend_row, 0, wx.EXPAND | wx.ALL, 3)

        stepping_row = wx.BoxSizer(wx.HORIZONTAL)
        stepping_row.Add(wx.StaticText(solver_panel, label="Time stepping", size=(165, -1)), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.time_stepping_choice = wx.Choice(
            solver_panel,
            choices=["Auto", "Legacy 3-phase", "2-phase", "Uniform BDF2"],
        )
        self.time_stepping_choice.SetSelection(0)
        stepping_row.Add(self.time_stepping_choice, 1, wx.EXPAND)
        box_solver.Add(stepping_row, 0, wx.EXPAND | wx.ALL, 3)

        solver_panel.SetSizer(box_solver)
        sizer.Add(self.solver_pane, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # --- Grid limits ---
        self.grid_pane = wx.CollapsiblePane(panel, label="Grid Limits")
        grid_panel = self.grid_pane.GetPane()
        box_grid = wx.BoxSizer(wx.VERTICAL)
        self.chk_grid_expert_limits = wx.CheckBox(grid_panel, label="Use expert grid limits")
        self.chk_grid_expert_limits.SetValue(False)
        self.chk_grid_expert_limits.SetToolTip(TOOLTIP_TEXTS['grid_expert_limits'])
        self.chk_grid_expert_limits.Bind(wx.EVT_CHECKBOX, self._on_grid_expert_limits_toggle)
        box_grid.Add(self.chk_grid_expert_limits, 0, wx.ALL, 3)

        self.grid_max_cells_input = self._add_int_spin_field(
            box_grid, grid_panel, "Coarsen above cells", DEFAULT_GRID_MAX_CELLS,
            min_val=1000, max_val=10000000,
            tooltip_key='grid_max_cells'
        )
        self.grid_target_cells_input = self._add_int_spin_field(
            box_grid, grid_panel, "Target cells", DEFAULT_GRID_TARGET_CELLS,
            min_val=1000, max_val=10000000,
            tooltip_key='grid_target_cells'
        )
        self._apply_grid_expert_state(reset_to_defaults=True)

        grid_panel.SetSizer(box_grid)
        sizer.Add(self.grid_pane, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # --- Capabilities (read-only) ---
        solver_str = "Solver: SciPy + PyPardiso" if HAS_PARDISO else "Solver: SciPy"
        lbl_cap = wx.StaticText(panel, label=solver_str)
        lbl_cap.SetToolTip(TOOLTIP_TEXTS['capabilities'])
        sizer.Add(lbl_cap, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        pane_event = getattr(wx, "EVT_COLLAPSIBLEPANE_CHANGED", None)
        if pane_event is not None:
            for pane in (self.geometry_pane, self.thermal_pad_pane, self.solver_pane, self.grid_pane):
                pane.Bind(pane_event, self._on_advanced_pane_changed)

        panel.SetSizer(sizer)

    def _build_current_tab(self, panel):
        """Build the current-path/Joule-heating tab."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.chk_current_enabled = wx.CheckBox(panel, label="Enable current heating")
        self.chk_current_enabled.SetValue(False)
        self.chk_current_enabled.SetToolTip(TOOLTIP_TEXTS['current_enable'])
        self.chk_current_enabled.Bind(wx.EVT_CHECKBOX, self._on_current_enabled_toggle)
        sizer.Add(self.chk_current_enabled, 0, wx.ALL, 5)

        help_text = wx.StaticText(
            panel,
            label="Positive currents enter the PCB; negative currents leave it."
        )
        sizer.Add(help_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        balance_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Net Balance")
        balance_parent = balance_box.GetStaticBox()
        self.current_balance_text = wx.TextCtrl(
            balance_parent, value="", style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
        )
        self.current_balance_text.SetMinSize((-1, 48))
        balance_box.Add(self.current_balance_text, 0, wx.EXPAND | wx.ALL, 3)
        sizer.Add(balance_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        current_content = wx.BoxSizer(wx.HORIZONTAL)

        group_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Current Groups")
        group_parent = group_box.GetStaticBox()
        self.current_group_list = wx.ListCtrl(
            group_parent,
            style=getattr(wx, "LC_REPORT", 0) | getattr(wx, "LC_SINGLE_SEL", 0)
        )
        for idx, (title, width) in enumerate([
            ("Name", 65), ("Net", 75), ("Pads", 35), ("Current", 55),
        ]):
            self.current_group_list.InsertColumn(idx, title, width=width)
        self.current_group_list.SetToolTip(TOOLTIP_TEXTS['current_groups'])
        self.current_group_list.Bind(wx.EVT_LIST_ITEM_SELECTED, self._on_current_group_selected)
        self.current_group_list.SetMinSize((255, -1))
        group_box.Add(self.current_group_list, 1, wx.EXPAND | wx.ALL, 3)

        group_buttons = wx.BoxSizer(wx.HORIZONTAL)
        btn_new = wx.Button(group_parent, label="New")
        btn_new.Bind(wx.EVT_BUTTON, self._on_current_new_group)
        group_buttons.Add(btn_new, 0, wx.ALL, 2)
        btn_add = wx.Button(group_parent, label="Add")
        btn_add.Bind(wx.EVT_BUTTON, self._on_current_add_selection)
        group_buttons.Add(btn_add, 0, wx.ALL, 2)
        btn_remove = wx.Button(group_parent, label="Remove")
        btn_remove.Bind(wx.EVT_BUTTON, self._on_current_remove_pads)
        group_buttons.Add(btn_remove, 0, wx.ALL, 2)
        group_box.Add(group_buttons, 0, wx.EXPAND | wx.ALL, 3)

        group_more_buttons = wx.BoxSizer(wx.HORIZONTAL)
        btn_duplicate = wx.Button(group_parent, label="Duplicate")
        btn_duplicate.Bind(wx.EVT_BUTTON, self._on_current_duplicate_group)
        group_more_buttons.Add(btn_duplicate, 0, wx.ALL, 2)
        btn_delete = wx.Button(group_parent, label="Delete")
        btn_delete.Bind(wx.EVT_BUTTON, self._on_current_delete_group)
        group_more_buttons.Add(btn_delete, 0, wx.ALL, 2)
        group_box.Add(group_more_buttons, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 3)
        current_content.Add(group_box, 0, wx.EXPAND | wx.RIGHT, 5)

        edit_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Selected Group")
        selected_parent = edit_box.GetStaticBox()
        row_name = wx.BoxSizer(wx.HORIZONTAL)
        row_name.Add(wx.StaticText(selected_parent, label="Name:", size=(105, -1)), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.current_name_input = wx.TextCtrl(selected_parent, value="")
        row_name.Add(self.current_name_input, 1, wx.EXPAND | wx.RIGHT, 5)
        btn_apply_name = wx.Button(selected_parent, label="Apply")
        btn_apply_name.Bind(wx.EVT_BUTTON, self._on_current_apply_group_fields)
        row_name.Add(btn_apply_name, 0)
        edit_box.Add(row_name, 0, wx.EXPAND | wx.ALL, 2)

        row_mode = wx.BoxSizer(wx.HORIZONTAL)
        row_mode.Add(wx.StaticText(selected_parent, label="Mode:", size=(105, -1)), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.current_mode_choice = wx.Choice(
            selected_parent,
            choices=["Per-Pad Currents", "Distribute Total Current"]
        )
        self.current_mode_choice.SetSelection(0)
        self.current_mode_choice.Bind(wx.EVT_CHOICE, self._on_current_mode_changed)
        row_mode.Add(self.current_mode_choice, 1, wx.EXPAND)
        edit_box.Add(row_mode, 0, wx.EXPAND | wx.ALL, 2)

        self.current_total_input = self._add_spin_field(
            edit_box, selected_parent, "Group Current (A):", 0.0,
            min_val=-10000.0, max_val=10000.0, inc=0.1, digits=3,
            tooltip_key='current_total'
        )
        self.current_total_row = self.current_total_input._thermal_row_sizer

        row_pad_current = wx.BoxSizer(wx.HORIZONTAL)
        row_pad_current.Add(wx.StaticText(selected_parent, label="Selected Pad A:", size=(105, -1)), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.current_pad_value_input = wx.SpinCtrlDouble(
            selected_parent, value="0.0", min=-10000.0, max=10000.0, inc=0.1
        )
        self.current_pad_value_input.SetDigits(3)
        self.current_pad_value_input.SetToolTip(TOOLTIP_TEXTS['current_per_pad'])
        row_pad_current.Add(self.current_pad_value_input, 1, wx.EXPAND | wx.RIGHT, 5)
        btn_set_pad_current = wx.Button(selected_parent, label="Apply to Selected")
        btn_set_pad_current.Bind(wx.EVT_BUTTON, self._on_current_apply_pad_current)
        row_pad_current.Add(btn_set_pad_current, 0)
        edit_box.Add(row_pad_current, 0, wx.EXPAND | wx.ALL, 2)
        self.current_pad_current_row = row_pad_current

        row_pad_list = wx.BoxSizer(wx.HORIZONTAL)
        row_pad_list.Add(wx.StaticText(selected_parent, label="Pad Currents:", size=(105, -1)), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.current_pad_list_input = wx.TextCtrl(selected_parent, value="")
        self.current_pad_list_input.SetToolTip(TOOLTIP_TEXTS['current_pad_list'])
        row_pad_list.Add(self.current_pad_list_input, 1, wx.EXPAND | wx.RIGHT, 5)
        btn_apply_list = wx.Button(selected_parent, label="Apply List")
        btn_apply_list.Bind(wx.EVT_BUTTON, self._on_current_apply_pad_current_list)
        row_pad_list.Add(btn_apply_list, 0)
        edit_box.Add(row_pad_list, 0, wx.EXPAND | wx.ALL, 2)
        self.current_pad_list_row = row_pad_list

        self.current_pad_list = wx.ListCtrl(
            selected_parent,
            style=getattr(wx, "LC_REPORT", 0) | getattr(wx, "LC_SINGLE_SEL", 0)
        )
        for idx, (title, width) in enumerate([
            ("Pad", 145), ("Net", 110), ("Layer", 60), ("Current A", 75), ("Status", 80),
        ]):
            self.current_pad_list.InsertColumn(idx, title, width=width)
        self.current_pad_list.SetMinSize((-1, 130))
        edit_box.Add(self.current_pad_list, 1, wx.EXPAND | wx.ALL, 3)
        current_content.Add(edit_box, 1, wx.EXPAND)
        sizer.Add(current_content, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        panel.SetSizer(sizer)

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

    def _on_preview(self, event):
        """Handle Preview button click."""
        if self.preview_callback:
            settings = self.get_values()
            if settings and self._refresh_preflight(settings):
                try:
                    output_path = self.preview_callback(settings, self.layer_names)
                    if output_path:
                        self.lbl_preflight_status.SetLabel("Preview ready")
                        self.lbl_preflight.SetLabel(os.path.basename(output_path))
                    else:
                        self.lbl_preflight_status.SetLabel("Preview failed")
                        self.lbl_preflight.SetLabel("No preview image was created.")
                except Exception:
                    self.lbl_preflight_status.SetLabel("Preview failed")
                    self.lbl_preflight.SetLabel("The geometry preview could not be created.")
                    wx.MessageBox("Geometry preview failed.", "ThermalSim")

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

    def _refresh_preflight(self, settings=None):
        """Refresh the permanent readiness summary and Run availability."""
        settings = settings if settings is not None else self.get_values()
        if settings is None:
            self.lbl_preflight_status.SetLabel("Blocked")
            self.lbl_preflight.SetLabel("One or more numeric settings are invalid.")
            self.btn_run.Enable(False)
            return False
        if not self.preflight_callback:
            self.lbl_preflight_status.SetLabel("Ready")
            self.lbl_preflight.SetLabel("Settings are valid.")
            self.btn_run.Enable(True)
            return True
        try:
            result = self.preflight_callback(settings)
            grid = getattr(result, "grid", None)
            details = []
            if grid is not None:
                details.append(
                    f"{grid.actual_res_mm:.3f} mm, {grid.rows} x {grid.cols} x "
                    f"{grid.layer_count} = {grid.nodes:,} nodes, {grid.complexity} complexity"
                )
            messages = list(getattr(result, "errors", []) or getattr(result, "warnings", []))
            if messages:
                details.append(messages[0])
            label = details[0] if details else "Settings checked."
            if len(details) > 1:
                label += "\n" + details[1]
            self.lbl_preflight_status.SetLabel(str(result.status))
            self.lbl_preflight.SetLabel(label)
            self.btn_run.Enable(bool(result.ready))
            return bool(result.ready)
        except Exception:
            self.lbl_preflight_status.SetLabel("Blocked")
            self.lbl_preflight.SetLabel("Preflight could not be completed.")
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
        self.lbl_preflight_status.SetLabel(title)
        self.lbl_preflight.SetLabel(message or default_message)
        self.btn_run.Enable(self.last_run_status != "running")

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
        self.lbl_preflight_status.SetLabel("Completed")
        self.lbl_preflight.SetLabel("Simulation results are ready.")
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

    def _on_snapshots_toggle(self, event):
        """Handle Snapshots checkbox toggle."""
        self.snap_count_input.Enable(self.chk_snapshots.GetValue())
        self._refresh_preflight()

    def _on_heatsink_toggle(self, event):
        """Enable thermal-interface fields only when their model is active."""
        enabled = self.chk_heatsink.GetValue()
        for control in self._thermal_pad_controls:
            control.Enable(enabled)
        if enabled:
            self.thermal_pad_pane.Expand()
        self._refresh_preflight()

    def _on_advanced_pane_changed(self, event):
        """Relayout the Advanced tab when a native pane is expanded."""
        try:
            self.tab_adv.Layout()
        except Exception:
            pass

    def _apply_grid_expert_state(self, reset_to_defaults=False):
        """Enable expert grid limit controls and optionally restore defaults."""
        expert_enabled = self.chk_grid_expert_limits.GetValue()
        if reset_to_defaults or not expert_enabled:
            self.grid_max_cells_input.SetValue(DEFAULT_GRID_MAX_CELLS)
            self.grid_target_cells_input.SetValue(DEFAULT_GRID_TARGET_CELLS)
        self.grid_max_cells_input.Enable(expert_enabled)
        self.grid_target_cells_input.Enable(expert_enabled)

    def _on_grid_expert_limits_toggle(self, event):
        """Handle Expert Grid Limits checkbox toggle."""
        self._apply_grid_expert_state(reset_to_defaults=not self.chk_grid_expert_limits.GetValue())
        if self.chk_grid_expert_limits.GetValue():
            self.grid_pane.Expand()
        self._refresh_preflight()

    def _on_current_enabled_toggle(self, event):
        """Refresh summaries when current simulation is toggled."""
        self._render_current_groups()

    def _apply_current_mode_visibility(self):
        """Show only the current-entry controls relevant to the selected mode."""
        total_mode = self.current_mode_choice.GetSelection() == 1
        self.current_total_row.ShowItems(total_mode)
        self.current_pad_current_row.ShowItems(not total_mode)
        self.current_pad_list_row.ShowItems(not total_mode)
        try:
            self.tab_current.Layout()
        except Exception:
            pass

    def _on_current_mode_changed(self, event):
        """Apply a current distribution mode and refresh its summary."""
        self._sync_current_group_from_fields()
        self._apply_current_mode_visibility()
        self._render_current_groups()

    def _on_current_new_group(self, event):
        """Create a current group from the live KiCad pad selection."""
        pads = self._selection_descriptors()
        if not pads:
            wx.MessageBox("Select one or more pads in KiCad first.", "ThermalSim")
            return
        group_idx = len(self.current_groups)
        self.current_groups.append({
            'name': f"Group {group_idx + 1}",
            'color': CURRENT_GROUP_COLORS[group_idx % len(CURRENT_GROUP_COLORS)],
            'mode': 'per_pad',
            'total_current_a': 0.0,
            'pads': [normalize_pad_descriptor(p) for p in pads],
        })
        self.current_group_index = group_idx
        self.chk_current_enabled.SetValue(True)
        self._render_current_groups()

    def _on_current_add_selection(self, event):
        """Add the live KiCad pad selection to the selected current group."""
        self._sync_current_group_from_fields()
        group = self._selected_current_group()
        if group is None:
            self._on_current_new_group(event)
            return
        pads = self._selection_descriptors()
        if not pads:
            wx.MessageBox("Select one or more pads in KiCad first.", "ThermalSim")
            return
        existing = {p.get('pad_key') for p in group.get('pads', [])}
        for pad in pads:
            descriptor = normalize_pad_descriptor(pad)
            if descriptor['pad_key'] not in existing:
                group.setdefault('pads', []).append(descriptor)
                existing.add(descriptor['pad_key'])
        self._render_current_groups()

    def _on_current_remove_pads(self, event):
        """Remove selected pad rows from the selected group."""
        self._sync_current_group_from_fields()
        group = self._selected_current_group()
        if group is None:
            return
        indices = self._selected_current_pad_indices()
        if not indices:
            return
        group['pads'] = [
            pad for idx, pad in enumerate(group.get('pads', []))
            if idx not in set(indices)
        ]
        self._render_current_groups()

    def _on_current_duplicate_group(self, event):
        """Duplicate the selected group."""
        self._sync_current_group_from_fields()
        group = self._selected_current_group()
        if group is None:
            return
        duplicate = copy.deepcopy(group)
        duplicate['name'] = f"{group.get('name', 'Group')} Copy"
        duplicate['color'] = CURRENT_GROUP_COLORS[len(self.current_groups) % len(CURRENT_GROUP_COLORS)]
        self.current_groups.append(duplicate)
        self.current_group_index = len(self.current_groups) - 1
        self._render_current_groups()

    def _on_current_delete_group(self, event):
        """Delete the selected group."""
        self._sync_current_group_from_fields()
        if 0 <= self.current_group_index < len(self.current_groups):
            del self.current_groups[self.current_group_index]
        self.current_group_index = min(self.current_group_index, len(self.current_groups) - 1)
        self._render_current_groups()

    def _on_current_group_selected(self, event):
        """Handle current group list selection changes."""
        self._sync_current_group_from_fields()
        try:
            self.current_group_index = event.GetIndex()
        except Exception:
            try:
                self.current_group_index = self.current_group_list.GetFirstSelected()
            except Exception:
                self.current_group_index = -1
        self._render_current_group_editor()

    def _on_current_apply_group_fields(self, event):
        """Apply name, mode, and total current edits to the selected group."""
        self._sync_current_group_from_fields()
        self._render_current_groups()

    def _on_current_apply_pad_current(self, event):
        """Apply a per-pad current to selected pad rows or all pads."""
        group = self._selected_current_group()
        if group is None:
            return
        group['mode'] = 'per_pad'
        current = float(self.current_pad_value_input.GetValue())
        indices = self._selected_current_pad_indices()
        if not indices:
            indices = list(range(len(group.get('pads', []))))
        for idx in indices:
            if 0 <= idx < len(group.get('pads', [])):
                group['pads'][idx]['current_a'] = current
        self.current_mode_choice.SetSelection(0)
        self._render_current_groups()

    def _on_current_apply_pad_current_list(self, event):
        """Apply a comma-separated list of pad currents to the selected group."""
        group = self._selected_current_group()
        if group is None:
            return
        text = self.current_pad_list_input.GetValue().strip()
        if not text:
            return
        try:
            values = [
                float(part.strip())
                for part in text.replace(";", ",").split(",")
                if part.strip()
            ]
        except ValueError:
            wx.MessageBox("Enter comma-separated numeric currents, e.g. +6, -4, -2.", "ThermalSim")
            return
        pads = group.get('pads', [])
        if len(values) != len(pads):
            wx.MessageBox(
                f"Current list has {len(values)} values, but the group has {len(pads)} pads.",
                "ThermalSim"
            )
            return
        group['mode'] = 'per_pad'
        for idx, value in enumerate(values):
            pads[idx]['current_a'] = value
        self.current_mode_choice.SetSelection(0)
        self._render_current_groups()

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
        else:
            _, balance_rows = summarize_current_groups(self.current_groups)
            if not balance_rows:
                current_text = "Current heating on / no terminals"
            elif any(status != "OK" for _, _, status in balance_rows):
                current_text = "Current heating on / needs balance"
            else:
                current_text = "Current heating on / balanced"
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

    def _selected_current_group(self):
        """Return the selected current group or None."""
        if 0 <= self.current_group_index < len(self.current_groups):
            return self.current_groups[self.current_group_index]
        return None

    def _selected_current_pad_indices(self):
        """Return selected pad row indices in the pad table."""
        indices = []
        try:
            idx = self.current_pad_list.GetFirstSelected()
            while idx != -1:
                indices.append(idx)
                idx = self.current_pad_list.GetNextItem(idx, getattr(wx, "LIST_NEXT_ALL", 0), getattr(wx, "LIST_STATE_SELECTED", 0))
        except Exception:
            pass
        return indices

    def _sync_current_group_from_fields(self):
        """Copy editor fields into the selected group."""
        group = self._selected_current_group()
        if group is None:
            return
        group['name'] = self.current_name_input.GetValue().strip() or group.get('name', 'Group')
        group['mode'] = 'total' if self.current_mode_choice.GetSelection() == 1 else 'per_pad'
        group['total_current_a'] = float(self.current_total_input.GetValue())

    def _render_current_groups(self):
        """Refresh group list, pad table, and net-balance text."""
        self.current_groups = prepare_current_groups(self.current_groups)
        if self.current_groups and not (0 <= self.current_group_index < len(self.current_groups)):
            self.current_group_index = 0
        try:
            self.current_group_list.DeleteAllItems()
            rows, balance_rows = summarize_current_groups(self.current_groups)
            for row_idx, row in enumerate(rows):
                self.current_group_list.InsertItem(row_idx, row[0])
                for col_idx, value in enumerate(row[1:], start=1):
                    self.current_group_list.SetItem(row_idx, col_idx, value)
            if 0 <= self.current_group_index < len(self.current_groups):
                self.current_group_list.Select(self.current_group_index)
        except Exception:
            _, balance_rows = summarize_current_groups(self.current_groups)

        lines = []
        for net, total, status in balance_rows:
            lines.append(f"{net}: {total} - {status}")
        if not lines:
            lines.append("No current groups configured.")
        self.current_balance_text.SetValue("\n".join(lines))
        self._render_current_group_editor()
        if hasattr(self, "lbl_context"):
            self._refresh_context_summary()
        if hasattr(self, "lbl_preflight"):
            self._refresh_preflight()

    def _render_current_group_editor(self):
        """Refresh editor fields for the selected group."""
        group = self._selected_current_group()
        if group is None:
            self.current_name_input.SetValue("")
            self.current_total_input.SetValue(0.0)
            self.current_mode_choice.SetSelection(0)
            self.current_pad_list_input.SetValue("")
            try:
                self.current_pad_list.DeleteAllItems()
            except Exception:
                pass
            self._apply_current_mode_visibility()
            return
        self.current_name_input.SetValue(str(group.get('name', '')))
        self.current_mode_choice.SetSelection(1 if group.get('mode') == 'total' else 0)
        self.current_total_input.SetValue(float(group.get('total_current_a', 0.0)))
        self.current_pad_list_input.SetValue(
            ", ".join(f"{float(pad.get('current_a', 0.0)):.6g}" for pad in group.get('pads', []))
        )

        try:
            self.current_pad_list.DeleteAllItems()
            group_nets = {pad.get('net_name') or "(no net)" for pad in group.get('pads', [])}
            for row_idx, pad in enumerate(group.get('pads', [])):
                net = pad.get('net_name') or "(no net)"
                if net == "(no net)":
                    status = "No net"
                elif len(group_nets) > 1:
                    status = "Mixed nets"
                else:
                    status = "OK"
                values = [
                    pad.get('name', ''),
                    net,
                    pad.get('layer', ''),
                    f"{float(pad.get('current_a', 0.0)):.6g}",
                    status,
                ]
                self.current_pad_list.InsertItem(row_idx, values[0])
                for col_idx, value in enumerate(values[1:], start=1):
                    self.current_pad_list.SetItem(row_idx, col_idx, value)
        except Exception:
            pass
        self._apply_current_mode_visibility()

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
        - use_heatsink : bool
        - pad_th : float
        - pad_k : float
        - pad_cap_areal : float
        - h_conv : float
        - grid_expert_limits : bool
        - grid_max_cells : int
        - grid_target_cells : int
        """
        try:
            self._sync_current_group_from_fields()
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
            current_groups = prepare_current_groups(self.current_groups)
            power_str = power_pads_to_power_str(power_pads, self.power_input.GetValue())
            grid_expert_limits = self.chk_grid_expert_limits.GetValue()
            if grid_expert_limits:
                grid_max_cells = int(self.grid_max_cells_input.GetValue())
                grid_target_cells = int(self.grid_target_cells_input.GetValue())
                if (
                    grid_max_cells < 1000
                    or grid_target_cells < 1000
                    or grid_target_cells > grid_max_cells
                ):
                    raise ValueError
            else:
                grid_max_cells = DEFAULT_GRID_MAX_CELLS
                grid_target_cells = DEFAULT_GRID_TARGET_CELLS
            backend_label = self.solver_backend_choice.GetStringSelection().lower()
            if "pardiso" in backend_label:
                solver_backend = "pardiso"
            elif "scipy" in backend_label:
                solver_backend = "scipy"
            else:
                solver_backend = "auto"
            stepping_modes = ["auto", "multi_phase", "two_phase", "uniform"]
            stepping_idx = self.time_stepping_choice.GetSelection()
            time_stepping = stepping_modes[stepping_idx] if 0 <= stepping_idx < len(stepping_modes) else "auto"
            return {
                'power_str': power_str,
                'power_pads': power_pads,
                'time': float(self.time_input.GetValue()),
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
                'use_heatsink': self.chk_heatsink.GetValue(),
                'pad_th': float(self.pad_thick.GetValue()),
                'pad_k': float(self.pad_k.GetValue()),
                'pad_cap_areal': float(self.pad_cap.GetValue()),
                'h_conv': float(self.h_conv_input.GetValue()),
                'grid_expert_limits': grid_expert_limits,
                'grid_max_cells': grid_max_cells,
                'grid_target_cells': grid_target_cells,
                'solver_backend': solver_backend,
                'time_stepping': time_stepping,
                'current_enabled': self.chk_current_enabled.GetValue(),
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
            if 'thick' in defaults:
                self.thick_input.SetValue(float(defaults['thick']))
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
            self.snap_count_input.Enable(self.chk_snapshots.GetValue())

            out_dir = defaults.get('output_dir')
            if out_dir:
                self.output_dir_input.SetValue(str(out_dir))

            self.chk_ignore_traces.SetValue(
                bool(defaults.get('ignore_traces', self.chk_ignore_traces.GetValue()))
            )
            self.chk_limit_area.SetValue(
                bool(defaults.get('limit_area', self.chk_limit_area.GetValue()))
            )
            if 'pad_dist_mm' in defaults:
                self.pad_dist_input.SetValue(float(defaults['pad_dist_mm']))
            self.pad_dist_input.Enable(self.chk_limit_area.GetValue())

            self.chk_heatsink.SetValue(
                bool(defaults.get('use_heatsink', self.chk_heatsink.GetValue()))
            )
            for control in self._thermal_pad_controls:
                control.Enable(self.chk_heatsink.GetValue())
            if 'pad_th' in defaults:
                self.pad_thick.SetValue(float(defaults['pad_th']))
            if 'pad_k' in defaults:
                self.pad_k.SetValue(float(defaults['pad_k']))
            if 'pad_cap_areal' in defaults:
                self.pad_cap.SetValue(float(defaults['pad_cap_areal']))

            if 'h_conv' in defaults:
                self.h_conv_input.SetValue(float(defaults['h_conv']))

            backend = str(defaults.get('solver_backend', 'auto')).lower()
            if backend == 'pardiso' and HAS_PARDISO:
                self.solver_backend_choice.SetSelection(1)
            elif backend == 'scipy':
                self.solver_backend_choice.SetSelection(2 if HAS_PARDISO else 1)
            else:
                self.solver_backend_choice.SetSelection(0)

            stepping = str(defaults.get('time_stepping', 'auto')).lower()
            stepping_idx = {'auto': 0, 'multi_phase': 1, 'two_phase': 2, 'uniform': 3}.get(stepping, 0)
            self.time_stepping_choice.SetSelection(stepping_idx)

            grid_expert_limits = bool(defaults.get('grid_expert_limits', False))
            self.chk_grid_expert_limits.SetValue(grid_expert_limits)
            if grid_expert_limits:
                grid_max_cells = _safe_int(defaults.get('grid_max_cells'), DEFAULT_GRID_MAX_CELLS)
                grid_target_cells = _safe_int(
                    defaults.get('grid_target_cells'),
                    DEFAULT_GRID_TARGET_CELLS,
                )
                if (
                    grid_max_cells < 1000
                    or grid_target_cells < 1000
                    or grid_target_cells > grid_max_cells
                ):
                    grid_expert_limits = False
                    grid_max_cells = DEFAULT_GRID_MAX_CELLS
                    grid_target_cells = DEFAULT_GRID_TARGET_CELLS
                    self.chk_grid_expert_limits.SetValue(False)
                self.grid_max_cells_input.SetValue(grid_max_cells)
                self.grid_target_cells_input.SetValue(grid_target_cells)
                self._apply_grid_expert_state(reset_to_defaults=not grid_expert_limits)
            else:
                self._apply_grid_expert_state(reset_to_defaults=True)

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
            self.current_groups = prepare_current_groups(defaults.get('current_groups', []))
            self.current_group_index = 0 if self.current_groups else -1
            self._render_current_groups()
            if self.chk_heatsink.GetValue():
                self.thermal_pad_pane.Expand()
            if self.chk_grid_expert_limits.GetValue():
                self.grid_pane.Expand()
            if (
                str(defaults.get('solver_backend', 'auto')).lower() != 'auto'
                or str(defaults.get('time_stepping', 'auto')).lower() != 'auto'
                or float(defaults.get('h_conv', 10.0)) != 10.0
                or float(defaults.get('thick', 1.6)) != 1.6
            ):
                self.solver_pane.Expand()
        except Exception:
            pass
