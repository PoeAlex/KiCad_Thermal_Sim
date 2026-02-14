"""
GUI dialogs for ThermalSim.

This module provides the wxPython dialog for configuring thermal
simulation parameters, organized into two notebook tabs:
Simulation (everyday settings) and Advanced (rarely changed).
"""

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
    'pads': "Selected pads that will be used as heat sources.",
    'power': "Heat dissipation per pad in Watts. Single value, comma-separated, or PWL file path.",
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
    'capabilities': "Detected solver backends. PyPardiso accelerates large grids significantly.",
    'help': "Open the ThermalSim documentation in your web browser.",
    'preview': "Generate a geometry preview image without running the simulation.",
    'current_paths': "Enable I\u00b2R analysis: define pad pairs with current values to compute trace heating.",
    'current_pad_plus': "Select the source pad (current injection). Pad - will auto-filter to same net.",
    'current_pad_minus': "Select the sink pad (current extraction). Only pads on the same net as Pad + are shown.",
    'current_add': "Add the selected pad pair to the analysis list.",
    'current_remove': "Remove the selected pair from the list.",
    'current_input': "Current in amperes for the next pair to add.",
}


class SettingsDialog(wx.Dialog):
    """
    Dialog for configuring thermal simulation parameters.

    Organized into two notebook tabs:
    - Simulation: everyday settings (power, duration, output)
    - Advanced: geometry filters, thermal pad, solver options

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
        stackup_details="",
        pad_names=None,
        default_output_dir="",
        defaults=None,
        all_pads=None
    ):
        super().__init__(parent, title="Thermal Sim")

        self.layer_names = layer_names
        self.preview_callback = preview_callback
        # all_pads: list of (ref_str, net_name, net_code) for every pad on the board
        self._all_pads = all_pads or []

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Notebook ---
        self.notebook = wx.Notebook(self)

        # Tab 1: Simulation
        self.tab_sim = wx.Panel(self.notebook)
        self._build_simulation_tab(
            self.tab_sim, layer_names, stackup_details,
            pad_names, suggested_res, default_output_dir
        )
        self.notebook.AddPage(self.tab_sim, "Simulation")

        # Tab 2: Advanced
        self.tab_adv = wx.Panel(self.notebook)
        self._build_advanced_tab(self.tab_adv)
        self.notebook.AddPage(self.tab_adv, "Advanced")

        # Tab 3: Current Paths
        self.tab_current = wx.Panel(self.notebook)
        self._build_current_paths_tab(self.tab_current)
        self.notebook.AddPage(self.tab_current, "Current Paths")

        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)

        # --- Button Bar (always visible below notebook) ---
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Help link/button
        if _HAS_WX_ADV:
            help_link = wx.adv.HyperlinkCtrl(
                self, wx.ID_ANY, "Help",
                "https://github.com/PoeAlex/KiCad_Thermal_Sim#readme"
            )
            help_link.SetToolTip(TOOLTIP_TEXTS['help'])
            btn_sizer.Add(help_link, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        else:
            btn_help = wx.Button(self, label="Help")
            btn_help.SetToolTip(TOOLTIP_TEXTS['help'])
            btn_help.Bind(wx.EVT_BUTTON, self._on_help)
            btn_sizer.Add(btn_help, 0, wx.ALL, 5)

        self.btn_preview = wx.Button(self, label="Preview")
        self.btn_preview.Bind(wx.EVT_BUTTON, self._on_preview)
        self.btn_preview.SetToolTip(TOOLTIP_TEXTS['preview'])
        btn_sizer.Add(self.btn_preview, 0, wx.ALL, 5)

        btn_sizer.AddStretchSpacer()

        btn_run = wx.Button(self, wx.ID_OK, "Run")
        btn_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(btn_run, 0, wx.ALL, 5)
        btn_sizer.Add(btn_cancel, 0, wx.ALL, 5)

        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(main_sizer)
        self.SetSize((520, 680))
        self.Center()

        if defaults:
            self._apply_defaults(defaults)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_simulation_tab(self, panel, layer_names, stackup_details,
                              pad_names, suggested_res, default_output_dir):
        """Build the Simulation tab contents."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Board Info ---
        info_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Board Info")

        l_str = f"{len(layer_names)} Layers found"
        lbl_layers = wx.StaticText(panel, label=l_str)
        info_box.Add(lbl_layers, 0, wx.ALL, 3)

        if stackup_details:
            self.txt_stackup = wx.TextCtrl(
                panel, value=stackup_details,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
            )
            self.txt_stackup.SetMinSize((-1, 80))
            self.txt_stackup.SetToolTip(TOOLTIP_TEXTS['stackup'])
            info_box.Add(self.txt_stackup, 0, wx.EXPAND | wx.ALL, 3)

        pad_lines = pad_names if isinstance(pad_names, (list, tuple)) else []
        pad_text = "\n".join(str(x) for x in pad_lines)
        self.txt_pads = wx.TextCtrl(
            panel, value=pad_text,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
        )
        self.txt_pads.SetMinSize((-1, 60))
        self.txt_pads.SetToolTip(TOOLTIP_TEXTS['pads'])
        info_box.Add(self.txt_pads, 0, wx.EXPAND | wx.ALL, 3)

        sizer.Add(info_box, 0, wx.EXPAND | wx.ALL, 5)

        # --- Parameters ---
        box_params = wx.StaticBoxSizer(wx.VERTICAL, panel, "Parameters")

        # Power (TextCtrl + Browse)
        lbl_pwr = wx.StaticText(
            panel, label="Power (W or PWL file), comma-sep per pad"
        )
        box_params.Add(lbl_pwr, 0, wx.ALL, 3)
        row_pwr = wx.BoxSizer(wx.HORIZONTAL)
        self.power_input = wx.TextCtrl(panel, value="1.0")
        self.power_input.SetToolTip(TOOLTIP_TEXTS['power'])
        row_pwr.Add(self.power_input, 1, wx.EXPAND | wx.RIGHT, 5)
        btn_browse_pwl = wx.Button(panel, label="Browse PWL...")
        btn_browse_pwl.Bind(wx.EVT_BUTTON, self._on_browse_pwl)
        btn_browse_pwl.SetToolTip(TOOLTIP_TEXTS['browse_pwl'])
        row_pwr.Add(btn_browse_pwl, 0)
        box_params.Add(row_pwr, 0, wx.EXPAND | wx.ALL, 3)

        # Duration
        self.time_input = self._add_spin_field(
            box_params, panel, "Duration (sec):", 20.0,
            min_val=0.1, max_val=3600.0, inc=1.0, digits=1,
            tooltip_key='duration'
        )

        # Ambient Temperature
        self.amb_input = self._add_spin_field(
            box_params, panel, "Ambient Temp (\u00b0C):", 25.0,
            min_val=-40.0, max_val=200.0, inc=1.0, digits=1,
            tooltip_key='ambient'
        )

        # Resolution
        self.res_input = self._add_spin_field(
            box_params, panel, "Resolution (mm):", suggested_res,
            min_val=0.05, max_val=10.0, inc=0.05, digits=2,
            tooltip_key='resolution'
        )

        sizer.Add(box_params, 0, wx.EXPAND | wx.ALL, 5)

        # --- Output ---
        box_out = wx.StaticBoxSizer(wx.VERTICAL, panel, "Output")

        self.chk_all_layers = wx.CheckBox(panel, label="Show All Layers")
        self.chk_all_layers.SetValue(True)
        self.chk_all_layers.SetToolTip(TOOLTIP_TEXTS['show_all'])
        box_out.Add(self.chk_all_layers, 0, wx.ALL, 3)

        self.chk_snapshots = wx.CheckBox(panel, label="Save Snapshots")
        self.chk_snapshots.SetValue(False)
        self.chk_snapshots.SetToolTip(TOOLTIP_TEXTS['snapshots'])
        box_out.Add(self.chk_snapshots, 0, wx.ALL, 3)

        self.snap_count_input = self._add_int_spin_field(
            box_out, panel, "Snapshot Count:", 5,
            min_val=1, max_val=50,
            tooltip_key='snap_count'
        )
        self.snap_count_input.Enable(False)
        self.chk_snapshots.Bind(wx.EVT_CHECKBOX, self._on_snapshots_toggle)

        # Output folder
        row_path = wx.BoxSizer(wx.HORIZONTAL)
        lbl_path = wx.StaticText(panel, label="Output Folder:", size=(100, -1))
        self.output_dir_input = wx.TextCtrl(panel, value=default_output_dir)
        self.output_dir_input.SetToolTip(TOOLTIP_TEXTS['output_dir'])
        btn_browse = wx.Button(panel, label="Browse...")
        btn_browse.Bind(wx.EVT_BUTTON, self._on_browse_output)
        btn_browse.SetToolTip(TOOLTIP_TEXTS['browse_output'])
        row_path.Add(lbl_path, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        row_path.Add(self.output_dir_input, 1, wx.EXPAND | wx.RIGHT, 5)
        row_path.Add(btn_browse, 0)
        box_out.Add(row_path, 0, wx.EXPAND | wx.ALL, 3)

        sizer.Add(box_out, 0, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(sizer)

    def _build_advanced_tab(self, panel):
        """Build the Advanced tab contents."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Geometry Filters ---
        box_filter = wx.StaticBoxSizer(wx.VERTICAL, panel, "Geometry Filters")

        self.chk_ignore_traces = wx.CheckBox(panel, label="Ignore Traces")
        self.chk_ignore_traces.SetValue(False)
        self.chk_ignore_traces.SetToolTip(TOOLTIP_TEXTS['ignore_traces'])
        box_filter.Add(self.chk_ignore_traces, 0, wx.ALL, 3)

        self.chk_limit_area = wx.CheckBox(panel, label="Limit Area to Pads")
        self.chk_limit_area.SetValue(False)
        self.chk_limit_area.SetToolTip(TOOLTIP_TEXTS['limit_area'])
        box_filter.Add(self.chk_limit_area, 0, wx.ALL, 3)

        self.pad_dist_input = self._add_spin_field(
            box_filter, panel, "Limit Distance (mm):", 30.0,
            min_val=1.0, max_val=200.0, inc=5.0, digits=1,
            tooltip_key='limit_dist'
        )
        self.pad_dist_input.Enable(False)
        self.chk_limit_area.Bind(wx.EVT_CHECKBOX, self._on_limit_area_toggle)

        sizer.Add(box_filter, 0, wx.EXPAND | wx.ALL, 5)

        # --- Thermal Pad ---
        box_pad = wx.StaticBoxSizer(wx.VERTICAL, panel, "Thermal Pad (User.Eco1)")

        self.chk_heatsink = wx.CheckBox(panel, label="Enable Pad Simulation")
        self.chk_heatsink.SetValue(False)
        self.chk_heatsink.SetToolTip(TOOLTIP_TEXTS['enable_pad'])
        box_pad.Add(self.chk_heatsink, 0, wx.ALL, 3)

        self.pad_thick = self._add_spin_field(
            box_pad, panel, "Pad Thickness (mm):", 1.0,
            min_val=0.1, max_val=50.0, inc=0.5, digits=2,
            tooltip_key='pad_thick'
        )

        self.pad_k = self._add_spin_field(
            box_pad, panel, "Pad Cond. (W/mK):", 3.0,
            min_val=0.01, max_val=500.0, inc=1.0, digits=1,
            tooltip_key='pad_k'
        )

        self.pad_cap = self._add_spin_field(
            box_pad, panel, "Pad Heat Cap. (J/m\u00b2K):", 0.0,
            min_val=0.0, max_val=100000.0, inc=100.0, digits=0,
            tooltip_key='pad_cap'
        )

        sizer.Add(box_pad, 0, wx.EXPAND | wx.ALL, 5)

        # --- Solver ---
        box_solver = wx.StaticBoxSizer(wx.VERTICAL, panel, "Solver")

        self.h_conv_input = self._add_spin_field(
            box_solver, panel, "Convection h (W/m\u00b2K):", 10.0,
            min_val=1.0, max_val=200.0, inc=1.0, digits=1,
            tooltip_key='h_conv'
        )

        self.thick_input = self._add_spin_field(
            box_solver, panel, "PCB Thickness (mm):", 1.6,
            min_val=0.1, max_val=10.0, inc=0.1, digits=2,
            tooltip_key='pcb_thick'
        )

        sizer.Add(box_solver, 0, wx.EXPAND | wx.ALL, 5)

        # --- Capabilities (read-only) ---
        box_cap = wx.StaticBoxSizer(wx.VERTICAL, panel, "Capabilities")
        solver_str = "Solver: SciPy + PyPardiso" if HAS_PARDISO else "Solver: SciPy"
        lbl_cap = wx.StaticText(panel, label=solver_str)
        lbl_cap.SetToolTip(TOOLTIP_TEXTS['capabilities'])
        box_cap.Add(lbl_cap, 0, wx.ALL, 5)
        sizer.Add(box_cap, 0, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(sizer)

    def _build_current_paths_tab(self, panel):
        """Build the Current Paths tab for I²R analysis."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.chk_current_path = wx.CheckBox(panel, label="Enable Current Path Analysis")
        self.chk_current_path.SetValue(False)
        self.chk_current_path.SetToolTip(TOOLTIP_TEXTS['current_paths'])
        sizer.Add(self.chk_current_path, 0, wx.ALL, 5)

        # --- Pair list ---
        box_pairs = wx.StaticBoxSizer(wx.VERTICAL, panel, "Current Path Pairs")

        self.current_list = wx.ListCtrl(
            panel, style=wx.LC_REPORT | wx.LC_SINGLE_SEL
        )
        self.current_list.InsertColumn(0, "#", width=30)
        self.current_list.InsertColumn(1, "Pad +", width=100)
        self.current_list.InsertColumn(2, "Pad -", width=100)
        self.current_list.InsertColumn(3, "Current (A)", width=80)
        self.current_list.InsertColumn(4, "Net", width=100)
        self.current_list.SetMinSize((-1, 120))
        box_pairs.Add(self.current_list, 1, wx.EXPAND | wx.ALL, 3)

        # --- Pad picker dropdowns ---
        # Build unique pad labels grouped by net
        self._pad_by_net = {}  # net_code -> [(ref_str, net_name)]
        self._pad_labels = []  # ordered list of "REF-NUM [net]"
        for ref_str, net_name, net_code in self._all_pads:
            label = f"{ref_str} [{net_name}]" if net_name else ref_str
            self._pad_labels.append(label)
            self._pad_by_net.setdefault(net_code, []).append(
                (ref_str, net_name, net_code, label)
            )

        # Pad + dropdown
        row_pad_a = wx.BoxSizer(wx.HORIZONTAL)
        lbl_pad_a = wx.StaticText(panel, label="Pad + :", size=(80, -1))
        self.choice_pad_a = wx.Choice(panel, choices=self._pad_labels)
        self.choice_pad_a.SetToolTip(TOOLTIP_TEXTS['current_pad_plus'])
        self.choice_pad_a.Bind(wx.EVT_CHOICE, self._on_pad_a_changed)
        row_pad_a.Add(lbl_pad_a, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        row_pad_a.Add(self.choice_pad_a, 1, wx.EXPAND)
        box_pairs.Add(row_pad_a, 0, wx.EXPAND | wx.ALL, 3)

        # Pad - dropdown (populated dynamically based on Pad + selection)
        row_pad_b = wx.BoxSizer(wx.HORIZONTAL)
        lbl_pad_b = wx.StaticText(panel, label="Pad - :", size=(80, -1))
        self.choice_pad_b = wx.Choice(panel, choices=[])
        self.choice_pad_b.SetToolTip(TOOLTIP_TEXTS['current_pad_minus'])
        row_pad_b.Add(lbl_pad_b, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        row_pad_b.Add(self.choice_pad_b, 1, wx.EXPAND)
        box_pairs.Add(row_pad_b, 0, wx.EXPAND | wx.ALL, 3)

        # Current input
        row_cur = wx.BoxSizer(wx.HORIZONTAL)
        lbl_cur = wx.StaticText(panel, label="Current (A):", size=(80, -1))
        self.current_input = wx.SpinCtrlDouble(
            panel, value="1.0", min=0.001, max=1000.0, inc=0.5
        )
        self.current_input.SetDigits(3)
        self.current_input.SetToolTip(TOOLTIP_TEXTS['current_input'])
        row_cur.Add(lbl_cur, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        row_cur.Add(self.current_input, 1, wx.EXPAND)
        box_pairs.Add(row_cur, 0, wx.EXPAND | wx.ALL, 3)

        # Buttons
        row_btn = wx.BoxSizer(wx.HORIZONTAL)
        btn_add = wx.Button(panel, label="Add Pair")
        btn_add.Bind(wx.EVT_BUTTON, self._on_add_current_pair)
        btn_add.SetToolTip(TOOLTIP_TEXTS['current_add'])
        btn_remove = wx.Button(panel, label="Remove Selected")
        btn_remove.Bind(wx.EVT_BUTTON, self._on_remove_current_pair)
        btn_remove.SetToolTip(TOOLTIP_TEXTS['current_remove'])
        row_btn.Add(btn_add, 0, wx.RIGHT, 5)
        row_btn.Add(btn_remove, 0)
        box_pairs.Add(row_btn, 0, wx.ALL, 3)

        sizer.Add(box_pairs, 1, wx.EXPAND | wx.ALL, 5)

        # Info label
        lbl_info = wx.StaticText(
            panel,
            label="Select Pad + and Pad - from the dropdowns.\n"
                  "Pad - auto-filters to pads on the same net. Max 10 pairs."
        )
        lbl_info.SetForegroundColour(wx.Colour(100, 100, 100))
        sizer.Add(lbl_info, 0, wx.ALL, 5)

        panel.SetSizer(sizer)

        # Internal storage for pairs
        self._current_pairs = []  # list of dicts

    def _on_pad_a_changed(self, event):
        """Update Pad - dropdown to show only same-net pads when Pad + changes."""
        sel = self.choice_pad_a.GetSelection()
        self.choice_pad_b.Clear()
        if sel < 0 or sel >= len(self._all_pads):
            return
        ref_a, net_name_a, net_code_a = self._all_pads[sel]
        same_net = self._pad_by_net.get(net_code_a, [])
        for ref_str, net_name, net_code, label in same_net:
            if ref_str != ref_a:
                self.choice_pad_b.Append(label, (ref_str, net_name, net_code))

    def _on_add_current_pair(self, event):
        """Add a current path pair from the dropdown selections."""
        if len(self._current_pairs) >= 10:
            wx.MessageBox("Maximum 10 pairs allowed.", "Info")
            return

        sel_a = self.choice_pad_a.GetSelection()
        sel_b = self.choice_pad_b.GetSelection()
        if sel_a < 0 or sel_a >= len(self._all_pads):
            wx.MessageBox("Select a Pad + from the dropdown.", "Selection Error")
            return
        if sel_b < 0:
            wx.MessageBox("Select a Pad - from the dropdown.", "Selection Error")
            return

        ref_a, net_name_a, net_code_a = self._all_pads[sel_a]
        # Pad B data stored as client data on the Choice item
        pad_b_data = self.choice_pad_b.GetClientData(sel_b)
        if pad_b_data is None:
            # Fallback: parse from label
            wx.MessageBox("Select a valid Pad - from the dropdown.", "Selection Error")
            return
        ref_b, net_name_b, net_code_b = pad_b_data

        if net_code_a <= 0:
            wx.MessageBox(
                f"Pad + ({ref_a}) is not connected to any net.\n"
                f"Select a pad that belongs to a valid net.",
                "Invalid Net"
            )
            return

        current = float(self.current_input.GetValue())
        pair_info = {
            'pad_a_ref': ref_a,
            'pad_b_ref': ref_b,
            'current_a': current,
            'net_code': net_code_a,
            'net_name': net_name_a,
        }
        self._current_pairs.append(pair_info)
        self._refresh_current_list()

    def _on_remove_current_pair(self, event):
        """Remove the selected current pair from the list."""
        sel = self.current_list.GetFirstSelected()
        if sel >= 0 and sel < len(self._current_pairs):
            self._current_pairs.pop(sel)
            self._refresh_current_list()

    def _refresh_current_list(self):
        """Refresh the ListCtrl from internal pairs data."""
        self.current_list.DeleteAllItems()
        for i, p in enumerate(self._current_pairs):
            idx = self.current_list.InsertItem(i, str(i + 1))
            self.current_list.SetItem(idx, 1, p['pad_a_ref'])
            self.current_list.SetItem(idx, 2, p['pad_b_ref'])
            self.current_list.SetItem(idx, 3, f"{p['current_a']:.3f}")
            self.current_list.SetItem(idx, 4, p.get('net_name', ''))

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
        return spin

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_preview(self, event):
        """Handle Preview button click."""
        if self.preview_callback:
            settings = self.get_values()
            if settings:
                self.preview_callback(settings, self.layer_names)

    def _on_limit_area_toggle(self, event):
        """Handle Limit Area checkbox toggle."""
        self.pad_dist_input.Enable(self.chk_limit_area.GetValue())

    def _on_snapshots_toggle(self, event):
        """Handle Snapshots checkbox toggle."""
        self.snap_count_input.Enable(self.chk_snapshots.GetValue())

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
        """
        try:
            return {
                'power_str': self.power_input.GetValue(),
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
                'current_path_enabled': self.chk_current_path.GetValue(),
                'current_paths': list(self._current_pairs),
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
            if 'pad_th' in defaults:
                self.pad_thick.SetValue(float(defaults['pad_th']))
            if 'pad_k' in defaults:
                self.pad_k.SetValue(float(defaults['pad_k']))
            if 'pad_cap_areal' in defaults:
                self.pad_cap.SetValue(float(defaults['pad_cap_areal']))

            if 'h_conv' in defaults:
                self.h_conv_input.SetValue(float(defaults['h_conv']))

            # Current Paths tab
            self.chk_current_path.SetValue(
                bool(defaults.get('current_path_enabled', False))
            )
            saved_paths = defaults.get('current_paths', [])
            if isinstance(saved_paths, list):
                self._current_pairs = [
                    p for p in saved_paths
                    if isinstance(p, dict) and 'pad_a_ref' in p
                ]
                self._refresh_current_list()
        except Exception:
            pass
