"""
GUI dialogs for ThermalSim.

This module provides the wxPython dialog for configuring thermal
simulation parameters.
"""

import os
import wx


class SettingsDialog(wx.Dialog):
    """
    Dialog for configuring thermal simulation parameters.

    This dialog allows users to set simulation parameters including:
    - Power per pad
    - Simulation duration
    - Ambient temperature
    - Grid resolution
    - Output options
    - Geometry filters
    - Thermal pad settings

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
        defaults=None
    ):
        super().__init__(parent, title="Thermal Sim")

        self.layer_names = layer_names
        self.preview_callback = preview_callback

        sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Info ---
        info_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Stackup")
        l_str = f"{len(layer_names)} Layers found"
        lbl_layers = wx.StaticText(self, label=l_str)
        info_box.Add(lbl_layers, 0, wx.ALL, 5)

        # Detailed stackup (from saved .kicad_pcb stackup), shown in um
        if stackup_details:
            self.txt_stackup = wx.TextCtrl(
                self,
                value=stackup_details,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
            )
            # Keep the dialog compact but scrollable
            self.txt_stackup.SetMinSize((-1, 120))
            info_box.Add(self.txt_stackup, 0, wx.EXPAND | wx.ALL, 5)

        sizer.Add(info_box, 0, wx.EXPAND | wx.ALL, 5)

        # --- Pads (selected / recognized) ---
        pad_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Pads")
        pad_lines = pad_names if isinstance(pad_names, (list, tuple)) else []
        pad_text = "\n".join(str(x) for x in pad_lines)
        self.txt_pads = wx.TextCtrl(
            self,
            value=pad_text,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
        )
        self.txt_pads.SetMinSize((-1, 90))
        pad_box.Add(self.txt_pads, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(pad_box, 0, wx.EXPAND | wx.ALL, 5)

        # --- Setup ---
        box_main = wx.StaticBoxSizer(wx.VERTICAL, self, "Parameters")

        lbl_pwr = wx.StaticText(
            self,
            label="Pad Power (W): one value for all, or comma-separated per pad"
        )
        box_main.Add(lbl_pwr, 0, wx.ALL, 5)
        self.power_input = wx.TextCtrl(self, value="1.0")
        box_main.Add(self.power_input, 0, wx.EXPAND | wx.ALL, 5)

        self.time_input = self._add_field(box_main, "Duration (sec):", "20.0")
        self.amb_input = self._add_field(box_main, "Ambient Temp (C):", "25.0")
        self.thick_input = self._add_field(box_main, "PCB Thickness (mm):", "1.6")

        sizer.Add(box_main, 0, wx.EXPAND | wx.ALL, 5)

        # --- Options ---
        box_out = wx.StaticBoxSizer(wx.VERTICAL, self, "Output")
        self.chk_all_layers = wx.CheckBox(self, label="Show All Layers")
        self.chk_all_layers.SetValue(True)
        box_out.Add(self.chk_all_layers, 0, wx.ALL, 5)

        self.chk_snapshots = wx.CheckBox(self, label="Save Snapshots")
        self.chk_snapshots.SetValue(False)
        box_out.Add(self.chk_snapshots, 0, wx.ALL, 5)

        self.snap_count_input = self._add_field(box_out, "Snapshots count:", "5")
        self.snap_count_input.Enable(False)
        self.chk_snapshots.Bind(wx.EVT_CHECKBOX, self._on_snapshots_toggle)

        sizer.Add(box_out, 0, wx.EXPAND | wx.ALL, 5)

        # --- Output Folder ---
        box_path = wx.StaticBoxSizer(wx.VERTICAL, self, "Output Folder")
        row_path = wx.BoxSizer(wx.HORIZONTAL)
        self.output_dir_input = wx.TextCtrl(self, value=default_output_dir)
        btn_browse = wx.Button(self, label="Browse...")
        btn_browse.Bind(wx.EVT_BUTTON, self._on_browse_output)
        row_path.Add(self.output_dir_input, 1, wx.EXPAND | wx.RIGHT, 5)
        row_path.Add(btn_browse, 0)
        box_path.Add(row_path, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(box_path, 0, wx.EXPAND | wx.ALL, 5)

        # --- Filters ---
        box_filter = wx.StaticBoxSizer(wx.VERTICAL, self, "Geometry Filters")
        self.chk_ignore_traces = wx.CheckBox(self, label="Ignore Traces")
        self.chk_ignore_traces.SetValue(False)
        box_filter.Add(self.chk_ignore_traces, 0, wx.ALL, 5)

        self.chk_limit_area = wx.CheckBox(self, label="Limit Area to Pads")
        self.chk_limit_area.SetValue(False)
        box_filter.Add(self.chk_limit_area, 0, wx.ALL, 5)

        self.pad_dist_input = self._add_field(box_filter, "Limit Distance (mm):", "30")
        self.pad_dist_input.Enable(False)
        self.chk_limit_area.Bind(wx.EVT_CHECKBOX, self._on_limit_area_toggle)
        sizer.Add(box_filter, 0, wx.EXPAND | wx.ALL, 5)

        # --- Thermal Pad ---
        box_cool = wx.StaticBoxSizer(wx.VERTICAL, self, "Thermal Pad (User.Eco1)")
        self.chk_heatsink = wx.CheckBox(self, label="Enable Pad Simulation")
        self.chk_heatsink.SetValue(False)
        box_cool.Add(self.chk_heatsink, 0, wx.ALL, 5)

        self.pad_thick = self._add_field(box_cool, "Pad Thickness (mm):", "1.0")
        self.pad_k = self._add_field(box_cool, "Pad Cond. (W/mK):", "3.0")
        self.pad_cap = self._add_field(box_cool, "Pad Heat Cap. (J/m2K):", "0.0")

        sizer.Add(box_cool, 0, wx.EXPAND | wx.ALL, 5)

        # --- Grid ---
        box_grid = wx.StaticBoxSizer(wx.VERTICAL, self, "Grid Resolution")
        self.res_input = self._add_field(box_grid, "Resolution (mm):", str(suggested_res))
        sizer.Add(box_grid, 0, wx.EXPAND | wx.ALL, 5)

        # --- Buttons ---
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.btn_preview = wx.Button(self, label="Preview")
        self.btn_preview.Bind(wx.EVT_BUTTON, self._on_preview)
        btn_sizer.Add(self.btn_preview, 0, wx.ALL, 5)

        btn_sizer.AddStretchSpacer()

        btn_run = wx.Button(self, wx.ID_OK, "Run")
        btn_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(btn_run, 0, wx.ALL, 5)
        btn_sizer.Add(btn_cancel, 0, wx.ALL, 5)

        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(sizer)
        self.Fit()
        self.Center()

        if defaults:
            self._apply_defaults(defaults)

    def _add_field(self, sizer, label_text, default_val):
        """
        Add a labeled text field to a sizer.

        Parameters
        ----------
        sizer : wx.Sizer
            Parent sizer to add the field to.
        label_text : str
            Label text for the field.
        default_val : str
            Default value for the text control.

        Returns
        -------
        wx.TextCtrl
            The created text control.
        """
        row = wx.BoxSizer(wx.HORIZONTAL)
        lbl = wx.StaticText(self, label=label_text, size=(160, -1))
        txt = wx.TextCtrl(self, value=default_val)
        row.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        row.Add(txt, 1, wx.EXPAND)
        sizer.Add(row, 0, wx.EXPAND | wx.ALL, 2)
        return txt

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
                'pad_cap_areal': float(self.pad_cap.GetValue())
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
            self.time_input.SetValue(
                str(defaults.get('time', self.time_input.GetValue()))
            )
            self.amb_input.SetValue(
                str(defaults.get('amb', self.amb_input.GetValue()))
            )
            self.thick_input.SetValue(
                str(defaults.get('thick', self.thick_input.GetValue()))
            )
            self.res_input.SetValue(
                str(defaults.get('res', self.res_input.GetValue()))
            )
            self.chk_all_layers.SetValue(
                bool(defaults.get('show_all', self.chk_all_layers.GetValue()))
            )
            self.chk_snapshots.SetValue(
                bool(defaults.get('snapshots', self.chk_snapshots.GetValue()))
            )
            self.snap_count_input.SetValue(
                str(defaults.get('snap_count', self.snap_count_input.GetValue()))
            )
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
            self.pad_dist_input.SetValue(
                str(defaults.get('pad_dist_mm', self.pad_dist_input.GetValue()))
            )
            self.pad_dist_input.Enable(self.chk_limit_area.GetValue())
            self.chk_heatsink.SetValue(
                bool(defaults.get('use_heatsink', self.chk_heatsink.GetValue()))
            )
            self.pad_thick.SetValue(
                str(defaults.get('pad_th', self.pad_thick.GetValue()))
            )
            self.pad_k.SetValue(
                str(defaults.get('pad_k', self.pad_k.GetValue()))
            )
            self.pad_cap.SetValue(
                str(defaults.get('pad_cap_areal', self.pad_cap.GetValue()))
            )
        except Exception:
            pass
