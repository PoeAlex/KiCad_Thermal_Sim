import pcbnew
import os
import sys
import traceback
import math
import time
import tempfile
import re
import html
import json
import shutil
import subprocess
import importlib.util

import scipy.sparse as sp
import scipy.sparse.linalg as spla

_pardiso_spec = importlib.util.find_spec("pypardiso")
if _pardiso_spec is not None:
    import pypardiso
    HAS_PARDISO = True
else:
    HAS_PARDISO = False

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for file output
    import matplotlib.pyplot as plt
    import wx
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False


# -----------------------------------------------------------------------------
# Stackup parsing from saved .kicad_pcb (robust, avoids SWIG stackup limitations)
# -----------------------------------------------------------------------------
def _sexpr_extract_from_index(s, i):
    """Return balanced '(...)' block starting at s[i] and the next index."""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return s[i:j+1], j+1
    return None, None


def _sexpr_extract_block(s, token, start=0):
    key = "(" + token
    i = s.find(key, start)
    if i < 0:
        return None
    blk, _ = _sexpr_extract_from_index(s, i)
    return blk


def _sexpr_find_all_blocks(s, token):
    key = "(" + token
    blocks = []
    i = 0
    while True:
        i = s.find(key, i)
        if i < 0:
            break
        blk, end = _sexpr_extract_from_index(s, i)
        if not blk:
            break
        blocks.append(blk)
        i = end
    return blocks


def parse_stackup_from_board_file(board):
    """
    Parse stackup (and general thickness) from the saved .kicad_pcb file.

    Returns dict with:
      - board_thickness_mm (float|None)
      - copper (list of dict): order, name, layer_id, thickness_mm
      - dielectrics (list of dict): order, name, type, thickness_mm
      - copper_ids (list[int]) in stackup order (top->bottom)
      - dielectric_gaps_mm (list[float]) between adjacent copper layers (same length as copper_ids-1)
    """
    fn = ""
    try:
        fn = board.GetFileName()
    except Exception:
        fn = ""
    if not fn:
        return {"error": "Board has no filename (save board first)."}

    try:
        txt = open(fn, "r", encoding="utf-8", errors="ignore").read()
    except Exception as e:
        return {"error": f"Failed to read board file: {e}"}

    # general thickness
    board_thickness_mm = None
    general = _sexpr_extract_block(txt, "general")
    if general:
        m = re.search(r"\(thickness\s+([0-9.]+)\)", general)
        if m:
            try:
                board_thickness_mm = float(m.group(1))
            except Exception:
                board_thickness_mm = None

    stackup = _sexpr_extract_block(txt, "stackup")
    if not stackup:
        return {"error": "No (stackup ...) found in file. Save board after editing stackup."}

    layer_blocks = _sexpr_find_all_blocks(stackup, "layer")

    copper = []
    dielectrics = []

    for idx, lb in enumerate(layer_blocks):
        # Formats observed:
        # (layer "F.Cu" 3 (type "copper") (thickness 0.035) ...)
        # (layer "F.SilkS" (type "Top Silk Screen"))
        # (layer dielectric 4 (type "core") (thickness ...))  (rare in some files)
        name = None
        order = None

        m = re.match(r'\(layer\s+"([^"]+)"(?:\s+(\d+))?', lb)
        if m:
            name = m.group(1)
            if m.group(2):
                try:
                    order = int(m.group(2))
                except Exception:
                    order = None
        else:
            m2 = re.match(r'\(layer\s+dielectric(?:\s+(\d+))?', lb)
            if m2:
                name = "dielectric"
                if m2.group(1):
                    try:
                        order = int(m2.group(1))
                    except Exception:
                        order = None

        if order is None:
            # If no explicit numeric order, keep file order for those layers
            order = idx

        mt = re.search(r'\(type\s+"([^"]+)"\)', lb)
        typ = (mt.group(1).strip().lower() if mt else "")

        # thickness: prefer sum of sublayers if present
        th_mm = None
        sublayers = _sexpr_find_all_blocks(lb, "sublayer")
        if sublayers:
            vals = []
            for sb in sublayers:
                vals += [float(x) for x in re.findall(r"\(thickness\s+([0-9.]+)\)", sb)]
            th_mm = sum(vals) if vals else None
        else:
            mm = re.search(r"\(thickness\s+([0-9.]+)\)", lb)
            if mm:
                try:
                    th_mm = float(mm.group(1))
                except Exception:
                    th_mm = None

        # Categorize
        if typ == "copper" and name:
            try:
                lid = int(board.GetLayerID(name))
            except Exception:
                lid = None
            copper.append({"order": order, "name": name, "layer_id": lid, "thickness_mm": th_mm})
        elif ("core" in typ) or ("prepreg" in typ) or (typ == "dielectric") or (name == "dielectric"):
            dielectrics.append({"order": order, "name": name or "dielectric", "type": typ or "dielectric", "thickness_mm": th_mm})

    # Sort copper by explicit stackup order (NOT by layer id!)
    copper.sort(key=lambda d: d["order"])

    copper_ids = [d["layer_id"] for d in copper if isinstance(d.get("layer_id"), int)]
    # dielectric gaps between adjacent copper layers: sum all dielectric items with order in-between
    dielectric_gaps_mm = []
    for a, b in zip(copper, copper[1:]):
        oa, ob = a["order"], b["order"]
        if oa > ob:
            oa, ob = ob, oa
        gap = 0.0
        found = False
        for d in dielectrics:
            if oa < d["order"] < ob and d.get("thickness_mm") is not None:
                gap += float(d["thickness_mm"])
                found = True
        dielectric_gaps_mm.append(gap if found else 0.0)

    return {
        "board_thickness_mm": board_thickness_mm,
        "copper": copper,
        "dielectrics": dielectrics,
        "copper_ids": copper_ids,
        "dielectric_gaps_mm": dielectric_gaps_mm,
        "file_layer_count": len(layer_blocks),
    }


def format_stackup_report_um(stack):
    """Human-readable stackup report (compact) for GUI; thickness shown in µm."""
    if not stack or stack.get("error"):
        return f"Stackup: {stack.get('error', 'unavailable')}"
    lines = []
    bt = stack.get("board_thickness_mm", None)
    if bt is not None:
        lines.append(f"Board thickness (general): {bt:.3f} mm")
    copper = stack.get("copper", [])
    if copper:
        lines.append("Copper layers (top→bottom):")
        for c in copper:
            th = c.get("thickness_mm", None)
            th_um = (th * 1000.0) if isinstance(th, (int, float)) else None
            th_s = f"{th_um:.1f} µm" if th_um is not None else "(n/a)"
            lid = c.get("layer_id", None)
            lid_s = str(lid) if isinstance(lid, int) else "?"
            lines.append(f"  {c['name']:<8s}  {th_s:<10s}  id={lid_s}")
    gaps = stack.get("dielectric_gaps_mm", [])
    if copper and gaps and len(gaps) == max(0, len(copper)-1):
        lines.append("Dielectric gaps between copper layers:")
        for i, g in enumerate(gaps):
            a = copper[i]["name"]
            b = copper[i+1]["name"]
            lines.append(f"  {a} → {b}: {g*1000.0:.1f} µm")
    return "\n".join(lines)

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

class SettingsDialog(wx.Dialog):
    def __init__(self, parent, selected_count, suggested_res, layer_names, preview_callback=None, stackup_details="", pad_names=None, default_output_dir="", defaults=None):
        super().__init__(parent, title="Thermal Sim")
        
        self.layer_names = layer_names
        self.preview_callback = preview_callback
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # --- Info ---
        info_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Stackup")
        l_str = f"{len(layer_names)} Layers found"
        lbl_layers = wx.StaticText(self, label=l_str)
        info_box.Add(lbl_layers, 0, wx.ALL, 5)

        # Detailed stackup (from saved .kicad_pcb stackup), shown in µm
        if stackup_details:
            self.txt_stackup = wx.TextCtrl(
                self,
                value=stackup_details,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
            )
            # Keep the dialog compact but scrollable
            self.txt_stackup.SetMinSize((-1, 120))
            info_box.Add(self.txt_stackup, 0, wx.EXPAND | wx.ALL, 5)

        sizer.Add(info_box, 0, wx.EXPAND|wx.ALL, 5)

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
        sizer.Add(pad_box, 0, wx.EXPAND|wx.ALL, 5)

        # --- Setup ---
        box_main = wx.StaticBoxSizer(wx.VERTICAL, self, "Parameters")
        
        lbl_pwr = wx.StaticText(self, label="Pad Power (W): one value for all, or comma-separated per pad")
        box_main.Add(lbl_pwr, 0, wx.ALL, 5)
        self.power_input = wx.TextCtrl(self, value="1.0")
        box_main.Add(self.power_input, 0, wx.EXPAND|wx.ALL, 5)

        self.time_input = self.add_field(box_main, "Duration (sec):", "20.0")
        self.amb_input  = self.add_field(box_main, "Ambient Temp (°C):", "25.0")
        self.thick_input = self.add_field(box_main, "PCB Thickness (mm):", "1.6")
        
        sizer.Add(box_main, 0, wx.EXPAND|wx.ALL, 5)
        
        # --- Options ---
        box_out = wx.StaticBoxSizer(wx.VERTICAL, self, "Output")
        self.chk_all_layers = wx.CheckBox(self, label="Show All Layers")
        self.chk_all_layers.SetValue(True) 
        box_out.Add(self.chk_all_layers, 0, wx.ALL, 5)
        
        self.chk_snapshots = wx.CheckBox(self, label="Save Snapshots")
        self.chk_snapshots.SetValue(False)
        box_out.Add(self.chk_snapshots, 0, wx.ALL, 5)

        self.snap_count_input = self.add_field(box_out, "Snapshots count:", "5")
        self.snap_count_input.Enable(False)
        self.chk_snapshots.Bind(wx.EVT_CHECKBOX, self.on_snapshots_toggle)
        
        sizer.Add(box_out, 0, wx.EXPAND|wx.ALL, 5)

        # --- Output Folder ---
        box_path = wx.StaticBoxSizer(wx.VERTICAL, self, "Output Folder")
        row_path = wx.BoxSizer(wx.HORIZONTAL)
        self.output_dir_input = wx.TextCtrl(self, value=default_output_dir)
        btn_browse = wx.Button(self, label="Browse...")
        btn_browse.Bind(wx.EVT_BUTTON, self.on_browse_output)
        row_path.Add(self.output_dir_input, 1, wx.EXPAND|wx.RIGHT, 5)
        row_path.Add(btn_browse, 0)
        box_path.Add(row_path, 0, wx.EXPAND|wx.ALL, 5)
        sizer.Add(box_path, 0, wx.EXPAND|wx.ALL, 5)

        # --- Filters ---
        box_filter = wx.StaticBoxSizer(wx.VERTICAL, self, "Geometry Filters")
        self.chk_ignore_traces = wx.CheckBox(self, label="Ignore Traces")
        self.chk_ignore_traces.SetValue(False)
        self.chk_ignore_traces.SetValue(False)
        box_filter.Add(self.chk_ignore_traces, 0, wx.ALL, 5)

        # self.chk_ignore_polygons = wx.CheckBox(self, label="Ignore Polygons Not Attached to Selected Pads")
        # self.chk_ignore_polygons.SetValue(False)
        # box_filter.Add(self.chk_ignore_polygons, 0, wx.ALL, 5)

        self.chk_limit_area = wx.CheckBox(self, label="Limit Area to Pads")
        self.chk_limit_area.SetValue(False)
        box_filter.Add(self.chk_limit_area, 0, wx.ALL, 5)

        self.pad_dist_input = self.add_field(box_filter, "Limit Distance (mm):", "30")
        self.pad_dist_input.Enable(False)
        self.chk_limit_area.Bind(wx.EVT_CHECKBOX, self.on_limit_area_toggle)
        sizer.Add(box_filter, 0, wx.EXPAND|wx.ALL, 5)

        # --- Pad ---
        box_cool = wx.StaticBoxSizer(wx.VERTICAL, self, "Thermal Pad (User.Eco1)")
        self.chk_heatsink = wx.CheckBox(self, label="Enable Pad Simulation")
        self.chk_heatsink.SetValue(False) 
        box_cool.Add(self.chk_heatsink, 0, wx.ALL, 5)
        
        self.pad_thick = self.add_field(box_cool, "Pad Thickness (mm):", "1.0")
        self.pad_k     = self.add_field(box_cool, "Pad Cond. (W/mK):", "3.0")
        self.pad_cap = self.add_field(box_cool, "Pad Heat Cap. (J/m²·K):", "0.0")
        
        sizer.Add(box_cool, 0, wx.EXPAND|wx.ALL, 5)
        
        # --- Grid ---
        box_grid = wx.StaticBoxSizer(wx.VERTICAL, self, "Grid Resolution")
        self.res_input = self.add_field(box_grid, "Resolution (mm):", str(suggested_res))
        sizer.Add(box_grid, 0, wx.EXPAND|wx.ALL, 5)

        # --- Buttons ---
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.btn_preview = wx.Button(self, label="Preview")
        self.btn_preview.Bind(wx.EVT_BUTTON, self.on_preview)
        btn_sizer.Add(self.btn_preview, 0, wx.ALL, 5)
        
        btn_sizer.AddStretchSpacer()
        
        btn_run = wx.Button(self, wx.ID_OK, "Run")
        btn_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        btn_sizer.Add(btn_run, 0, wx.ALL, 5)
        btn_sizer.Add(btn_cancel, 0, wx.ALL, 5)
        
        sizer.Add(btn_sizer, 0, wx.EXPAND|wx.ALL, 10)
        
        self.SetSizer(sizer)
        self.Fit()
        self.Center()

        if defaults:
            self._apply_defaults(defaults)
    
    def on_preview(self, event):
        if self.preview_callback:
            settings = self.get_values()
            if settings:
                self.preview_callback(settings, self.layer_names)

    def add_field(self, sizer, label_text, default_val):
        row = wx.BoxSizer(wx.HORIZONTAL)
        lbl = wx.StaticText(self, label=label_text, size=(160, -1))
        txt = wx.TextCtrl(self, value=default_val)
        row.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5)
        row.Add(txt, 1, wx.EXPAND)
        sizer.Add(row, 0, wx.EXPAND|wx.ALL, 2)
        return txt

    def get_values(self):
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
                # 'ignore_polygons': self.chk_ignore_polygons.GetValue(), # Disabled by request
                'ignore_polygons': False,
                'limit_area': self.chk_limit_area.GetValue(),
                'pad_dist_mm': float(self.pad_dist_input.GetValue()),
                'use_heatsink': self.chk_heatsink.GetValue(),
                'pad_th': float(self.pad_thick.GetValue()),
                'pad_k': float(self.pad_k.GetValue()),
                'pad_cap_areal': float(self.pad_cap.GetValue())
            }
        except ValueError:
            return None

    def on_limit_area_toggle(self, event):
        self.pad_dist_input.Enable(self.chk_limit_area.GetValue())

    def on_snapshots_toggle(self, event):
        self.snap_count_input.Enable(self.chk_snapshots.GetValue())

    def on_browse_output(self, event):
        start_dir = self.output_dir_input.GetValue()
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = os.path.dirname(__file__)
        dlg = wx.DirDialog(self, "Select Output Folder", defaultPath=start_dir, style=wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            self.output_dir_input.SetValue(dlg.GetPath())
        dlg.Destroy()

    def _apply_defaults(self, defaults):
        try:
            self.power_input.SetValue(str(defaults.get('power_str', self.power_input.GetValue())))
            self.time_input.SetValue(str(defaults.get('time', self.time_input.GetValue())))
            self.amb_input.SetValue(str(defaults.get('amb', self.amb_input.GetValue())))
            self.thick_input.SetValue(str(defaults.get('thick', self.thick_input.GetValue())))
            self.res_input.SetValue(str(defaults.get('res', self.res_input.GetValue())))
            self.chk_all_layers.SetValue(bool(defaults.get('show_all', self.chk_all_layers.GetValue())))
            self.chk_snapshots.SetValue(bool(defaults.get('snapshots', self.chk_snapshots.GetValue())))
            self.snap_count_input.SetValue(str(defaults.get('snap_count', self.snap_count_input.GetValue())))
            self.snap_count_input.Enable(self.chk_snapshots.GetValue())
            out_dir = defaults.get('output_dir')
            if out_dir:
                self.output_dir_input.SetValue(str(out_dir))
            self.chk_ignore_traces.SetValue(bool(defaults.get('ignore_traces', self.chk_ignore_traces.GetValue())))
            self.chk_limit_area.SetValue(bool(defaults.get('limit_area', self.chk_limit_area.GetValue())))
            self.pad_dist_input.SetValue(str(defaults.get('pad_dist_mm', self.pad_dist_input.GetValue())))
            self.pad_dist_input.Enable(self.chk_limit_area.GetValue())
            self.chk_heatsink.SetValue(bool(defaults.get('use_heatsink', self.chk_heatsink.GetValue())))
            self.pad_thick.SetValue(str(defaults.get('pad_th', self.pad_thick.GetValue())))
            self.pad_k.SetValue(str(defaults.get('pad_k', self.pad_k.GetValue())))
            self.pad_cap.SetValue(str(defaults.get('pad_cap_areal', self.pad_cap.GetValue())))
        except Exception:
            pass

class ThermalPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "2.5D Thermal Sim"
        self.category = "Simulation"
        self.description = "Crash-safe Multilayer Sim"
        self.show_toolbar_button = True
        self.icon_file_name = "" 
        
        # Store references for preview
        self.board = None
        self.copper_ids = []
        self.bbox = None
        self.pads_list = []

    def _settings_path(self):
        return os.path.join(os.path.dirname(__file__), "thermal_sim_last_settings.json")

    def _load_settings(self):
        try:
            with open(self._settings_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_settings(self, settings):
        try:
            with open(self._settings_path(), "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def Run(self):
        try:
            self.RunSafe()
        except Exception:
            # Show every error so we know what's happening
            wx.MessageBox(traceback.format_exc(), "Thermal Sim Error")

    def RunSafe(self):
        if not HAS_LIBS:
            wx.MessageBox("Please install numpy & matplotlib!", "Error"); return

        board = pcbnew.GetBoard()

        # Keep zone fills up-to-date (required for HitTestFilledArea-based zone mapping in KiCad 9)
        try:
            pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        except Exception:
            pass

        # --- 1. Robust Layer Detection ---
        # Get all enabled copper layers in stackup order
        copper_ids = []
        layer_names = []
        enabled_layers = board.GetEnabledLayers()
        
        for lid in range(64): # Scan all possible layers
            try:
                is_copper = pcbnew.IsCopperLayer(lid)
            except:
                is_copper = (lid < 32) # Standard KiCad copper layer range
                
            if enabled_layers.Contains(lid) and is_copper:
                copper_ids.append(lid)
                layer_names.append(board.GetLayerName(lid))
        
        # Determine physical copper order (top → bottom).
        # NOTE: KiCad PCB_LAYER_ID numeric ordering is NOT the physical stackup order (e.g. B.Cu is often id=2).
        stack_info = parse_stackup_from_board_file(board)
        copper_ids_stack = stack_info.get("copper_ids") if isinstance(stack_info, dict) else None

        if copper_ids_stack and len(copper_ids_stack) >= 2:
            copper_ids = copper_ids_stack
        else:
            # Prefer KiCad's copper ordinal if available (top->bottom)
            try:
                copper_ids = sorted(copper_ids, key=lambda lid: int(pcbnew.CopperLayerToOrdinal(lid)))
            except Exception:
                # Fallback: F.Cu first, B.Cu last, inner layers by "InN.Cu"
                def _copper_key(lid):
                    nm = board.GetLayerName(lid)
                    if nm == "F.Cu": return -1000
                    if nm == "B.Cu": return 1000
                    m = re.match(r"In(\d+)\.Cu", nm)
                    return int(m.group(1)) if m else 0
                copper_ids = sorted(copper_ids, key=_copper_key)

        # Re-map layer names in sorted order
        layer_names = [board.GetLayerName(lid) for lid in copper_ids]
        copper_layer_count = len(copper_ids)

        # --- 2. Auto-Resolution ---
        try: bbox = board.GetBoundingBox()
        except: bbox = board.ComputeBoundingBox(True)
        w_mm = bbox.GetWidth() * 1e-6
        h_mm = bbox.GetHeight() * 1e-6
        
        target_nodes = 25000
        if len(copper_ids) > 2: target_nodes = 15000
            
        area = w_mm * h_mm
        suggested_res = 0.5
        if area > 0:
            suggested_res = round(math.sqrt(area / target_nodes), 2)
            if suggested_res < 0.2: suggested_res = 0.2

        # --- 3. Pad Selection ---
        selected_pads = []
        # Safe Iterator
        try:
            footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
            for fp in footprints:
                for pad in fp.Pads():
                    if pad.IsSelected():
                        name = f"{fp.GetReference()}-{pad.GetNumber()}"
                        selected_pads.append((name, pad))
        except Exception as e:
            wx.MessageBox(f"Error reading pads: {e}", "Error"); return
        
        if not selected_pads:
            wx.MessageBox("Select pads first!", "Info"); return
        selected_pads.sort(key=lambda x: x[0])
        pads_list = [p[1] for p in selected_pads] 

        # --- 4. UI ---
        # Store references for preview
        self.board = board
        self.copper_ids = copper_ids
        self.bbox = bbox
        self.pads_list = pads_list
        
        # Prepare compact GUI report: stackup (µm) + recognized pad names
        try:
            stackup_details = format_stackup_report_um(stack_info)
        except Exception:
            stackup_details = ""

        pad_names = []
        for nm, pad in selected_pads:
            net = ""
            try:
                net = pad.GetNetname()
            except Exception:
                try:
                    net = pad.GetNet().GetNetname()
                except Exception:
                    net = ""
            pad_names.append(f"{nm} [{net}]" if net else nm)

        default_output_dir = os.path.dirname(__file__)
        last_settings = self._load_settings()
        if last_settings.get("output_dir") and os.path.isdir(last_settings.get("output_dir")):
            default_output_dir = last_settings.get("output_dir")

        dlg = SettingsDialog(None, len(pads_list), suggested_res, layer_names,
                             preview_callback=self.generate_preview,
                             stackup_details=stackup_details,
                             pad_names=pad_names,
                             default_output_dir=default_output_dir,
                             defaults=last_settings)
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy(); return
        settings = dlg.get_values()
        dlg.Destroy()
        if not settings: return

        self._save_settings(settings)

        # --- Stackup-driven thicknesses (prefer parsed stackup over defaults) ---
        stackup_derived = self._derive_stackup_thicknesses(board, copper_ids, stack_info, settings)
        total_thick_mm = stackup_derived["total_thick_mm_used"]

        # --- Output Folder Setup ---
        base_output_dir = settings.get('output_dir') or default_output_dir or os.path.dirname(__file__)
        if not os.path.isdir(base_output_dir):
            base_output_dir = os.path.dirname(__file__)
        run_dir = os.path.join(base_output_dir, time.strftime("Thermalsim_%Y%m%d_%H%M%S"))
        try:
            os.makedirs(run_dir, exist_ok=True)
        except Exception:
            run_dir = base_output_dir
        try:
            os.makedirs(run_dir, exist_ok=True)
            test_path = os.path.join(run_dir, ".write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
        except Exception:
            run_dir = tempfile.mkdtemp(prefix="ThermalSim_")

        # --- 5. Grid Setup ---
        res = settings['res']
        if (w_mm/res)*(h_mm/res) > 200000: # Memory protection
            res = math.sqrt(area / 100000)

        x_min = bbox.GetX() * 1e-6
        y_min = bbox.GetY() * 1e-6
        x_max = x_min + w_mm
        y_max = y_min + h_mm

        if settings.get('limit_area') and settings.get('pad_dist_mm', 0.0) > 0:
            radius_mm = settings['pad_dist_mm']
            pad_xs = []
            pad_ys = []
            for pad in pads_list:
                try:
                    pos = pad.GetPosition()
                    pad_xs.append(pos.x * 1e-6)
                    pad_ys.append(pos.y * 1e-6)
                except Exception:
                    continue
            if pad_xs and pad_ys:
                x_min = max(x_min, min(pad_xs) - radius_mm)
                y_min = max(y_min, min(pad_ys) - radius_mm)
                x_max = min(x_max, max(pad_xs) + radius_mm)
                y_max = min(y_max, max(pad_ys) + radius_mm)
                w_mm = max(res, x_max - x_min)
                h_mm = max(res, y_max - y_min)
                area = w_mm * h_mm

        cols = int(w_mm / res) + 4
        rows = int(h_mm / res) + 4
        
        # --- 6. Maps ---
        layer_count = len(copper_ids)
        
        # Physical parameters - relative thermal conductivity
        # FR4: ~0.3 W/mK, Copper: ~390 W/mK => ratio ~1300
        # We use relative values for stability
        k_fr4_rel = 1.0
        k_cu_rel  = 400.0
        via_factor = 390.0 / 0.3
        ref_cu_thick_m = 35e-6

        total_thick = max(0.2, total_thick_mm)
        cu_thick_mm_used = stackup_derived["copper_thickness_mm_used"]
        gap_mm_used = stackup_derived["gap_mm_used"]
        cu_thick_m = [max(1e-9, th * 1e-3) for th in cu_thick_mm_used]
        gap_m = [max(1e-9, g * 1e-3) for g in gap_mm_used]
        k_cu_layers = [k_cu_rel * (th / ref_cu_thick_m) for th in cu_thick_m]
        
        # Create geometry maps
        try:
            K, V_map, H_map = self.create_multilayer_maps(board, copper_ids, rows, cols, x_min, y_min, res, settings, k_fr4_rel, k_cu_layers, via_factor, pads_list)
        except Exception as e:
            wx.MessageBox(f"Error mapping geometry: {e}", "Error"); return

        # --- 7. Time Step Calculation ---
        dx = res * 1e-3  # Grid spacing in meters
        dy = dx
        sim_time = settings['time']

        def _clamp(val, lo, hi):
            return max(lo, min(hi, val))

        steps_target = _clamp(int(120 * (sim_time ** 0.35)), 80, 600)
        steps_target = max(1, steps_target)
        dt = sim_time / steps_target

        # --- 8. Power Injection & Physical Properties ---
        eps = 1e-12
        amb = settings['amb']
        pixel_area = dx * dy

        k_cu = 390.0
        k_fr4 = 0.3
        rho_cu, cp_cu = 8960.0, 385.0
        rho_fr4, cp_fr4 = 1850.0, 1100.0

        fr4_baseline_rel = k_fr4_rel
        copper_threshold_rel = fr4_baseline_rel * 1.5
        copper_mask = K > copper_threshold_rel

        if layer_count > 1 and gap_m:
            t_fr4_eff = []
            for i in range(layer_count):
                if i == 0:
                    gap = gap_m[0]
                elif i == layer_count - 1:
                    gap = gap_m[-1]
                else:
                    gap = 0.5 * (gap_m[i - 1] + gap_m[i])
                t_fr4_eff.append(gap)
        else:
            t_fr4_eff = [max(total_thick_mm * 1e-3, 1e-5)] * layer_count
        t_fr4_eff = np.clip(np.array(t_fr4_eff), 1e-6, 5e-3)
        t_fr4_eff_mm = (t_fr4_eff * 1e3).tolist()
        t_cu = np.array(cu_thick_m)

        dielectric_under_copper_factor = 1.0
        C_layers = np.empty((layer_count, rows, cols), dtype=np.float64)
        for l in range(layer_count):
            V_cu = pixel_area * t_cu[l]
            V_fr4 = pixel_area * t_fr4_eff[l]
            mask = copper_mask[l]
            C_layer = np.where(mask, rho_cu * cp_cu * V_cu, rho_fr4 * cp_fr4 * V_fr4)
            C_layer += mask * (rho_fr4 * cp_fr4 * V_fr4 * dielectric_under_copper_factor)
            C_layers[l] = C_layer
        pad_cap_areal = float(settings.get('pad_cap_areal', 0.0) or 0.0)
        pad_cap_per_cell = 0.0
        pad_cap_total = 0.0
        if pad_cap_areal > 0.0 and np.any(H_map):
            pad_cap_per_cell = pad_cap_areal * pixel_area
            pad_cap_add = pad_cap_per_cell * H_map
            C_layers[-1] += pad_cap_add
            pad_cap_total = float(np.sum(pad_cap_add))
        C = C_layers.reshape(-1)

        try:
            p_parts = [float(x.strip()) for x in settings['power_str'].split(',')]
            p_vals = [p_parts[0]] * len(pads_list) if len(p_parts) == 1 else p_parts
        except:
            return
        pad_power = []
        for idx, pad_name in enumerate(pad_names):
            power_val = p_vals[idx] if idx < len(p_vals) else None
            pad_power.append((pad_name, power_val))

        R = rows
        Cc = cols
        RC = R * Cc
        N = RC * layer_count
        Q = np.zeros(N, dtype=np.float64)

        for idx, pad in enumerate(pads_list):
            pad_lid = pad.GetLayer()
            target_idx = 0  # Default: Top

            if pad_lid in copper_ids:
                target_idx = copper_ids.index(pad_lid)
            else:
                lname = board.GetLayerName(pad_lid).upper()
                if "B." in lname or "BOT" in lname:
                    target_idx = layer_count - 1
                else:
                    target_idx = 0

            if idx >= len(p_vals):
                continue
            pixels = self.get_pad_pixels(pad, rows, cols, x_min, y_min, res)
            if pixels:
                pix = np.array(pixels, dtype=np.int64)
                r = pix[:, 0]
                c = pix[:, 1]
                valid = (r < rows) & (c < cols) & (r >= 0) & (c >= 0)
                r = r[valid]
                c = c[valid]
                if r.size > 0:
                    idxs = target_idx * RC + r * Cc + c
                    q_each = p_vals[idx] / float(r.size)
                    np.add.at(Q, idxs, q_each)

        # --- 9. SOLVER ---
        pd = None
        try:
            rows_list = []
            cols_list = []
            data_list = []

            col_right = np.arange(Cc - 1)[None, :]
            row_all = np.arange(R)[:, None]
            col_all = np.arange(Cc)[None, :]
            row_down = np.arange(R - 1)[:, None]

            for l in range(layer_count):
                base = l * RC
                mask = copper_mask[l]
                k_layer = np.where(mask, k_cu, k_fr4)
                t_layer = np.where(mask, t_cu[l], t_fr4_eff[l])

                k_h = 2.0 * k_layer[:, :-1] * k_layer[:, 1:] / (k_layer[:, :-1] + k_layer[:, 1:] + eps)
                t_edge = 0.5 * (t_layer[:, :-1] + t_layer[:, 1:])
                Gx = k_h * (t_edge * dy) / dx

                idx_left = base + row_all * Cc + col_right
                idx_right = idx_left + 1

                g = Gx.ravel()
                i_idx = idx_left.ravel()
                j_idx = idx_right.ravel()
                rows_list.extend([i_idx, j_idx, i_idx, j_idx])
                cols_list.extend([i_idx, j_idx, j_idx, i_idx])
                data_list.extend([g, g, -g, -g])

                k_h = 2.0 * k_layer[:-1, :] * k_layer[1:, :] / (k_layer[:-1, :] + k_layer[1:, :] + eps)
                t_edge = 0.5 * (t_layer[:-1, :] + t_layer[1:, :])
                Gy = k_h * (t_edge * dx) / dy

                idx_up = base + row_down * Cc + col_all
                idx_down = idx_up + Cc

                g = Gy.ravel()
                i_idx = idx_up.ravel()
                j_idx = idx_down.ravel()
                rows_list.extend([i_idx, j_idx, i_idx, j_idx])
                cols_list.extend([i_idx, j_idx, j_idx, i_idx])
                data_list.extend([g, g, -g, -g])

            if layer_count > 1 and gap_m:
                V_enh = np.clip(V_map, 1.0, 50.0)
                plane_idx = np.arange(RC, dtype=np.int64)
                for l in range(layer_count - 1):
                    gap_val = max(gap_m[l], 1e-6)
                    Gz_base = k_fr4 * pixel_area / gap_val
                    Gz = (Gz_base * V_enh).ravel()
                    i_idx = l * RC + plane_idx
                    j_idx = (l + 1) * RC + plane_idx
                    rows_list.extend([i_idx, j_idx, i_idx, j_idx])
                    cols_list.extend([i_idx, j_idx, j_idx, i_idx])
                    data_list.extend([Gz, Gz, -Gz, -Gz])

            K_base = sp.coo_matrix(
                (np.concatenate(data_list), (np.concatenate(rows_list), np.concatenate(cols_list))),
                shape=(N, N),
                dtype=np.float64
            ).tocsr()

            diag_extra = np.zeros(N, dtype=np.float64)
            b = np.zeros(N, dtype=np.float64)

            h_top = 10.0
            diag_add_top = h_top * pixel_area
            top_idx = np.arange(RC, dtype=np.int64)
            diag_extra[top_idx] += diag_add_top
            b[top_idx] += diag_add_top * amb

            h_air_bot = 10.0
            pad_thick_m = max(0.0001, settings['pad_th'] * 1e-3)
            pad_k = settings['pad_k']
            contact_factor = 0.2
            h_sink = (pad_k / pad_thick_m) * contact_factor
            h_bot_eff = (1.0 - H_map) * h_air_bot + H_map * h_sink
            diag_add_bot = (h_bot_eff * pixel_area).ravel()
            bot_base = (layer_count - 1) * RC
            bot_idx = bot_base + top_idx
            diag_extra[bot_idx] += diag_add_bot
            b[bot_idx] += diag_add_bot * amb
            hA = np.zeros(N, dtype=np.float64)
            hA[top_idx] += diag_add_top
            hA[bot_idx] += diag_add_bot

            K = K_base + sp.diags(diag_extra, format="csr")
            use_pardiso = bool(settings.get("use_pardiso", False)) and HAS_PARDISO
            backend = "PARDISO" if use_pardiso else "SciPy"

            snap_cnt = 1
            step_counter = 0
            snap_times = []
            next_snap_idx = 0
            if settings.get('snapshots'):
                try:
                    snap_count = int(settings.get('snap_count', 5))
                except Exception:
                    snap_count = 5
                snap_count = max(1, min(50, snap_count))
                snap_times = [sim_time * (k / (snap_count + 1)) for k in range(1, snap_count + 1)]
            snap_times = sorted({t for t in snap_times if 0.0 < t < sim_time})
            print(
                f"[ThermalSim] snapshots={settings.get('snapshots')} snap_count={settings.get('snap_count')} "
                f"dt_base={dt:.6f} steps_target={steps_target} snap_times={snap_times}"
            )
            print(f"[ThermalSim] base_output_dir={base_output_dir} run_dir={run_dir}")

            pd = wx.ProgressDialog(
                "Simulating...",
                "Initializing...",
                100,
                style=wx.PD_CAN_ABORT | wx.PD_APP_MODAL | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE
            )
            start_time = time.time()
            aborted = False
            snapshot_stats = []
            snapshot_files = []
            total_solve_time = 0.0
            pin = float(np.sum(Q))
            balance_history = []
            phase_metrics = []
            total_factor_time = 0.0
            factor_count = 0

            try:
                def _progress_update(current, total):
                    percent = int((current / total) * 100) if total else 0
                    msg = f"Step {current}/{total}"
                    keep_going = True
                    try:
                        result = pd.Update(percent, msg)
                        keep_going = result[0] if isinstance(result, tuple) else result
                        if hasattr(pd, "WasCancelled") and pd.WasCancelled():
                            keep_going = False
                    except Exception:
                        keep_going = False
                    return keep_going

                Tn = np.ones(N, dtype=np.float64) * amb
                Tnm1 = Tn.copy()
                e0 = float(np.sum(C * (Tn - amb)))
                balance_integral = 0.0
                pout_step = float(np.sum(hA * (Tn - amb)))
                prev_snap_time = 0.0
                prev_snap_energy = e0

                use_multi_phase = backend == "PARDISO" or bool(settings.get("multi_phase", False))
                if use_multi_phase:
                    phase_defs = [
                        {"name": "A", "frac": 0.08, "dt_scale": 0.5},
                        {"name": "B", "frac": 0.35, "dt_scale": 1.0},
                        {"name": "C", "frac": 0.57, "dt_scale": 2.0},
                    ]
                    phase_times = [sim_time * p["frac"] for p in phase_defs]
                    phase_times[-1] = sim_time - sum(phase_times[:-1])
                    phase_steps = []
                    phase_dts = []
                    for phase_time, pdef in zip(phase_times, phase_defs):
                        phase_dt = dt * pdef["dt_scale"]
                        step_count = max(1, int(round(phase_time / phase_dt)))
                        phase_dt = phase_time / step_count if phase_time > 0 else dt
                        phase_steps.append(step_count)
                        phase_dts.append(phase_dt)
                    total_steps = sum(phase_steps)
                    update_interval = max(1, total_steps // 100)
                    phase_plan = list(zip(phase_defs, phase_times, phase_steps, phase_dts))
                else:
                    total_steps = steps_target
                    update_interval = max(1, total_steps // 100)
                    phase_plan = [({"name": "single"}, sim_time, total_steps, dt)]

                current_time = 0.0

                def _record_snapshot(t_elapsed):
                    nonlocal prev_snap_time, prev_snap_energy
                    T_view = Tn.reshape((layer_count, rows, cols))
                    max_top = float(np.max(T_view[0]))
                    max_bot = float(np.max(T_view[-1]))
                    delta_t = Tn - amb
                    energy = float(np.sum(C * delta_t))
                    dE = energy - e0
                    eps_abs = abs(dE - balance_integral)
                    eps_rel = eps_abs / max(abs(t_elapsed * pin), 1e-9)
                    balance_warn = eps_rel > 0.01 or eps_abs > 0.01
                    print(
                        "[ThermalSim][EnergyBalance] "
                        f"t={t_elapsed:.3f}s E={energy:.6f}J Pin={pin:.6f}W "
                        f"Pout={pout_step:.6f}W eps_abs={eps_abs:.6f}J eps_rel={eps_rel:.6f}"
                    )
                    if balance_warn:
                        print(
                            "[ThermalSim][EnergyBalance][WARN] "
                            f"eps_rel={eps_rel:.6f} eps_abs={eps_abs:.6f}J"
                        )
                    interval_t = t_elapsed - prev_snap_time
                    if interval_t > 0:
                        balance_history.append({
                            "delta_t": interval_t,
                            "dE": energy - prev_snap_energy
                        })
                    prev_snap_time = t_elapsed
                    prev_snap_energy = energy
                    snapshot_stats.append({
                        "t": t_elapsed,
                        "max_top": max_top,
                        "max_bottom": max_bot,
                        "energy": energy,
                        "pin": pin,
                        "pout": pout_step,
                        "eps_abs": eps_abs,
                        "eps_rel": eps_rel,
                        "energy_warn": balance_warn
                    })
                    snap_path = self.save_snapshot(T_view, H_map, amb, layer_names, snap_cnt, t_elapsed, out_dir=run_dir)
                    if snap_path:
                        try:
                            snap_abs = os.path.abspath(snap_path)
                            run_abs = os.path.abspath(run_dir)
                            if not snap_abs.startswith(run_abs):
                                dest = os.path.join(run_dir, os.path.basename(snap_path))
                                shutil.copy2(snap_path, dest)
                                snap_path = dest
                        except Exception:
                            pass
                        snapshot_files.append((t_elapsed, os.path.basename(snap_path)))

                for pdef, phase_time, phase_step_count, phase_dt in phase_plan:
                    if phase_step_count <= 0:
                        continue
                    assembly_start = time.perf_counter()
                    D = C / phase_dt
                    A_be = K + sp.diags(D, format="csr")
                    A_bdf2 = K + sp.diags(1.5 * D, format="csr")
                    assembly_time = time.perf_counter() - assembly_start

                    factor_start = time.perf_counter()
                    if use_pardiso and hasattr(pypardiso, "factorized"):
                        solve_be = pypardiso.factorized(A_be.tocsc())
                        solve_bdf2 = pypardiso.factorized(A_bdf2.tocsc())
                    elif use_pardiso:
                        solve_be = lambda rhs: pypardiso.spsolve(A_be.tocsc(), rhs)
                        solve_bdf2 = lambda rhs: pypardiso.spsolve(A_bdf2.tocsc(), rhs)
                    else:
                        lu_be = spla.splu(A_be.tocsc())
                        lu_bdf2 = spla.splu(A_bdf2.tocsc())
                        solve_be = lu_be.solve
                        solve_bdf2 = lu_bdf2.solve
                    factor_time = time.perf_counter() - factor_start
                    total_factor_time += factor_time
                    factor_count += 1

                    phase_solve_time = 0.0
                    phase_steps_done = 0

                    def _advance_step(rhs, solver, dt_k):
                        nonlocal Tnm1, Tn, current_time, step_counter, pout_step, balance_integral
                        solve_start = time.perf_counter()
                        Tnp1 = solver(rhs)
                        solve_elapsed = time.perf_counter() - solve_start
                        Tnm1, Tn = Tn, Tnp1
                        current_time += dt_k
                        step_counter += 1
                        delta_t = Tn - amb
                        pout_step = float(np.sum(hA * delta_t))
                        balance_integral += dt_k * (pin - pout_step)
                        return solve_elapsed

                    rhs1 = D * Tn + Q + b
                    solve_elapsed = _advance_step(rhs1, solve_be, phase_dt)
                    phase_solve_time += solve_elapsed
                    total_solve_time += solve_elapsed
                    phase_steps_done += 1

                    if step_counter % update_interval == 0 or step_counter == total_steps:
                        if not _progress_update(step_counter, total_steps):
                            aborted = True
                            print("[ThermalSim] Simulation cancelled by user.")
                            break

                    if settings['snapshots']:
                        while next_snap_idx < len(snap_times) and current_time >= snap_times[next_snap_idx]:
                            t_elapsed = snap_times[next_snap_idx]
                            _record_snapshot(t_elapsed)
                            snap_cnt += 1
                            next_snap_idx += 1

                    while phase_steps_done < phase_step_count:
                        rhs = (2.0 * D) * Tn - (0.5 * D) * Tnm1 + Q + b
                        solve_elapsed = _advance_step(rhs, solve_bdf2, phase_dt)
                        phase_solve_time += solve_elapsed
                        total_solve_time += solve_elapsed
                        phase_steps_done += 1

                        if step_counter % update_interval == 0 or step_counter == total_steps:
                            if not _progress_update(step_counter, total_steps):
                                aborted = True
                                print("[ThermalSim] Simulation cancelled by user.")
                                break

                        if settings['snapshots']:
                            while next_snap_idx < len(snap_times) and current_time >= snap_times[next_snap_idx]:
                                t_elapsed = snap_times[next_snap_idx]
                                _record_snapshot(t_elapsed)
                                snap_cnt += 1
                                next_snap_idx += 1
                        if aborted:
                            break

                    phase_avg_solve = phase_solve_time / max(phase_steps_done, 1)
                    phase_metrics.append({
                        "phase": pdef["name"],
                        "dt": phase_dt,
                        "steps": phase_steps_done,
                        "assembly_s": assembly_time,
                        "factorization_s": factor_time,
                        "avg_solve_s": phase_avg_solve
                    })
                    if aborted:
                        break
            finally:
                if pd:
                    if not aborted:
                        try:
                            pd.Update(100, "Done")
                        except Exception:
                            pass
                    pd.Hide()
                    pd.Destroy()
                try:
                    wx.GetApp().Yield()
                except:
                    pass

            avg_solve_time = total_solve_time / max(step_counter, 1)
            T = Tn.reshape((layer_count, rows, cols))
            t_elapsed = sim_time
            delta_t_final = Tn - amb
            pout_final = float(np.sum(hA * delta_t_final))
            steady_ok = False
            rel_diff = None
            if balance_history:
                recent = balance_history[-min(3, len(balance_history)):]
                steady_ok = all(
                    abs(item["dE"]) / max(abs(item["delta_t"] * pin), 1e-9) < 0.01
                    for item in recent
                )
            if steady_ok:
                rel_diff = abs(pin - pout_final) / max(abs(pin), 1e-9)
                print(
                    "[ThermalSim][EnergyBalance][Final] "
                    f"Pin={pin:.6f}W Pout={pout_final:.6f}W rel_diff={rel_diff:.6f}"
                )
            k_norm_info = {
                "strategy": "implicit_fvm_bdf2",
                "backend": backend,
                "multi_phase": use_multi_phase,
                "N": N,
                "nnz_K": int(K.nnz),
                "dt_base": dt,
                "steps_target": steps_target,
                "steps_total": step_counter,
                "copper_threshold_rel": copper_threshold_rel,
                "t_fr4_eff_min": float(np.min(t_fr4_eff)),
                "t_fr4_eff_max": float(np.max(t_fr4_eff)),
                "t_fr4_eff_per_plane_mm": t_fr4_eff_mm,
                "pad_cap_input_areal": pad_cap_areal,
                "pad_cap_per_cell": pad_cap_per_cell,
                "pad_cap_total": pad_cap_total,
                "h_top": h_top,
                "h_air_bottom": h_air_bot,
                "h_sink": h_sink,
                "contact_factor": contact_factor,
                "factorization_s": total_factor_time,
                "factorizations": factor_count,
                "avg_solve_s": avg_solve_time,
                "pin_w": pin,
                "pout_final_w": pout_final,
                "steady_rel_diff": rel_diff
            }
            snapshot_stats_json = json.dumps(snapshot_stats) if snapshot_stats else ""
            snapshot_debug_extra = {
                "solver_backend": backend,
                "solve_steps": step_counter,
                "avg_solve_s": avg_solve_time,
                "factorizations": factor_count,
                "factorization_s": total_factor_time,
                "phase_metrics": json.dumps(phase_metrics),
                "snapshot_stats": snapshot_stats_json
            }
        except Exception:
            nnz_val = "n/a"
            if 'K' in locals() and sp.issparse(K):
                nnz_val = int(K.nnz)
            backend_val = backend if 'backend' in locals() else "n/a"
            N_val = N if 'N' in locals() else "n/a"
            msg = (
                "Solver failed.\n\n"
                f"{traceback.format_exc()}\n"
                f"Params: N={N_val}, nnz={nnz_val}, dt_base={dt}, steps_target={steps_target}, backend={backend_val}"
            )
            wx.MessageBox(msg, "Solver Error")
            return
        if not aborted:
            if settings['show_all']:
                heatmap_path = self.show_results_all_layers(T, H_map, settings['amb'], layer_names, open_file=False, t_elapsed=sim_time, out_dir=run_dir)
            else:
                heatmap_path = self.show_results_top_bot(T, H_map, settings['amb'], open_file=False, t_elapsed=sim_time, out_dir=run_dir)
            preview_path = self._save_preview_image(settings, layer_names, open_file=False, stack_info=stack_info, out_dir=run_dir)
            snapshot_debug = {
                "snapshots_enabled": settings.get('snapshots'),
                "snap_count": settings.get('snap_count'),
                "dt_base": dt,
                "steps_target": steps_target,
                "steps_total": step_counter,
                "snap_times": snap_times,
                "base_output_dir": base_output_dir,
                "run_dir": run_dir,
            }
            snapshot_debug.update(snapshot_debug_extra)
            report_path = self._write_html_report(
                settings=settings,
                stack_info=stack_info,
                stackup_derived=stackup_derived,
                pad_power=pad_power,
                layer_names=layer_names,
                preview_path=preview_path,
                heatmap_path=heatmap_path,
                k_norm_info=k_norm_info,
                out_dir=run_dir,
                snapshot_debug=snapshot_debug,
                snapshot_files=snapshot_files
            )
            snapshot_count = 0
            snapshot_count = len(snapshot_files)
            print(f"[ThermalSim] snapshots_found={snapshot_count}")
            if settings.get('snapshots') and snapshot_count == 0:
                wx.MessageBox("Snapshots enabled but none were created.\nCheck settings and debug info in the report.", "Snapshot Warning")
            if report_path:
                def _open_outputs():
                    try:
                        import webbrowser
                        webbrowser.open("file://" + os.path.abspath(report_path))
                    except Exception:
                        pass
                    if heatmap_path:
                        try:
                            if sys.platform.startswith("win"):
                                os.startfile(os.path.abspath(heatmap_path))
                            elif sys.platform == "darwin":
                                subprocess.Popen(["open", os.path.abspath(heatmap_path)])
                            else:
                                subprocess.Popen(["xdg-open", os.path.abspath(heatmap_path)])
                        except Exception:
                            pass
                wx.CallAfter(_open_outputs)

    def create_multilayer_maps(self, board, copper_ids, rows, cols, x_min, y_min, res, settings, k_fr4, k_cu_layers, via_factor, pads_list):
        num_layers = len(copper_ids)
        K = np.ones((num_layers, rows, cols)) * k_fr4
        V = np.ones((rows, cols))
        H = np.zeros((rows, cols))

        limit_area = settings.get('limit_area', False)
        radius_mm = settings.get('pad_dist_mm', 0.0) if limit_area else 0.0
        area_mask = self.build_pad_distance_mask(pads_list, rows, cols, x_min, y_min, res, radius_mm)

        pad_net_codes = set()
        pad_net_names = set()
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
        
        def fill_box(l_idx, bbox, val):
            x0, y0 = bbox.GetX()*1e-6, bbox.GetY()*1e-6
            w, h   = bbox.GetWidth()*1e-6, bbox.GetHeight()*1e-6
            cs = max(0, int((x0 - x_min)/res))
            rs = max(0, int((y0 - y_min)/res))
            ce = min(cols, int((x0+w - x_min)/res)+1)
            re = min(rows, int((y0+h - y_min)/res)+1)
            # Safe Slice Assignment
            if cs < ce and rs < re:
                if rs < rows and cs < cols:
                    if area_mask is None:
                        K[l_idx, rs:re, cs:ce] = np.maximum(K[l_idx, rs:re, cs:ce], val)
                    else:
                        region_mask = area_mask[rs:re, cs:ce]
                        if np.any(region_mask):
                            K_slice = K[l_idx, rs:re, cs:ce]
                            np.maximum(K_slice, val, out=K_slice, where=region_mask)

        def fill_via(bbox, val):
            x0, y0 = bbox.GetX()*1e-6, bbox.GetY()*1e-6
            w, h   = bbox.GetWidth()*1e-6, bbox.GetHeight()*1e-6
            cs = max(0, int((x0 - x_min)/res))
            rs = max(0, int((y0 - y_min)/res))
            ce = min(cols, int((x0+w - x_min)/res)+1)
            re = min(rows, int((y0+h - y_min)/res)+1)
            if cs < ce and rs < re:
                 if rs < rows and cs < cols:
                    if area_mask is None:
                        V[rs:re, cs:ce] = np.maximum(V[rs:re, cs:ce], val)
                    else:
                        region_mask = area_mask[rs:re, cs:ce]
                        if np.any(region_mask):
                            V_slice = V[rs:re, cs:ce]
                            np.maximum(V_slice, val, out=V_slice, where=region_mask)

        def fill_hs(bbox):
            x0, y0 = bbox.GetX()*1e-6, bbox.GetY()*1e-6
            w, h   = bbox.GetWidth()*1e-6, bbox.GetHeight()*1e-6
            cs = max(0, int((x0 - x_min)/res))
            rs = max(0, int((y0 - y_min)/res))
            ce = min(cols, int((x0+w - x_min)/res)+1)
            re = min(rows, int((y0+h - y_min)/res)+1)
            if cs < ce and rs < re: 
                 if rs < rows and cs < cols:
                    if area_mask is None:
                        H[rs:re, cs:ce] = 1.0
                    else:
                        region_mask = area_mask[rs:re, cs:ce]
                        if np.any(region_mask):
                            H_slice = H[rs:re, cs:ce]
                            H_slice[region_mask] = 1.0

        def fill_zone(l_idx, lid, zone, val):
            # Use *filled* area hit-test so clearance/holes (pads other nets, tracks, keepouts) are respected.
            bbox = zone.GetBoundingBox()
            x0, y0 = bbox.GetX()*1e-6, bbox.GetY()*1e-6
            w, h   = bbox.GetWidth()*1e-6, bbox.GetHeight()*1e-6
            cs = max(0, int((x0 - x_min)/res))
            rs = max(0, int((y0 - y_min)/res))
            ce = min(cols, int((x0+w - x_min)/res)+1)
            re = min(rows, int((y0+h - y_min)/res)+1)
            if cs >= ce or rs >= re:
                return

            # Fallback if KiCad build lacks HitTestFilledArea (older versions)
            has_filled_hit = hasattr(zone, "HitTestFilledArea")

            def to_iu(value_mm):
                try:
                    return pcbnew.FromMM(value_mm)
                except Exception:
                    return int(value_mm * 1e6)

            # Use a tiny margin (in internal units) to avoid edge quantization misses at polygon borders.
            margin_iu = 1

            for r in range(rs, re):
                y = y_min + (r + 0.5) * res
                y_iu = to_iu(y)
                for c in range(cs, ce):
                    x = x_min + (c + 0.5) * res
                    pos = pcbnew.VECTOR2I(to_iu(x), y_iu)
                    try:
                        if area_mask is not None and not area_mask[r, c]:
                            continue
                        hit = False
                        if has_filled_hit:
                            hit = zone.HitTestFilledArea(lid, pos, margin_iu)
                        elif hasattr(zone, "HitTest"):
                            hit = zone.HitTest(pos)
                        if hit:
                            K[l_idx, r, c] = max(K[l_idx, r, c], val)
                    except Exception:
                        continue

        # --- 2. Tracks & Vias ---
        # Map layer IDs to indices for fast lookup
        lid_to_idx = {lid: i for i, lid in enumerate(copper_ids)}
        
        def safe_fill(lid, bbox):
            if lid in lid_to_idx:
                idx = lid_to_idx[lid]
                fill_box(idx, bbox, k_cu_layers[idx])

        try:
            tracks = board.Tracks() if hasattr(board, 'Tracks') else board.GetTracks()
            for t in tracks:
                is_via = "VIA" in str(type(t)).upper()
                if settings.get('ignore_traces') and not is_via:
                    continue
                lid = t.GetLayer()
                safe_fill(lid, t.GetBoundingBox())
                
                # Check if it is a via
                if is_via:
                    fill_via(t.GetBoundingBox(), via_factor)

            footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
            for fp in footprints:
                for pad in fp.Pads():
                    bb = pad.GetBoundingBox()
                    if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                        # PTH pads exist on all copper layers
                        for i in range(num_layers):
                            fill_box(i, bb, k_cu_layers[i])
                        fill_via(bb, via_factor)
                    else:
                        # SMD pads
                        safe_fill(pad.GetLayer(), bb)

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
                # Check all layers the zone might be on (multiselection)
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
                        fill_zone(idx, lid, z, k_cu_layers[idx])
                
                if settings['use_heatsink']:
                    z_ls = z.GetLayerSet()
                    if z_ls.Contains(pcbnew.Eco1_User):
                        fill_hs(z.GetBoundingBox())
            
            if settings['use_heatsink']:
                for d in board.GetDrawings():
                    if d.GetLayer() == pcbnew.Eco1_User: 
                        fill_hs(d.GetBoundingBox())
        except Exception as e:
            pass # Silent fail for single element errors

        return K, V, H

    def build_pad_distance_mask(self, pads_list, rows, cols, x_min, y_min, res, radius_mm):
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

    def get_pad_pixels(self, pad, rows, cols, x_min, y_min, res):
        bb = pad.GetBoundingBox()
        x0, y0 = bb.GetX()*1e-6, bb.GetY()*1e-6
        w, h   = bb.GetWidth()*1e-6, bb.GetHeight()*1e-6
        cs = max(0, int((x0 - x_min)/res))
        rs = max(0, int((y0 - y_min)/res))
        ce = min(cols, int((x0+w - x_min)/res)+1)
        re = min(rows, int((y0+h - y_min)/res)+1)
        pixels = []
        for r in range(rs, re):
            for c in range(cs, ce): pixels.append((r, c))
        return pixels

    def save_snapshot(self, T, H, amb, layer_names, idx, t_elapsed, out_dir=None):
        out_dir = out_dir or os.path.dirname(__file__)
        try:
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.join(out_dir, f"snap_{idx:02d}_t{t_elapsed:.1f}.png")
            self._save_stackup_plot(T, H, amb, layer_names, fname, t_elapsed=t_elapsed)
            return fname
        except Exception:
            tmp = tempfile.gettempdir()
            fname = os.path.join(tmp, f"snap_{idx:02d}_t{t_elapsed:.1f}.png")
            self._save_stackup_plot(T, H, amb, layer_names, fname, t_elapsed=t_elapsed)
            return fname

    def _save_plot(self, T, amb, fname):
        vmax = np.max(T)
        if vmax > amb+200: vmax = amb+200
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(T[0], cmap='inferno', origin='upper', vmin=amb, vmax=vmax)
        ax1.set_title("Top Layer")
        ax1.axis('off')
        ax2.imshow(T[-1], cmap='inferno', origin='upper', vmin=amb, vmax=vmax)
        ax2.set_title("Bottom Layer")
        ax2.axis('off')
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

    def _save_stackup_plot(self, T, H, amb, layer_names, fname, t_elapsed=None):
        vmax = np.max(T)
        if vmax > amb + 250: vmax = amb + 250

        count = len(T)
        if count == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        elif count == 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes = axes.flatten()
        else:
            cols_grid = 2
            rows_grid = math.ceil(count / 2)
            fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(12, 4*rows_grid))
            axes = axes.flatten()

        labels = []
        for i in range(count):
            if i < len(layer_names):
                labels.append(layer_names[i])
            elif i == 0:
                labels.append("Top (F.Cu)")
            elif i == count - 1:
                labels.append("Bottom (B.Cu)")
            else:
                labels.append(f"Inner {i}")

        for i in range(count):
            if i >= len(axes): break
            ax = axes[i]
            name = labels[i]
            max_temp = np.max(T[i])
            if t_elapsed is not None:
                ax.set_title(f"{name} - t = {t_elapsed:.1f} s - Max: {max_temp:.1f}°C")
            else:
                ax.set_title(f"{name} - Max: {max_temp:.1f}°C")
            im = ax.imshow(T[i], cmap='inferno', origin='upper', vmin=amb, vmax=vmax, interpolation='bilinear')
            plt.colorbar(im, ax=ax)
            ax.axis('off')
            if i == count - 1 and np.max(H) > 0:
                ax.contour(H, levels=[0.5], colors='white', linewidths=2, linestyles='--')

        for j in range(count, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    def show_results_top_bot(self, T, H, amb, open_file=True, t_elapsed=None, out_dir=None):
        out_dir = out_dir or os.path.dirname(__file__)
        output_file = os.path.join(out_dir, "thermal_final.png")
        vmax = np.max(T)
        if vmax > amb + 250: vmax = amb + 250
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        time_label = f"t = {t_elapsed:.1f} s - " if t_elapsed is not None else ""
        ax1.set_title(f"TOP Layer ({time_label}Max: {np.max(T[0]):.1f} °C)")
        im1 = ax1.imshow(T[0], cmap='inferno', origin='upper', vmin=amb, vmax=vmax, interpolation='bilinear')
        plt.colorbar(im1, ax=ax1)
        ax2.set_title(f"BOTTOM Layer ({time_label}Max: {np.max(T[-1]):.1f} °C)")
        im2 = ax2.imshow(T[-1], cmap='inferno', origin='upper', vmin=amb, vmax=vmax, interpolation='bilinear')
        plt.colorbar(im2, ax=ax2)
        if np.max(H) > 0:
            ax2.contour(H, levels=[0.5], colors='white', linewidths=2, linestyles='--')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        if open_file:
            if sys.platform == 'win32': os.startfile(output_file)
            else:
                import subprocess
                subprocess.call(['xdg-open', output_file])
        return output_file

    def generate_preview(self, settings, layer_names):
        """Generates a preview image of detected copper and vias"""
        output_file = self._save_preview_image(settings, layer_names, open_file=True)
        if not output_file:
            wx.MessageBox("Board data missing for preview", "Error")

    def show_results_all_layers(self, T, H, amb, layer_names, open_file=True, t_elapsed=None, out_dir=None):
        out_dir = out_dir or os.path.dirname(__file__)
        output_file = os.path.join(out_dir, "thermal_stackup.png")
        vmax = np.max(T)
        if vmax > amb + 250: vmax = amb + 250
        
        # Number of actual layers being simulated
        count = len(T)
        
        # Create appropriate layout based on layer count
        if count == 1:
            # Single layer (unlikely but handle it)
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        elif count == 2:
            # Top + Bottom only - horizontal layout
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes = axes.flatten()
        else:
            # Multiple layers: Top row for Top/Bottom, additional rows for inner layers
            # Layout: 2 columns, rows as needed
            cols_grid = 2
            rows_grid = math.ceil(count / 2)
            fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(12, 4*rows_grid))
            axes = axes.flatten()
        
        # Generate proper layer labels
        labels = []
        for i in range(count):
            if i < len(layer_names):
                labels.append(layer_names[i])
            elif i == 0:
                labels.append("Top (F.Cu)")
            elif i == count - 1:
                labels.append("Bottom (B.Cu)")
            else:
                labels.append(f"Inner {i}")
        
        # Plot each layer
        for i in range(count):
            if i >= len(axes): break
            ax = axes[i]
            name = labels[i]
            max_temp = np.max(T[i])
            time_label = f"t = {t_elapsed:.1f} s - " if t_elapsed is not None else ""
            ax.set_title(f"{name} - {time_label}Max: {max_temp:.1f}°C")
            im = ax.imshow(T[i], cmap='inferno', origin='upper', vmin=amb, vmax=vmax, interpolation='bilinear')
            plt.colorbar(im, ax=ax)
            ax.axis('off')
            
            # Show heatsink contour on bottom layer
            if i == count - 1 and np.max(H) > 0:
                ax.contour(H, levels=[0.5], colors='white', linewidths=2, linestyles='--')

        # Hide any unused axes
        for j in range(count, len(axes)): 
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        if open_file:
            if sys.platform == 'win32':
                os.startfile(output_file)
            else:
                import subprocess
                subprocess.call(['xdg-open', output_file])
        return output_file

    def _derive_stackup_thicknesses(self, board, copper_ids, stack_info, settings):
        stack_copper = stack_info.get("copper", []) if isinstance(stack_info, dict) else []
        stack_gaps = stack_info.get("dielectric_gaps_mm", []) if isinstance(stack_info, dict) else []
        stack_board_thick = stack_info.get("board_thickness_mm") if isinstance(stack_info, dict) else None

        copper_thickness_by_id = {}
        copper_thickness_by_name = {}
        for c in stack_copper:
            th = c.get("thickness_mm")
            if isinstance(th, (int, float)):
                lid = c.get("layer_id")
                name = c.get("name")
                if isinstance(lid, int):
                    copper_thickness_by_id[lid] = th
                if isinstance(name, str):
                    copper_thickness_by_name[name] = th

        copper_thickness_mm_used = []
        for lid in copper_ids:
            th = copper_thickness_by_id.get(lid)
            if th is None:
                lname = board.GetLayerName(lid)
                th = copper_thickness_by_name.get(lname)
            if not isinstance(th, (int, float)) or th <= 0:
                th = 0.035
            copper_thickness_mm_used.append(th)

        total_thick_mm_used = settings['thick']
        if isinstance(stack_board_thick, (int, float)) and stack_board_thick > 0:
            total_thick_mm_used = stack_board_thick

        fallback_gap_mm = total_thick_mm_used / max(1, len(copper_ids) - 1)
        gap_mm_used = []
        use_uniform_gap = False
        if len(stack_gaps) != max(0, len(copper_ids) - 1):
            use_uniform_gap = True
        else:
            for g in stack_gaps:
                if not isinstance(g, (int, float)) or g <= 0:
                    use_uniform_gap = True
                    break
        if use_uniform_gap:
            gap_mm_used = [fallback_gap_mm] * max(0, len(copper_ids) - 1)
        else:
            gap_mm_used = [float(g) for g in stack_gaps]

        return {
            "total_thick_mm_used": total_thick_mm_used,
            "stack_board_thick_mm": stack_board_thick,
            "copper_thickness_mm_used": copper_thickness_mm_used,
            "gap_mm_used": gap_mm_used,
            "gap_fallback_used": use_uniform_gap,
        }

    def _save_preview_image(self, settings, layer_names, open_file=False, stack_info=None, out_dir=None):
        board = self.board
        copper_ids = self.copper_ids
        bbox = self.bbox

        if not board or not bbox:
            return None

        # Keep zone fills up-to-date (required for HitTestFilledArea-based zone mapping in KiCad 9)
        try:
            pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        except Exception:
            pass

        res = settings['res']
        w_mm = bbox.GetWidth() * 1e-6
        h_mm = bbox.GetHeight() * 1e-6
        x_min = bbox.GetX() * 1e-6
        y_min = bbox.GetY() * 1e-6
        cols = int(w_mm / res) + 4
        rows = int(h_mm / res) + 4

        # Physics constants for mapping
        k_fr4_rel = 1.0
        k_cu_rel  = 400.0
        via_factor = 390.0 / 0.3
        ref_cu_thick_m = 35e-6
        layer_count = len(copper_ids)
        if stack_info is None:
            stack_info = parse_stackup_from_board_file(board)
        stackup_derived = self._derive_stackup_thicknesses(board, copper_ids, stack_info, settings)
        total_thick = max(0.2, stackup_derived["total_thick_mm_used"])
        cu_thick_m = [max(1e-9, th * 1e-3) for th in stackup_derived["copper_thickness_mm_used"]]
        k_cu_layers = [k_cu_rel * (th / ref_cu_thick_m) for th in cu_thick_m]

        try:
            K, V_map, H_map = self.create_multilayer_maps(board, copper_ids, rows, cols, x_min, y_min, res, settings, k_fr4_rel, k_cu_layers, via_factor, self.pads_list)
            
            out_dir = out_dir or settings.get('output_dir') or os.path.dirname(__file__)
            if not os.path.isdir(out_dir):
                out_dir = os.path.dirname(__file__)
            output_file = os.path.join(out_dir, "thermal_preview.png")
            count = len(K)
            cols_grid = 2
            rows_grid = math.ceil(count / 2)
            
            fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(12, 4*rows_grid), squeeze=False)
            axes = axes.flatten()

            # Build pad masks per layer
            pad_masks = [np.zeros((rows, cols), dtype=bool) for _ in range(count)]
            pad_labels = []
            label_limit = 10
            for pad in self.pads_list or []:
                pad_lid = pad.GetLayer()
                target_indices = []
                if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                    target_indices = list(range(count))
                elif pad_lid in copper_ids:
                    target_indices = [copper_ids.index(pad_lid)]
                else:
                    lname = board.GetLayerName(pad_lid).upper()
                    target_indices = [count - 1 if ("B." in lname or "BOT" in lname) else 0]

                pixels = self.get_pad_pixels(pad, rows, cols, x_min, y_min, res)
                if pixels:
                    for idx in target_indices:
                        for r, c in pixels:
                            if r < rows and c < cols:
                                pad_masks[idx][r, c] = True
                    if len(pad_labels) < label_limit:
                        try:
                            pos = pad.GetPosition()
                            cx = int((pos.x * 1e-6 - x_min) / res)
                            cy = int((pos.y * 1e-6 - y_min) / res)
                        except Exception:
                            cx, cy = None, None
                        if cx is not None and cy is not None:
                            label = pad.GetNumber() if hasattr(pad, "GetNumber") else ""
                            pad_labels.append((target_indices[0], cx, cy, label))
            
            for i in range(count):
                ax = axes[i]
                name = layer_names[i] if i < len(layer_names) else f"Layer {i}"
                ax.set_title(f"Preview: {name}")
                
                # Show copper as a mask overlay
                copper_mask = K[i] > k_fr4_rel
                ax.imshow(copper_mask, cmap='Greens', origin='upper', interpolation='none', alpha=0.35)

                # Heatsink overlay (board-level)
                if settings.get('use_heatsink'):
                    is_bottom = (i == count - 1) or (name == "B.Cu")
                    if is_bottom:
                        ax.imshow(np.ma.masked_where(H_map <= 0, H_map), cmap='Blues', origin='upper', interpolation='none', alpha=0.45)
                
                # Overlay vias in red
                v_mask = V_map > 1.0
                if np.any(v_mask):
                    ax.imshow(np.ma.masked_where(~v_mask, v_mask), cmap='Reds', origin='upper', alpha=0.5, interpolation='none')

                # Overlay pads (heat sources)
                pad_mask = pad_masks[i]
                if np.any(pad_mask):
                    ax.imshow(np.ma.masked_where(~pad_mask, pad_mask), cmap='autumn', origin='upper', alpha=0.6, interpolation='none')
                    for layer_idx, cx, cy, label in pad_labels:
                        if layer_idx == i:
                            ax.text(cx, cy, str(label), color='black', fontsize=8, ha='center', va='center')
                
                ax.axis('off')

            for j in range(count, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.savefig(output_file, dpi=120)
            plt.close()

            if open_file:
                if sys.platform == 'win32':
                    os.startfile(output_file)
                else:
                    import subprocess
                    subprocess.call(['xdg-open', output_file])
            return output_file
        except Exception:
            if open_file:
                wx.MessageBox(f"Preview error: {traceback.format_exc()}", "Error")
            return None

    def _write_html_report(self, settings, stack_info, stackup_derived, pad_power, layer_names, preview_path, heatmap_path, k_norm_info=None, out_dir=None, snapshot_debug=None, snapshot_files=None):
        out_dir = out_dir or os.path.dirname(__file__)
        report_path = os.path.join(out_dir, "thermal_report.html")

        def _fmt(val, suffix=""):
            if val is None:
                return "n/a"
            if isinstance(val, float):
                return f"{val:.4f}{suffix}"
            return f"{val}{suffix}"

        def _esc(text):
            return html.escape(text if text is not None else "")

        total_thick_mm = stackup_derived.get("total_thick_mm_used")
        board_thick_mm = stackup_derived.get("stack_board_thick_mm")
        copper_thicknesses = stackup_derived.get("copper_thickness_mm_used", [])
        gaps_used = stackup_derived.get("gap_mm_used", [])
        gap_fallback_used = stackup_derived.get("gap_fallback_used", False)

        preview_rel = os.path.basename(preview_path) if preview_path else ""
        heatmap_rel = os.path.basename(heatmap_path) if heatmap_path else ""

        settings_rows = "\n".join(
            f"<tr><td>{_esc(k)}</td><td>{_esc(str(v))}</td></tr>"
            for k, v in settings.items()
        )
        pad_rows = "\n".join(
            f"<tr><td>{_esc(name)}</td><td>{_esc(_fmt(power, ' W'))}</td></tr>"
            for name, power in pad_power
        )
        layer_list = ", ".join(_esc(name) for name in layer_names)
        copper_rows = "\n".join(
            f"<tr><td>{_esc(layer_names[i])}</td><td>{_esc(_fmt(th, ' mm'))}</td></tr>"
            for i, th in enumerate(copper_thicknesses)
        )
        gap_rows = "\n".join(
            f"<tr><td>{_esc(layer_names[i])} → {_esc(layer_names[i + 1])}</td><td>{_esc(_fmt(g, ' mm'))}</td></tr>"
            for i, g in enumerate(gaps_used)
        )
        t_fr4_eff_mm = []
        if isinstance(k_norm_info, dict):
            t_fr4_eff_mm = k_norm_info.get("t_fr4_eff_per_plane_mm") or []
        fr4_eff_rows = "\n".join(
            f"<tr><td>{_esc(layer_names[i])}</td><td>{_esc(_fmt(val, ' mm'))}</td></tr>"
            for i, val in enumerate(t_fr4_eff_mm)
        )
        snapshot_items = []
        if snapshot_files is not None:
            for item in snapshot_files:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    t_val, fname = item
                else:
                    fname = os.path.basename(str(item))
                    m = re.search(r"_t([0-9.]+)", fname)
                    t_val = float(m.group(1)) if m else None
                snapshot_items.append((t_val, os.path.basename(fname)))
        else:
            try:
                import glob
                for path in glob.glob(os.path.join(out_dir, "snap_*.png")):
                    fname = os.path.basename(path)
                    m = re.search(r"_t([0-9.]+)", fname)
                    t_val = float(m.group(1)) if m else None
                    snapshot_items.append((t_val, fname))
            except Exception:
                snapshot_items = []
        snapshot_items.sort(key=lambda x: (x[0] if x[0] is not None else 1e9, x[1]))
        snapshots_html = ""
        for t_val, fname in snapshot_items:
            label = f"t = {t_val:.1f} s" if t_val is not None else fname
            snapshots_html += f"<div><p class='small'>{_esc(label)}</p><img src='{_esc(fname)}' alt='{_esc(label)}'></div>"
        if k_norm_info is None:
            k_norm_info = {}
        if snapshot_debug is None:
            snapshot_debug = {}
        k_norm_rows = "\n".join(
            f"<tr><td>{_esc(str(k))}</td><td>{_esc(_fmt(v))}</td></tr>"
            for k, v in k_norm_info.items()
        )
        snapshot_debug_rows = "\n".join(
            f"<tr><td>{_esc(str(k))}</td><td>{_esc(_fmt(v))}</td></tr>"
            for k, v in snapshot_debug.items()
        )

        html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>KiCad Thermal Sim Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1, h2 {{ margin-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    pre {{ background: #f7f7f7; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }}
    .images {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    .images img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
    .small {{ color: #666; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>KiCad Thermal Sim Report</h1>
  <p class="small">Generated by Thermal Sim plugin.</p>

  <h2>Thicknesses Used in Simulation</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Board thickness (stackup)</td><td>{_esc(_fmt(board_thick_mm, " mm"))}</td></tr>
    <tr><td>Total thickness used</td><td>{_esc(_fmt(total_thick_mm, " mm"))}</td></tr>
    <tr><td>Uniform gap fallback used</td><td>{_esc(str(bool(gap_fallback_used)))}</td></tr>
    <tr><td>Layer names</td><td>{layer_list}</td></tr>
  </table>

  <h2>Copper Thickness per Layer</h2>
  <table>
    <tr><th>Layer</th><th>Thickness</th></tr>
    {copper_rows}
  </table>

  <h2>Dielectric Gap per Interface</h2>
  <table>
    <tr><th>Interface</th><th>Gap</th></tr>
    {gap_rows}
  </table>
  <h2>Effective Dielectric Thickness per Plane</h2>
  <p class="small">t_fr4_eff is per-plane effective dielectric thickness derived by averaging adjacent interface gaps; therefore its max may be lower than the maximum single interface gap.</p>
  <table>
    <tr><th>Plane</th><th>t_fr4_eff</th></tr>
    {fr4_eff_rows if fr4_eff_rows else "<tr><td colspan='2'>n/a</td></tr>"}
  </table>

  <h2>Debug</h2>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    {k_norm_rows}
  </table>

  <h2>Snapshot Debug</h2>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    {snapshot_debug_rows}
  </table>

  <h2>Simulation Settings</h2>
  <table>
    <tr><th>Setting</th><th>Value</th></tr>
    {settings_rows}
  </table>

  <h2>Power per Pad</h2>
  <table>
    <tr><th>Pad</th><th>Power</th></tr>
    {pad_rows}
  </table>

  <h2>Images</h2>
  <div class="images">
    <div>
      <h3>Preview</h3>
      {"<img src='" + _esc(preview_rel) + "' alt='Preview image'>" if preview_rel else "<p class='small'>Preview image not available.</p>"}
    </div>
    <div>
      <h3>Heatmap</h3>
      {"<img src='" + _esc(heatmap_rel) + "' alt='Heatmap image'>" if heatmap_rel else "<p class='small'>Heatmap image not available.</p>"}
    </div>
  </div>

  <h2>Snapshots</h2>
  <div class="images">
    {snapshots_html if snapshots_html else "<p class='small'>No snapshots captured.</p>"}
  </div>
</body>
</html>
"""

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_body)
        except Exception:
            return None
        return report_path
