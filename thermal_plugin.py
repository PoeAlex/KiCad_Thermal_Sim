import pcbnew
import os
import sys
import traceback
import math
import time
import tempfile
import re
import html

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
    def __init__(self, parent, selected_count, suggested_res, layer_names, preview_callback=None, stackup_details="", pad_names=None):
        super().__init__(parent, title="Thermal Sim (Bulletproof)")
        
        self.layer_names = layer_names
        self.preview_callback = preview_callback
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # --- Info ---
        info_box = wx.StaticBoxSizer(wx.VERTICAL, self, "Stackup")
        l_str = f"{len(layer_names)} Layers found"
        if len(layer_names) > 0:
            l_str += f" ({layer_names[0]}..{layer_names[-1]})"
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
        
        lbl_pwr = wx.StaticText(self, label=f"Total Power (W) for {selected_count} Pad(s):")
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
        
        sizer.Add(box_out, 0, wx.EXPAND|wx.ALL, 5)

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
                'ignore_traces': self.chk_ignore_traces.GetValue(),
                # 'ignore_polygons': self.chk_ignore_polygons.GetValue(), # Disabled by request
                'ignore_polygons': False,
                'limit_area': self.chk_limit_area.GetValue(),
                'pad_dist_mm': float(self.pad_dist_input.GetValue()),
                'use_heatsink': self.chk_heatsink.GetValue(),
                'pad_th': float(self.pad_thick.GetValue()),
                'pad_k': float(self.pad_k.GetValue())
            }
        except ValueError:
            return None

    def on_limit_area_toggle(self, event):
        self.pad_dist_input.Enable(self.chk_limit_area.GetValue())

class ThermalPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "2.5D Thermal Sim (Bulletproof)"
        self.category = "Simulation"
        self.description = "Crash-safe Multilayer Sim"
        self.show_toolbar_button = True
        self.icon_file_name = "" 
        
        # Store references for preview
        self.board = None
        self.copper_ids = []
        self.bbox = None
        self.pads_list = []

    def Run(self):
        try:
            self.RunSafe()
        except Exception:
            # Show every error so we know what's happening
            wx.MessageBox(traceback.format_exc(), "Thermal Sim CRASH")

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

        dlg = SettingsDialog(None, len(pads_list), suggested_res, layer_names, 
                             preview_callback=self.generate_preview,
                             stackup_details=stackup_details,
                             pad_names=pad_names)
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy(); return
        settings = dlg.get_values()
        dlg.Destroy()
        if not settings: return

        # --- Stackup-driven thicknesses (prefer parsed stackup over defaults) ---
        stackup_derived = self._derive_stackup_thicknesses(board, copper_ids, stack_info, settings)
        total_thick_mm = stackup_derived["total_thick_mm_used"]

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
        
        # Physical layer thicknesses (convert to meters)
        layer_spacing_mm = (total_thick / max(1, layer_count - 1))
        layer_spacing_m = layer_spacing_mm * 1e-3  # Convert mm to meters
        
        # Thermal diffusivity - use copper value for stability
        alpha_eff = 1.1e-4  # Copper thermal diffusivity (m^2/s)
        
        # CFL stability: dt < dx^2 / (4 * alpha) for 2D explicit
        # 0.15 for speed while maintaining stability
        dt_limit = 0.15 * (dx**2) / alpha_eff
        dt = min(dt_limit, 0.010)  # Cap at 10ms
        sim_time = settings['time']
        steps = max(200, int(sim_time / dt))
        dt = sim_time / steps  # Recalculate dt to match exactly

        # --- 8. Power Injection ---
        P_map = np.zeros((layer_count, rows, cols))
        
        # Calculate thermal mass per pixel
        # Copper dominates transient thermal behavior
        # Copper: rho=8900 kg/m³, cp=385 J/kg·K
        rho_cu, cp_cu = 8900, 385
        
        pixel_area = dx * dx
        # Copper contribution per layer
        cu_vol = pixel_area * np.array(cu_thick_m)
        cu_heat_cap = cu_vol * rho_cu * cp_cu  # J/K per pixel of copper

        # For multi-layer, add some FR4 contribution (FR4: rho=1850, cp=1100)
        # Use a thin effective FR4 layer to model partial thermal mass coupling
        fr4_effective_thick = []
        if layer_count > 1 and gap_m:
            for i in range(layer_count):
                if i == 0:
                    gap = gap_m[0]
                elif i == layer_count - 1:
                    gap = gap_m[-1]
                else:
                    gap = 0.5 * (gap_m[i - 1] + gap_m[i])
                fr4_effective_thick.append(min(gap * 0.1, 0.0001))
        else:
            fr4_effective_thick = [min(layer_spacing_m * 0.1, 0.0001)] * layer_count
        fr4_vol = pixel_area * np.array(fr4_effective_thick)
        fr4_heat_cap = fr4_vol * 1850 * 1100  # J/K

        # Total heat capacity per pixel (per layer)
        pixel_heat_cap = cu_heat_cap + fr4_heat_cap

        # Power scale: dT per timestep = P * dt / heat_capacity
        power_scale = dt / pixel_heat_cap
        
        try:
            p_parts = [float(x.strip()) for x in settings['power_str'].split(',')]
            p_vals = [p_parts[0]]*len(pads_list) if len(p_parts)==1 else p_parts
        except: return
        pad_power = []
        for idx, pad_name in enumerate(pad_names):
            power_val = p_vals[idx] if idx < len(p_vals) else None
            pad_power.append((pad_name, power_val))

        for idx, pad in enumerate(pads_list):
            pad_lid = pad.GetLayer()
            target_idx = 0 # Default: Top
            
            # --- CRASH FIX: Ensure layer exists ---
            if pad_lid in copper_ids:
                target_idx = copper_ids.index(pad_lid)
            else:
                # Pad is on layer not in copper_ids (e.g. "All Layers")
                # Default to Top (0) or Bottom (-1) based on name
                lname = board.GetLayerName(pad_lid).upper()
                if "B." in lname or "BOT" in lname: target_idx = layer_count - 1
                else: target_idx = 0
            
            pixels = self.get_pad_pixels(pad, rows, cols, x_min, y_min, res)
            if pixels:
                val = (p_vals[idx] * power_scale[target_idx]) / len(pixels)
                for r, c in pixels: 
                    # Bounds Check
                    if r < rows and c < cols:
                        P_map[target_idx, r, c] += val

        # --- 9. SOLVER ---
        T = np.ones((layer_count, rows, cols)) * settings['amb']
        
        # Convective cooling coefficient
        # h * A * (T - Tamb) = heat loss, h ~ 10 W/m^2.K for natural convection
        h_conv = 10.0  # W/m^2.K
        pixel_area = dx * dx
        cool_air = (h_conv * pixel_area / pixel_heat_cap) * dt
        
        # Thermal pad/heatsink cooling
        pad_thick_m = max(0.0001, settings['pad_th'] * 1e-3)
        pad_k = settings['pad_k']
        # Effective heat transfer through thermal pad
        h_sink = (pad_k / pad_thick_m) * 0.1  # Simplified sink model
        cool_sink_factor = (h_sink * pixel_area / pixel_heat_cap[-1]) * dt

        # Use larger batches for speed - reduce Python loop overhead
        batch_size = 200
        num_batches = max(1, int(steps / batch_size))
        actual_steps = num_batches * batch_size
        
        pd = wx.ProgressDialog("Simulating...", "Initializing...", 100, style=wx.PD_CAN_ABORT|wx.PD_APP_MODAL|wx.PD_REMAINING_TIME)
        start_time = time.time()
        aborted = False
        roll = np.roll
        
        # Diffusion coefficient - scaled for stability (Max 0.25)
        # We aligned dt_limit with 0.15, so we match it here
        max_k = np.max(K)
        diff_factor = 0.15 / max(max_k, 1.0)
        K_safe = K * diff_factor
        
        # Vertical heat transfer coefficient
        # Q = k_fr4 * A * dT / d, where d = layer spacing
        # dT/dt = Q / (m * cp) = k_fr4 * A * dT / (d * m * cp)
        # For pixel: coefficient = k_fr4 * pixel_area / (layer_spacing * pixel_heat_cap) * dt
        k_fr4_thermal = 0.3  # FR4 thermal conductivity W/(m·K)
        
        # Vertical coupling: heat transfer rate through FR4 between layers
        if layer_count > 1 and gap_m:
            cap_pairs = 0.5 * (pixel_heat_cap[:-1] + pixel_heat_cap[1:])
            z_base = (k_fr4_thermal * pixel_area / np.array(gap_m)) * dt / cap_pairs
        else:
            z_base = np.zeros((max(0, layer_count - 1),))
        
        # Via enhancement factor (vias increase vertical conductance)
        # V_map has values: 1 for FR4, via_factor for vias
        # Normalize to get via locations: V_norm = 1 for FR4, higher for vias
        # Clamp via enhancement to prevent instability
        V_enhance = np.clip(V_map, 1.0, 50.0)  # Max 50x enhancement at vias
        z_eff = z_base[:, None, None] * V_enhance

        snap_int = max(1, int(num_batches / 10))
        snap_cnt = 1
        step_counter = 0
        amb = settings['amb']
        
        # Pre-compute smoothing kernel weights
        smooth_weight = 0.1
        
        v_chg = np.zeros_like(T)
        
        # --- OPTIMIZATION: Slicing Views & Buffers ---
        # Pre-allocate slice views to avoid constructing them inside the loop
        # Inner domain (excluding 1 pixel border)
        T_inner = T[:, 1:-1, 1:-1]
        K_inner = K_safe[:, 1:-1, 1:-1]
        P_inner = P_map[:, 1:-1, 1:-1]
        
        # Neighbor slices
        T_up    = T[:, :-2, 1:-1]
        T_down  = T[:, 2:, 1:-1]
        T_left  = T[:, 1:-1, :-2]
        T_right = T[:, 1:-1, 2:]

        # Buffer for vertical transfer
        if layer_count > 1:
            z_eff_inner = z_eff[:, 1:-1, 1:-1]
            H_map_inner = H_map[1:-1, 1:-1]
            dT_layer_buf = np.zeros((layer_count-1, rows-2, cols-2))
            v_chg_inner_buf = np.zeros_like(T_inner)
        else:
            # For single layer, these are not strictly used but defined to be safe
            H_map_inner = H_map[1:-1, 1:-1]
            z_eff_inner = None
            dT_layer_buf = None
            v_chg_inner_buf = None



        try:
            for b in range(num_batches):
                percent = int((b / num_batches) * 100)
                elapsed = time.time() - start_time
                msg = f"Step {b*batch_size}/{steps}"
                
                # Check for cancel
                keep_going = True
                try:
                    keep_going = pd.Update(percent, msg)
                except:
                    # Handle cases where dialog might be dead
                    keep_going = False
                    
                if not keep_going: 
                    aborted = True
                    break
                
                for _ in range(batch_size):
                    step_counter += 1
                    
                    # Lateral Heat Diffusion (2D Laplacian) on inner pixels
                    # L = Neighbors - 4*Center
                    L = (T_up + T_down + T_left + T_right)
                    L -= 4 * T_inner
                    
                    # Vertical Heat Transfer
                    if layer_count > 1:
                        v_chg_inner_buf.fill(0.0)
                        
                        # Gradient T[i+1] - T[i] using buffer
                        np.subtract(T_inner[1:], T_inner[:-1], out=dT_layer_buf)
                        
                        # Flux
                        dT_layer_buf *= z_eff_inner
                        np.clip(dT_layer_buf, -50, 50, out=dT_layer_buf)
                        
                        # Apply flux
                        v_chg_inner_buf[:-1] += dT_layer_buf
                        v_chg_inner_buf[1:]  -= dT_layer_buf
                        
                        # T += L*K + v + P
                        L *= K_inner
                        L += v_chg_inner_buf
                        L += P_inner
                        T_inner += L
                    else:
                        # Single layer
                        L *= K_inner
                        L += P_inner
                        T_inner += L
                    
                    # Clamp temperature
                    np.clip(T_inner, amb, amb + 500, out=T_inner)
                    
                    # Convective cooling (Boundary Conditions)
                    # Top Layer Inner
                    T_inner[0] -= (T_inner[0] - amb) * cool_air[0]
                    
                    # Bottom layer
                    if layer_count > 1:
                        T_inner[-1] -= (T_inner[-1] - amb) * ((1-H_map_inner)*cool_air[-1] + H_map_inner*cool_sink_factor)
                    else:
                        T_inner[0] -= (T_inner[0] - amb) * cool_air[0]
                    
                    # Smoothing
                    if step_counter % 50 == 0:
                        # Smoothing using slicing
                        sL = (T_up + T_down + T_left + T_right)
                        sL *= smooth_weight
                        sL += T_inner
                        sL /= (1 + 4*smooth_weight)
                        T_inner[:] = sL
                
                if settings['snapshots'] and (b % snap_int == 0):
                    self.save_snapshot(T, H_map, settings['amb'], layer_names, snap_cnt)
                    snap_cnt += 1
        finally:
            if pd:
                pd.Hide()
                pd.Destroy()
            # Force event processing to ensure modal state is cleared
            try:
                wx.GetApp().Yield()
            except:
                pass
        if not aborted:
            if settings['show_all']:
                heatmap_path = self.show_results_all_layers(T, H_map, settings['amb'], layer_names)
            else:
                heatmap_path = self.show_results_top_bot(T, H_map, settings['amb'])
            preview_path = self._save_preview_image(settings, layer_names, open_file=False, stack_info=stack_info)
            self._write_html_report(
                settings=settings,
                stack_info=stack_info,
                stackup_derived=stackup_derived,
                pad_power=pad_power,
                layer_names=layer_names,
                preview_path=preview_path,
                heatmap_path=heatmap_path
            )

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

    def save_snapshot(self, T, H, amb, layer_names, idx):
        try:
            out_dir = os.path.dirname(__file__)
            fname = os.path.join(out_dir, f"snap_{idx:02d}.png")
            self._save_stackup_plot(T, H, amb, layer_names, fname)
        except:
            tmp = tempfile.gettempdir()
            fname = os.path.join(tmp, f"thermal_snap_{idx:02d}.png")
            self._save_stackup_plot(T, H, amb, layer_names, fname)

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

    def _save_stackup_plot(self, T, H, amb, layer_names, fname):
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

    def show_results_top_bot(self, T, H, amb, open_file=True):
        output_file = os.path.join(os.path.dirname(__file__), "thermal_final.png")
        vmax = np.max(T)
        if vmax > amb + 250: vmax = amb + 250
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.set_title(f"TOP Layer (Max: {np.max(T[0]):.1f} °C)")
        im1 = ax1.imshow(T[0], cmap='inferno', origin='upper', vmin=amb, vmax=vmax, interpolation='bilinear')
        plt.colorbar(im1, ax=ax1)
        ax2.set_title(f"BOTTOM Layer (Max: {np.max(T[-1]):.1f} °C)")
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

    def show_results_all_layers(self, T, H, amb, layer_names, open_file=True):
        output_file = os.path.join(os.path.dirname(__file__), "thermal_stackup.png")
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
            ax.set_title(f"{name} - Max: {max_temp:.1f}°C")
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

    def _save_preview_image(self, settings, layer_names, open_file=False, stack_info=None):
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

            output_file = os.path.join(os.path.dirname(__file__), "thermal_preview.png")
            count = len(K)
            cols_grid = 2
            rows_grid = math.ceil(count / 2)

            fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(12, 4*rows_grid), squeeze=False)
            axes = axes.flatten()

            for i in range(count):
                ax = axes[i]
                name = layer_names[i] if i < len(layer_names) else f"Layer {i}"
                ax.set_title(f"Preview: {name}")

                # Show copper in green
                k_disp = (K[i] - 1.0) / (400.0 - 1.0)
                ax.imshow(k_disp, cmap='Greens', origin='upper', interpolation='none')

                # Overlay vias in red
                v_mask = V_map > 1.0
                if np.any(v_mask):
                    ax.imshow(np.ma.masked_where(~v_mask, v_mask), cmap='Reds', origin='upper', alpha=0.8, interpolation='none')

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

    def _write_html_report(self, settings, stack_info, stackup_derived, pad_power, layer_names, preview_path, heatmap_path):
        out_dir = os.path.dirname(__file__)
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
</body>
</html>
"""

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_body)
        except Exception:
            return None
        return report_path
