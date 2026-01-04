import pcbnew
import os
import sys
import traceback
import math
import time
import tempfile

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for file output
    import matplotlib.pyplot as plt
    import wx
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

class SettingsDialog(wx.Dialog):
    def __init__(self, parent, selected_count, suggested_res, layer_names, preview_callback=None):
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
        
        sizer.Add(info_box, 0, wx.EXPAND|wx.ALL, 5)
        
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
        enabled_layers = board.GetEnabledLayers()

        def add_unique(target, seen, lid):
            if lid in seen:
                return
            if enabled_layers.Contains(lid) and pcbnew.IsCopperLayer(lid):
                target.append(lid)
                seen.add(lid)

        copper_ids = []
        seen_layers = set()

        stackup = None
        for attr in ("GetLayerStackup", "GetStackup"):
            if hasattr(board, attr):
                try:
                    stackup = getattr(board, attr)()
                except Exception:
                    stackup = None
                if stackup:
                    break

        if stackup:
            for getter in ("GetCopperLayers", "GetLayerSequence", "GetLayers"):
                if not hasattr(stackup, getter):
                    continue
                try:
                    layer_seq = getattr(stackup, getter)()
                except Exception:
                    continue
                for item in layer_seq:
                    lid = None
                    if isinstance(item, int):
                        lid = item
                    elif hasattr(item, "GetLayerId"):
                        lid = item.GetLayerId()
                    elif hasattr(item, "GetLayer"):
                        try:
                            lid = item.GetLayer()
                        except Exception:
                            lid = None
                    if lid is not None:
                        add_unique(copper_ids, seen_layers, lid)
                if copper_ids:
                    break

        copper_count = None
        try:
            copper_count = board.GetCopperLayerCount()
        except Exception:
            copper_count = None

        def expected_copper_order():
            order = []
            if hasattr(pcbnew, "F_Cu"):
                order.append(pcbnew.F_Cu)
            if copper_count and hasattr(pcbnew, "In1_Cu"):
                for i in range(max(0, copper_count - 2)):
                    order.append(pcbnew.In1_Cu + i)
            if hasattr(pcbnew, "B_Cu"):
                order.append(pcbnew.B_Cu)
            return order

        if copper_ids:
            expected_order = expected_copper_order()
            if expected_order and len(copper_ids) < len(expected_order):
                for lid in expected_order:
                    add_unique(copper_ids, seen_layers, lid)

        if not copper_ids:
            for lid in expected_copper_order():
                add_unique(copper_ids, seen_layers, lid)

        if not copper_ids:
            for lid in range(64):  # Scan all possible layers
                try:
                    is_copper = pcbnew.IsCopperLayer(lid)
                except Exception:
                    is_copper = (lid < 32)  # Standard KiCad copper layer range
                if enabled_layers.Contains(lid) and is_copper:
                    add_unique(copper_ids, seen_layers, lid)
            copper_ids.sort()

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
        
        dlg = SettingsDialog(None, len(pads_list), suggested_res, layer_names, 
                             preview_callback=self.generate_preview)
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy(); return
        settings = dlg.get_values()
        dlg.Destroy()
        if not settings: return

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
        
        total_thick = max(0.2, settings['thick'])
        dielectric_thick = total_thick / max(1, (layer_count - 1))
        # Vertical conductance through dielectric vs via
        v_base = 0.3 / dielectric_thick   # FR4 vertical conductance
        v_via  = 390.0 / dielectric_thick  # Via copper conductance
        
        # Create geometry maps
        try:
            K, V_map, H_map = self.create_multilayer_maps(board, copper_ids, rows, cols, x_min, y_min, res, settings, k_fr4_rel, k_cu_rel, v_base, v_via, pads_list)
        except Exception as e:
            wx.MessageBox(f"Error mapping geometry: {e}", "Error"); return

        # --- 7. Time Step Calculation ---
        dx = res * 1e-3  # Grid spacing in meters
        
        # Physical layer thicknesses (convert to meters)
        copper_thick = 35e-6  # 1oz copper = 35 microns
        layer_spacing_m = (total_thick / max(1, layer_count - 1)) * 1e-3  # Convert mm to meters
        
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
        cu_vol = pixel_area * copper_thick
        cu_heat_cap = cu_vol * rho_cu * cp_cu  # J/K per pixel of copper
        
        # For multi-layer, add some FR4 contribution (FR4: rho=1850, cp=1100)
        # Use a thin effective FR4 layer to model partial thermal mass coupling
        fr4_effective_thick = min(layer_spacing_m * 0.1, 0.0001)  # 10% of spacing, max 0.1mm
        fr4_vol = pixel_area * fr4_effective_thick
        fr4_heat_cap = fr4_vol * 1850 * 1100  # J/K
        
        # Total heat capacity per pixel
        pixel_heat_cap = cu_heat_cap + fr4_heat_cap
        
        # Power scale: dT per timestep = P * dt / heat_capacity
        power_scale = dt / pixel_heat_cap
        
        try:
            p_parts = [float(x.strip()) for x in settings['power_str'].split(',')]
            p_vals = [p_parts[0]]*len(pads_list) if len(p_parts)==1 else p_parts
        except: return

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
                val = (p_vals[idx] * power_scale) / len(pixels)
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
        cool_sink_factor = (h_sink * pixel_area / pixel_heat_cap) * dt

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
        if layer_count > 1 and layer_spacing_m > 0:
            z_base = (k_fr4_thermal * pixel_area / layer_spacing_m) * dt / pixel_heat_cap
        else:
            z_base = 0.0
        
        # Via enhancement factor (vias increase vertical conductance)
        # V_map has values: v_base for FR4, v_via for vias
        # Normalize to get via locations: V_norm = 1 for FR4, higher for vias
        V_norm = V_map / v_base  # Will be 1 for FR4, ~1300 for vias
        # Clamp via enhancement to prevent instability
        V_enhance = np.clip(V_norm, 1.0, 50.0)  # Max 50x enhancement at vias
        z_eff = z_base * V_enhance

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
            z_eff_inner = z_eff[1:-1, 1:-1]
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
                    T_inner[0] -= (T_inner[0] - amb) * cool_air
                    
                    # Bottom layer
                    if layer_count > 1:
                        T_inner[-1] -= (T_inner[-1] - amb) * ((1-H_map_inner)*cool_air + H_map_inner*cool_sink_factor)
                    else:
                        T_inner[0] -= (T_inner[0] - amb) * cool_air
                    
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
                self.show_results_all_layers(T, H_map, settings['amb'], layer_names)
            else:
                self.show_results_top_bot(T, H_map, settings['amb'])

    def create_multilayer_maps(self, board, copper_ids, rows, cols, x_min, y_min, res, settings, k_fr4, k_cu, v_base, v_via, pads_list):
        num_layers = len(copper_ids)
        K = np.ones((num_layers, rows, cols)) * k_fr4
        V = np.ones((rows, cols)) * v_base
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
        
        def safe_fill(lid, bbox, val):
            if lid in lid_to_idx:
                fill_box(lid_to_idx[lid], bbox, val)

        try:
            tracks = board.Tracks() if hasattr(board, 'Tracks') else board.GetTracks()
            for t in tracks:
                is_via = "VIA" in str(type(t)).upper()
                if settings.get('ignore_traces') and not is_via:
                    continue
                lid = t.GetLayer()
                safe_fill(lid, t.GetBoundingBox(), k_cu)
                
                # Check if it is a via
                if is_via:
                    fill_via(t.GetBoundingBox(), v_via)

            footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
            for fp in footprints:
                for pad in fp.Pads():
                    bb = pad.GetBoundingBox()
                    if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                        # PTH pads exist on all copper layers
                        for i in range(num_layers): fill_box(i, bb, k_cu)
                        fill_via(bb, v_via)
                    else:
                        # SMD pads
                        safe_fill(pad.GetLayer(), bb, k_cu)

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
                        fill_zone(lid_to_idx[lid], lid, z, k_cu)
                
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

    def show_results_top_bot(self, T, H, amb):
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
        if sys.platform == 'win32': os.startfile(output_file)
        else: 
            import subprocess
            subprocess.call(['xdg-open', output_file])

    def generate_preview(self, settings, layer_names):
        """Generates a preview image of detected copper and vias"""
        board = self.board
        copper_ids = self.copper_ids
        bbox = self.bbox
        
        if not board or not bbox:
            wx.MessageBox("Board data missing for preview", "Error")
            return


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
        layer_count = len(copper_ids)
        total_thick = max(0.2, settings['thick'])
        dielectric_thick = total_thick / max(1, (layer_count - 1))
        v_base = 0.3 / dielectric_thick
        v_via  = 390.0 / dielectric_thick

        try:
            K, V_map, H_map = self.create_multilayer_maps(board, copper_ids, rows, cols, x_min, y_min, res, settings, k_fr4_rel, k_cu_rel, v_base, v_via, self.pads_list)
            
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
                v_mask = V_map > v_base
                if np.any(v_mask):
                    ax.imshow(np.ma.masked_where(~v_mask, v_mask), cmap='Reds', origin='upper', alpha=0.8, interpolation='none')
                
                ax.axis('off')

            for j in range(count, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.savefig(output_file, dpi=120)
            plt.close()
            
            if sys.platform == 'win32': os.startfile(output_file)
            else: 
                import subprocess
                subprocess.call(['xdg-open', output_file])
                
        except Exception as e:
            wx.MessageBox(f"Preview error: {traceback.format_exc()}", "Error")

    def show_results_all_layers(self, T, H, amb, layer_names):
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
        
        if sys.platform == 'win32': 
            os.startfile(output_file)
        else: 
            import subprocess
            subprocess.call(['xdg-open', output_file])
