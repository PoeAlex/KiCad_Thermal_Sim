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

class SettingsDialog(wx.Dialog):
    def __init__(self, parent, selected_count, suggested_res, layer_names):
        super().__init__(parent, title="Thermal Sim (Bulletproof)")
        
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
        btn_sizer = wx.StdDialogButtonSizer()
        btn_sizer.AddButton(wx.Button(self, wx.ID_OK, "Run"))
        btn_sizer.AddButton(wx.Button(self, wx.ID_CANCEL, "Cancel"))
        btn_sizer.Realize()
        sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT|wx.ALL, 10)
        
        self.SetSizer(sizer)
        self.Fit()
        self.Center()

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
                'use_heatsink': self.chk_heatsink.GetValue(),
                'pad_th': float(self.pad_thick.GetValue()),
                'pad_k': float(self.pad_k.GetValue())
            }
        except ValueError:
            return None

class ThermalPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "2.5D Thermal Sim (Bulletproof)"
        self.category = "Simulation"
        self.description = "Crash-safe Multilayer Sim"
        self.show_toolbar_button = True
        self.icon_file_name = "" 

    def Run(self):
        try:
            self.RunSafe()
        except Exception:
            # Zeige JEDEN Fehler an, damit wir wissen was los ist
            wx.MessageBox(traceback.format_exc(), "Thermal Sim CRASH")

    def RunSafe(self):
        if not HAS_LIBS:
            wx.MessageBox("Please install numpy & matplotlib!", "Error"); return

        board = pcbnew.GetBoard()
        
        # --- 1. Robust Layer Detection ---
        # Get actual copper layer count from board stackup
        try:
            copper_layer_count = board.GetCopperLayerCount()
        except:
            copper_layer_count = 2  # Fallback for older API
        
        # Use pcbnew constants for layer IDs (works across KiCad versions)
        # F_Cu = Front Copper (Top), B_Cu = Back Copper (Bottom)
        F_Cu = pcbnew.F_Cu
        B_Cu = pcbnew.B_Cu
        
        copper_ids = []
        layer_names = []
        
        # Always add F.Cu (Top)
        copper_ids.append(F_Cu)
        layer_names.append(board.GetLayerName(F_Cu))
        
        # Add inner copper layers if present (In1_Cu, In2_Cu, etc.)
        inner_count = copper_layer_count - 2
        inner_layer_ids = [
            pcbnew.In1_Cu, pcbnew.In2_Cu, pcbnew.In3_Cu, pcbnew.In4_Cu,
            pcbnew.In5_Cu, pcbnew.In6_Cu, pcbnew.In7_Cu, pcbnew.In8_Cu,
            pcbnew.In9_Cu, pcbnew.In10_Cu, pcbnew.In11_Cu, pcbnew.In12_Cu,
            pcbnew.In13_Cu, pcbnew.In14_Cu, pcbnew.In15_Cu, pcbnew.In16_Cu,
            pcbnew.In17_Cu, pcbnew.In18_Cu, pcbnew.In19_Cu, pcbnew.In20_Cu,
            pcbnew.In21_Cu, pcbnew.In22_Cu, pcbnew.In23_Cu, pcbnew.In24_Cu,
            pcbnew.In25_Cu, pcbnew.In26_Cu, pcbnew.In27_Cu, pcbnew.In28_Cu,
            pcbnew.In29_Cu, pcbnew.In30_Cu
        ]
        for i in range(inner_count):
            if i < len(inner_layer_ids):
                copper_ids.append(inner_layer_ids[i])
                layer_names.append(board.GetLayerName(inner_layer_ids[i]))
        
        # Always add B.Cu (Bottom) if more than 1 layer
        if copper_layer_count >= 2:
            copper_ids.append(B_Cu)
            layer_names.append(board.GetLayerName(B_Cu))

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
        dlg = SettingsDialog(None, len(pads_list), suggested_res, layer_names)
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy(); return
        settings = dlg.get_values()
        dlg.Destroy()
        if not settings: return

        # --- 5. Grid Setup ---
        res = settings['res']
        if (w_mm/res)*(h_mm/res) > 200000: # Speicher Schutz
            res = math.sqrt(area / 100000)
        
        x_min = bbox.GetX() * 1e-6 
        y_min = bbox.GetY() * 1e-6
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
            K, V_map, H_map = self.create_multilayer_maps(board, copper_ids, rows, cols, x_min, y_min, res, settings, k_fr4_rel, k_cu_rel, v_base, v_via)
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
        dt_limit = 0.15 * (dx**2) / alpha_eff
        dt = min(dt_limit, 0.005)  # Cap at 5ms for accuracy
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
            
            # --- CRASH FIX: Sicherstellen dass Layer existiert ---
            if pad_lid in copper_ids:
                target_idx = copper_ids.index(pad_lid)
            else:
                # Pad ist auf Layer, der nicht in copper_ids ist (z.B. "All Layers")
                # Wir legen es einfach auf Top (0) oder Bottom (-1) je nach Name
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
        
        # Diffusion coefficient - scaled for stability
        # CFL condition: coeff < 0.25 for 2D explicit
        max_k = np.max(K)
        diff_factor = 0.12 / max(max_k, 1.0)
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
        
        snap_int = max(1, int(num_batches / 10))
        snap_cnt = 1
        step_counter = 0
        amb = settings['amb']
        
        # Pre-compute smoothing kernel weights
        smooth_weight = 0.1
        
        for b in range(num_batches):
            percent = int((b / num_batches) * 100)
            elapsed = time.time() - start_time
            msg = f"Step {b*batch_size}/{steps}"
            if b > 0:
                rem = int((elapsed / b) * (num_batches - b))
                msg += f" (Est. {rem}s)"
            
            if not pd.Update(percent, msg): aborted = True; break
            
            for _ in range(batch_size):
                step_counter += 1
                
                # Lateral Heat Diffusion (2D Laplacian) - standard 5-point stencil
                L = (roll(T, -1, 2) + roll(T, 1, 2) + 
                     roll(T, -1, 1) + roll(T, 1, 1) - 4*T)
                
                # Vertical Heat Transfer between layers (vectorized)
                v_chg = np.zeros_like(T)
                if layer_count > 1:
                    # Temperature difference between adjacent layers
                    dT_down = T[1:] - T[:-1]
                    # Heat flux with via enhancement
                    z_eff = z_base * V_enhance
                    flux = dT_down * z_eff
                    # Clamp flux to prevent instability
                    flux = np.clip(flux, -50, 50)
                    v_chg[:-1] += flux   # Heat flows into upper layer
                    v_chg[1:] -= flux    # Heat flows out of lower layer
                
                # Update temperature
                T += (L * K_safe) + v_chg + P_map
                
                # Clamp temperature to prevent runaway (physical limit)
                np.clip(T, amb, amb + 500, out=T)
                
                # Apply convective cooling at boundaries
                T[0] -= (T[0] - amb) * cool_air
                
                # Bottom layer cooling
                if layer_count > 1:
                    T[-1] -= (T[-1] - amb) * ((1-H_map)*cool_air + H_map*cool_sink_factor)
                else:
                    T[0] -= (T[0] - amb) * cool_air
                
                # Apply smoothing periodically to dampen checkerboard oscillations
                if step_counter % 50 == 0:
                    # Simple 3x3 averaging filter (weighted)
                    T_smooth = (T + 
                               smooth_weight * (roll(T, -1, 2) + roll(T, 1, 2) + 
                                                roll(T, -1, 1) + roll(T, 1, 1))) / (1 + 4*smooth_weight)
                    T = T_smooth
            
            if settings['snapshots'] and (b % snap_int == 0):
                self.save_snapshot(T, settings['amb'], snap_cnt)
                snap_cnt += 1
            
        pd.Destroy()
        if not aborted:
            if settings['show_all']:
                self.show_results_all_layers(T, H_map, settings['amb'], layer_names)
            else:
                self.show_results_top_bot(T, H_map, settings['amb'])

    def create_multilayer_maps(self, board, copper_ids, rows, cols, x_min, y_min, res, settings, k_fr4, k_cu, v_base, v_via):
        num_layers = len(copper_ids)
        K = np.ones((num_layers, rows, cols)) * k_fr4
        V = np.ones((rows, cols)) * v_base
        H = np.zeros((rows, cols))
        
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
                    K[l_idx, rs:re, cs:ce] = np.maximum(K[l_idx, rs:re, cs:ce], val)

        def fill_via(bbox, val):
            x0, y0 = bbox.GetX()*1e-6, bbox.GetY()*1e-6
            w, h   = bbox.GetWidth()*1e-6, bbox.GetHeight()*1e-6
            cs = max(0, int((x0 - x_min)/res))
            rs = max(0, int((y0 - y_min)/res))
            ce = min(cols, int((x0+w - x_min)/res)+1)
            re = min(rows, int((y0+h - y_min)/res)+1)
            if cs < ce and rs < re:
                 if rs < rows and cs < cols:
                    V[rs:re, cs:ce] = np.maximum(V[rs:re, cs:ce], val)

        def fill_hs(bbox):
            x0, y0 = bbox.GetX()*1e-6, bbox.GetY()*1e-6
            w, h   = bbox.GetWidth()*1e-6, bbox.GetHeight()*1e-6
            cs = max(0, int((x0 - x_min)/res))
            rs = max(0, int((y0 - y_min)/res))
            ce = min(cols, int((x0+w - x_min)/res)+1)
            re = min(rows, int((y0+h - y_min)/res)+1)
            if cs < ce and rs < re: 
                 if rs < rows and cs < cols:
                    H[rs:re, cs:ce] = 1.0

        # Safe Object Iteration
        try:
            tracks = board.Tracks() if hasattr(board, 'Tracks') else board.GetTracks()
            for t in tracks:
                lid = t.GetLayer()
                if lid in copper_ids:
                    fill_box(copper_ids.index(lid), t.GetBoundingBox(), k_cu)
                if "VIA" in str(type(t)).upper(): 
                    fill_via(t.GetBoundingBox(), v_via)

            footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
            for fp in footprints:
                for pad in fp.Pads():
                    bb = pad.GetBoundingBox()
                    if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                        for i in range(num_layers): fill_box(i, bb, k_cu)
                        fill_via(bb, v_via)
                    else:
                        lid = pad.GetLayer()
                        if lid in copper_ids:
                            fill_box(copper_ids.index(lid), bb, k_cu)

            zones = board.Zones() if hasattr(board, 'Zones') else board.GetZones()
            for z in zones:
                if not z.IsFilled(): continue
                lid = z.GetLayer()
                if lid in copper_ids:
                    fill_box(copper_ids.index(lid), z.GetBoundingBox(), k_cu)
                if settings['use_heatsink'] and lid == pcbnew.Eco1_User:
                    fill_hs(z.GetBoundingBox())
            
            if settings['use_heatsink']:
                for d in board.GetDrawings():
                    if d.GetLayer() == pcbnew.Eco1_User: fill_hs(d.GetBoundingBox())
        except Exception:
            pass # Ignore single element errors

        return K, V, H

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

    def save_snapshot(self, T, amb, idx):
        try:
            out_dir = os.path.dirname(__file__)
            fname = os.path.join(out_dir, f"snap_{idx:02d}.png")
            self._save_plot(T, amb, fname)
        except:
            tmp = tempfile.gettempdir()
            fname = os.path.join(tmp, f"thermal_snap_{idx:02d}.png")
            self._save_plot(T, amb, fname)

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