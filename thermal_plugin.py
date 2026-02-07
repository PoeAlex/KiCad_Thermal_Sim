"""
ThermalSim - KiCad PCB thermal simulation plugin.

This is the main controller module that orchestrates the thermal simulation
workflow using the specialized sub-modules.
"""

import os
import sys
import re
import time
import math
import json
import shutil
import tempfile
import subprocess
import traceback

import pcbnew
import numpy as np
import wx

from .capabilities import HAS_LIBS, HAS_PARDISO
from .stackup_parser import parse_stackup_from_board_file, format_stackup_report_um
from .gui_dialogs import SettingsDialog
from .geometry_mapper import create_multilayer_maps, build_pad_distance_mask, get_pad_pixels
from .thermal_solver import SolverConfig, build_stiffness_matrix, run_simulation
from .pwl_parser import parse_pwl_file
from .visualization import (
    save_snapshot, show_results_top_bot, show_results_all_layers, save_preview_image
)
from .thermal_report import write_html_report


class ThermalPlugin(pcbnew.ActionPlugin):
    """
    KiCad Action Plugin for 2.5D transient thermal simulation.

    This plugin simulates heat spreading across multilayer PCBs using
    finite volume methods with BDF2 time integration.
    """

    def defaults(self):
        """Set plugin metadata and initialize state."""
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
        """Return path to settings persistence file."""
        return os.path.join(os.path.dirname(__file__), "thermal_sim_last_settings.json")

    def _load_settings(self):
        """Load settings from JSON file."""
        try:
            with open(self._settings_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_settings(self, settings):
        """Save settings to JSON file."""
        try:
            with open(self._settings_path(), "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def Run(self):
        """Plugin entry point with error handling."""
        try:
            self.RunSafe()
        except Exception:
            wx.MessageBox(traceback.format_exc(), "Thermal Sim Error")

    def RunSafe(self):
        """Main plugin execution logic."""
        if not HAS_LIBS:
            wx.MessageBox("Please install numpy & matplotlib!", "Error")
            return

        board = pcbnew.GetBoard()

        # Keep zone fills up-to-date
        try:
            pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        except Exception:
            pass

        # --- 1. Layer Detection ---
        copper_ids, layer_names = self._detect_copper_layers(board)
        stack_info = parse_stackup_from_board_file(board)

        # Use stackup order if available
        copper_ids_stack = stack_info.get("copper_ids") if isinstance(stack_info, dict) else None
        if copper_ids_stack and len(copper_ids_stack) >= 2:
            copper_ids = copper_ids_stack
            layer_names = [board.GetLayerName(lid) for lid in copper_ids]

        # --- 2. Auto-Resolution ---
        try:
            bbox = board.GetBoundingBox()
        except:
            bbox = board.ComputeBoundingBox(True)
        w_mm = bbox.GetWidth() * 1e-6
        h_mm = bbox.GetHeight() * 1e-6
        suggested_res = self._calculate_suggested_resolution(w_mm, h_mm, len(copper_ids))

        # --- 3. Pad Selection ---
        selected_pads = self._get_selected_pads(board)
        if not selected_pads:
            wx.MessageBox("Select pads first!", "Info")
            return
        pads_list = [p[1] for p in selected_pads]

        # Store for preview
        self.board = board
        self.copper_ids = copper_ids
        self.bbox = bbox
        self.pads_list = pads_list

        # --- 4. Show Dialog ---
        stackup_details = format_stackup_report_um(stack_info) if stack_info else ""
        pad_names = self._format_pad_names(selected_pads)
        default_output_dir = os.path.dirname(__file__)
        last_settings = self._load_settings()
        if last_settings.get("output_dir") and os.path.isdir(last_settings.get("output_dir")):
            default_output_dir = last_settings.get("output_dir")

        dlg = SettingsDialog(
            None, len(pads_list), suggested_res, layer_names,
            preview_callback=self.generate_preview,
            stackup_details=stackup_details,
            pad_names=pad_names,
            default_output_dir=default_output_dir,
            defaults=last_settings
        )
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        settings = dlg.get_values()
        dlg.Destroy()
        if not settings:
            return

        self._save_settings(settings)

        # --- 5. Run Simulation ---
        self._run_simulation(
            board, copper_ids, layer_names, bbox, pads_list,
            settings, stack_info, pad_names
        )

    def _detect_copper_layers(self, board):
        """Detect enabled copper layers in stackup order."""
        copper_ids = []
        layer_names = []
        enabled_layers = board.GetEnabledLayers()

        for lid in range(64):
            try:
                is_copper = pcbnew.IsCopperLayer(lid)
            except:
                is_copper = (lid < 32)

            if enabled_layers.Contains(lid) and is_copper:
                copper_ids.append(lid)
                layer_names.append(board.GetLayerName(lid))

        # Sort by copper ordinal
        try:
            copper_ids = sorted(copper_ids, key=lambda lid: int(pcbnew.CopperLayerToOrdinal(lid)))
        except Exception:
            def _copper_key(lid):
                nm = board.GetLayerName(lid)
                if nm == "F.Cu":
                    return -1000
                if nm == "B.Cu":
                    return 1000
                m = re.match(r"In(\d+)\.Cu", nm)
                return int(m.group(1)) if m else 0
            copper_ids = sorted(copper_ids, key=_copper_key)

        layer_names = [board.GetLayerName(lid) for lid in copper_ids]
        return copper_ids, layer_names

    def _calculate_suggested_resolution(self, w_mm, h_mm, layer_count):
        """Calculate suggested grid resolution based on board size."""
        target_nodes = 25000 if layer_count <= 2 else 15000
        area = w_mm * h_mm
        if area > 0:
            suggested_res = round(math.sqrt(area / target_nodes), 2)
            if suggested_res < 0.2:
                suggested_res = 0.2
        else:
            suggested_res = 0.5
        return suggested_res

    def _get_selected_pads(self, board):
        """Get list of selected pads."""
        selected_pads = []
        try:
            footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
            for fp in footprints:
                for pad in fp.Pads():
                    if pad.IsSelected():
                        name = f"{fp.GetReference()}-{pad.GetNumber()}"
                        selected_pads.append((name, pad))
        except Exception as e:
            wx.MessageBox(f"Error reading pads: {e}", "Error")
            return []
        selected_pads.sort(key=lambda x: x[0])
        return selected_pads

    def _format_pad_names(self, selected_pads):
        """Format pad names with net info for display."""
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
        return pad_names

    def _derive_stackup_thicknesses(self, board, copper_ids, stack_info, settings):
        """Derive thickness values from stackup or defaults."""
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

    def generate_preview(self, settings, layer_names):
        """Generate geometry preview image."""
        output_file = save_preview_image(
            self.board, self.copper_ids, self.bbox, self.pads_list,
            settings, layer_names,
            parse_stackup_from_board_file(self.board),
            get_pad_pixels,
            create_multilayer_maps,
            self._derive_stackup_thicknesses,
            open_file=True
        )
        if not output_file:
            wx.MessageBox("Board data missing for preview", "Error")

    def _run_simulation(self, board, copper_ids, layer_names, bbox, pads_list,
                        settings, stack_info, pad_names):
        """Execute the thermal simulation."""
        # Derive thicknesses
        stackup_derived = self._derive_stackup_thicknesses(board, copper_ids, stack_info, settings)
        total_thick_mm = stackup_derived["total_thick_mm_used"]

        # Output folder setup
        base_output_dir = settings.get('output_dir') or os.path.dirname(__file__)
        if not os.path.isdir(base_output_dir):
            base_output_dir = os.path.dirname(__file__)
        run_dir = os.path.join(base_output_dir, time.strftime("Thermalsim_%Y%m%d_%H%M%S"))
        try:
            os.makedirs(run_dir, exist_ok=True)
            test_path = os.path.join(run_dir, ".write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
        except Exception:
            run_dir = tempfile.mkdtemp(prefix="ThermalSim_")

        # Grid setup
        w_mm = bbox.GetWidth() * 1e-6
        h_mm = bbox.GetHeight() * 1e-6
        x_min = bbox.GetX() * 1e-6
        y_min = bbox.GetY() * 1e-6
        res = settings['res']
        area = w_mm * h_mm
        if (w_mm / res) * (h_mm / res) > 200000:
            res = math.sqrt(area / 100000)

        # Apply area limiting if enabled
        if settings.get('limit_area') and settings.get('pad_dist_mm', 0.0) > 0:
            radius_mm = settings['pad_dist_mm']
            pad_xs = [pad.GetPosition().x * 1e-6 for pad in pads_list]
            pad_ys = [pad.GetPosition().y * 1e-6 for pad in pads_list]
            if pad_xs and pad_ys:
                x_min = max(x_min, min(pad_xs) - radius_mm)
                y_min = max(y_min, min(pad_ys) - radius_mm)
                x_max = min(x_min + w_mm, max(pad_xs) + radius_mm)
                y_max = min(y_min + h_mm, max(pad_ys) + radius_mm)
                w_mm = max(res, x_max - x_min)
                h_mm = max(res, y_max - y_min)

        cols = int(w_mm / res) + 4
        rows = int(h_mm / res) + 4
        layer_count = len(copper_ids)

        # Physical parameters
        k_fr4_rel = 1.0
        k_cu_rel = 400.0
        via_factor = 390.0 / 0.3
        ref_cu_thick_m = 35e-6
        k_cu = 390.0
        k_fr4 = 0.3
        rho_cu, cp_cu = 8960.0, 385.0
        rho_fr4, cp_fr4 = 1850.0, 1100.0

        cu_thick_mm_used = stackup_derived["copper_thickness_mm_used"]
        gap_mm_used = stackup_derived["gap_mm_used"]
        cu_thick_m = [max(1e-9, th * 1e-3) for th in cu_thick_mm_used]
        gap_m = [max(1e-9, g * 1e-3) for g in gap_mm_used]
        k_cu_layers = [k_cu_rel * (th / ref_cu_thick_m) for th in cu_thick_m]

        # Create geometry maps
        try:
            K, V_map, H_map = create_multilayer_maps(
                board, copper_ids, rows, cols, x_min, y_min, res,
                settings, k_fr4_rel, k_cu_layers, via_factor, pads_list
            )
        except Exception as e:
            wx.MessageBox(f"Error mapping geometry: {e}", "Error")
            return

        # Time step calculation
        dx = res * 1e-3
        dy = dx
        sim_time = settings['time']
        steps_target = max(1, min(600, max(80, int(120 * (sim_time ** 0.35)))))
        dt = sim_time / steps_target

        # Build capacity array
        amb = settings['amb']
        pixel_area = dx * dy
        copper_threshold_rel = k_fr4_rel * 1.5
        copper_mask = K > copper_threshold_rel
        t_cu = np.array(cu_thick_m)

        # Effective FR4 thickness per layer
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

        # Heat capacity
        C_layers = np.empty((layer_count, rows, cols), dtype=np.float64)
        for l in range(layer_count):
            V_cu = pixel_area * t_cu[l]
            V_fr4 = pixel_area * t_fr4_eff[l]
            mask = copper_mask[l]
            C_layer = np.where(mask, rho_cu * cp_cu * V_cu, rho_fr4 * cp_fr4 * V_fr4)
            C_layer += mask * (rho_fr4 * cp_fr4 * V_fr4)
            C_layers[l] = C_layer
        pad_cap_areal = float(settings.get('pad_cap_areal', 0.0) or 0.0)
        if pad_cap_areal > 0.0 and np.any(H_map):
            pad_cap_per_cell = pad_cap_areal * pixel_area
            C_layers[-1] += pad_cap_per_cell * H_map
        C = C_layers.reshape(-1)

        # Power injection (supports constant values and PWL file paths)
        RC = rows * cols
        N = RC * layer_count

        entries = [x.strip() for x in settings['power_str'].split(',')]
        if len(entries) == 1:
            entries = entries * len(pads_list)

        # Parse each entry as constant float or PWL file path
        pad_sources = []  # ('const', float) or ('pwl', (times, powers))
        for entry in entries:
            try:
                pad_sources.append(('const', float(entry)))
            except ValueError:
                try:
                    times_pwl, powers_pwl = parse_pwl_file(entry)
                    pad_sources.append(('pwl', (times_pwl, powers_pwl)))
                except (FileNotFoundError, ValueError) as e:
                    wx.MessageBox(
                        f"Error reading PWL file:\n{entry}\n\n{e}",
                        "PWL Error"
                    )
                    return

        if len(pad_sources) != len(pads_list) and len(pad_sources) != 1:
            wx.MessageBox(
                f"Number of power entries ({len(pad_sources)}) does not match "
                f"number of pads ({len(pads_list)}).",
                "Warning"
            )

        # Build per-pad unit Q vectors (spatial distribution at 1W)
        Q_units = []
        for idx, pad in enumerate(pads_list):
            Q_pad = np.zeros(N, dtype=np.float64)
            pad_lid = pad.GetLayer()
            target_idx = 0
            if pad_lid in copper_ids:
                target_idx = copper_ids.index(pad_lid)
            else:
                lname = board.GetLayerName(pad_lid).upper()
                target_idx = layer_count - 1 if ("B." in lname or "BOT" in lname) else 0
            pixels = get_pad_pixels(pad, rows, cols, x_min, y_min, res)
            if pixels:
                pix = np.array(pixels, dtype=np.int64)
                r, c = pix[:, 0], pix[:, 1]
                valid = (r < rows) & (c < cols) & (r >= 0) & (c >= 0)
                r, c = r[valid], c[valid]
                if r.size > 0:
                    idxs = target_idx * RC + r * cols + c
                    np.add.at(Q_pad, idxs, 1.0 / float(r.size))
            Q_units.append(Q_pad)

        # Build constant Q and detect PWL usage
        has_pwl = any(s[0] == 'pwl' for s in pad_sources)

        Q = np.zeros(N, dtype=np.float64)
        for i, (stype, sval) in enumerate(pad_sources):
            if i >= len(Q_units):
                break
            if stype == 'const':
                Q += sval * Q_units[i]
            else:
                Q += float(np.interp(0.0, sval[0], sval[1])) * Q_units[i]

        Q_func = None
        if has_pwl:
            def Q_func(t, _sources=pad_sources, _units=Q_units, _N=N):
                Q_t = np.zeros(_N, dtype=np.float64)
                for i, (stype, sval) in enumerate(_sources):
                    if i >= len(_units):
                        break
                    if stype == 'const':
                        Q_t += sval * _units[i]
                    else:
                        Q_t += float(np.interp(t, sval[0], sval[1])) * _units[i]
                return Q_t

        # Build pad_power for reporting
        pad_power = []
        for i, name in enumerate(pad_names):
            if i < len(pad_sources):
                stype, sval = pad_sources[i]
                if stype == 'const':
                    pad_power.append((name, sval))
                else:
                    pad_power.append((name, f"PWL:{entries[i]}"))
            else:
                pad_power.append((name, None))

        # Build stiffness matrix
        K_matrix, b, hA, _ = build_stiffness_matrix(
            layer_count, rows, cols, copper_mask, t_cu, t_fr4_eff,
            k_cu, k_fr4, dx, dy, V_map, gap_m, H_map, settings, amb
        )

        # Snapshot configuration
        snap_times = []
        if settings.get('snapshots'):
            snap_count = max(1, min(50, int(settings.get('snap_count', 5))))
            snap_times = [sim_time * (k / (snap_count + 1)) for k in range(1, snap_count + 1)]
        snap_times = sorted({t for t in snap_times if 0.0 < t < sim_time})

        # Progress dialog
        pd = wx.ProgressDialog(
            "Simulating...", "Initializing...", 100,
            style=wx.PD_CAN_ABORT | wx.PD_APP_MODAL | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE
        )

        def progress_callback(current, total):
            percent = int((current / total) * 100) if total else 0
            try:
                result = pd.Update(percent, f"Step {current}/{total}")
                keep_going = result[0] if isinstance(result, tuple) else result
                if hasattr(pd, "WasCancelled") and pd.WasCancelled():
                    keep_going = False
                return keep_going
            except Exception:
                return False

        def snapshot_callback(T_view, t_elapsed, snap_idx):
            return save_snapshot(T_view, H_map, amb, layer_names, snap_idx, t_elapsed, out_dir=run_dir)

        # Run solver
        config = SolverConfig(
            sim_time=sim_time,
            amb=amb,
            dt_base=dt,
            steps_target=steps_target,
            use_pardiso=HAS_PARDISO,
            use_multi_phase=True,
            snapshots_enabled=settings.get('snapshots', False),
            snap_times=snap_times
        )

        try:
            result = run_simulation(
                config, K_matrix, C, Q, b, hA,
                layer_count, rows, cols,
                progress_callback, snapshot_callback,
                Q_func=Q_func
            )
        except Exception:
            pd.Destroy()
            wx.MessageBox(f"Solver failed:\n{traceback.format_exc()}", "Solver Error")
            return
        finally:
            try:
                pd.Update(100, "Done")
            except:
                pass
            pd.Hide()
            pd.Destroy()
            try:
                wx.GetApp().Yield()
            except:
                pass

        if result.aborted:
            return

        # Add extra info to k_norm_info
        result.k_norm_info.update({
            "copper_threshold_rel": copper_threshold_rel,
            "t_fr4_eff_min": float(np.min(t_fr4_eff)),
            "t_fr4_eff_max": float(np.max(t_fr4_eff)),
            "t_fr4_eff_per_plane_mm": t_fr4_eff_mm,
            "pad_cap_input_areal": pad_cap_areal,
            "h_top": float(settings.get('h_conv', 10.0)),
            "h_air_bottom": float(settings.get('h_conv', 10.0)),
        })

        # Save results
        if settings['show_all']:
            heatmap_path = show_results_all_layers(
                result.T, H_map, amb, layer_names,
                open_file=False, t_elapsed=sim_time, out_dir=run_dir
            )
        else:
            heatmap_path = show_results_top_bot(
                result.T, H_map, amb,
                open_file=False, t_elapsed=sim_time, out_dir=run_dir
            )

        preview_path = save_preview_image(
            board, copper_ids, bbox, pads_list,
            settings, layer_names, stack_info,
            get_pad_pixels, create_multilayer_maps,
            self._derive_stackup_thicknesses,
            open_file=False, out_dir=run_dir
        )

        snapshot_debug = {
            "snapshots_enabled": settings.get('snapshots'),
            "snap_count": settings.get('snap_count'),
            "dt_base": dt,
            "steps_target": steps_target,
            "steps_total": result.step_counter,
            "snap_times": snap_times,
            "base_output_dir": base_output_dir,
            "run_dir": run_dir,
            "solver_backend": result.k_norm_info.get("backend"),
            "avg_solve_s": result.total_solve_time / max(result.step_counter, 1),
            "factorizations": result.factor_count,
            "factorization_s": result.total_factor_time,
            "phase_metrics": json.dumps(result.phase_metrics),
        }

        report_path = write_html_report(
            settings=settings,
            stack_info=stack_info,
            stackup_derived=stackup_derived,
            pad_power=pad_power,
            layer_names=layer_names,
            preview_path=preview_path,
            heatmap_path=heatmap_path,
            k_norm_info=result.k_norm_info,
            out_dir=run_dir,
            snapshot_debug=snapshot_debug,
            snapshot_files=result.snapshot_files
        )

        # Open outputs
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
