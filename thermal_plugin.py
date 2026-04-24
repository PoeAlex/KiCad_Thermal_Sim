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
import tempfile
import subprocess
import traceback
from dataclasses import dataclass

import pcbnew
import numpy as np
import wx

from .capabilities import HAS_LIBS, HAS_PARDISO, get_pypardiso_optional_dependency
from .stackup_parser import parse_stackup_from_board_file, format_stackup_report_um
from .gui_dialogs import SettingsDialog, prepare_current_groups
from .electrical_solver import CurrentTerminal, ElectricalConfig, solve_electrical_heating
from .geometry_mapper import build_geometry_state, create_multilayer_maps, get_pad_pixels
from .thermal_solver import SolverConfig, build_stiffness_matrix, run_simulation
from .pwl_parser import parse_pwl_file
from .visualization import (
    save_snapshot, show_results_top_bot, show_results_all_layers, save_preview_image,
    build_interactive_heatmap_payload
)
from .thermal_report import write_html_report


@dataclass
class SparsePadContribution:
    """Sparse per-pad power distribution on the simulation grid."""

    indices: np.ndarray
    weights: np.ndarray


def _bbox_to_power_indices(bbox, target_idx, rows, cols, x_min, y_min, res, rc):
    """
    Convert a pad bounding box to flattened node indices on one copper layer.

    Parameters
    ----------
    bbox : pcbnew.EDA_RECT
        Pad bounding box in KiCad internal units.
    target_idx : int
        Target copper layer index.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    x_min : float
        Grid origin x coordinate in millimeters.
    y_min : float
        Grid origin y coordinate in millimeters.
    res : float
        Grid resolution in millimeters.
    rc : int
        Number of nodes per layer.

    Returns
    -------
    np.ndarray
        Flattened node indices for the rectangular pad extent.
    """
    x0 = bbox.GetX() * 1e-6
    y0 = bbox.GetY() * 1e-6
    w = bbox.GetWidth() * 1e-6
    h = bbox.GetHeight() * 1e-6
    cs = max(0, int((x0 - x_min) / res))
    rs = max(0, int((y0 - y_min) / res))
    ce = min(cols, int((x0 + w - x_min) / res) + 1)
    re = min(rows, int((y0 + h - y_min) / res) + 1)
    if cs >= ce or rs >= re:
        return np.empty(0, dtype=np.int64)

    row_offsets = np.arange(rs, re, dtype=np.int64) * cols
    col_offsets = np.arange(cs, ce, dtype=np.int64)
    return target_idx * rc + (row_offsets[:, None] + col_offsets[None, :]).ravel(order="C")


def _pad_target_layer_index(board, copper_ids, pad, lid_to_idx):
    """
    Resolve the solver layer index for a pad.

    Parameters
    ----------
    board : pcbnew.BOARD
        The active board object.
    copper_ids : list of int
        Copper layer IDs in stackup order.
    pad : pcbnew.PAD
        Pad to place on the thermal grid.
    lid_to_idx : dict
        Mapping from KiCad layer IDs to solver indices.

    Returns
    -------
    int
        Target copper layer index for the pad.
    """
    pad_lid = pad.GetLayer()
    target_idx = lid_to_idx.get(pad_lid)
    if target_idx is not None:
        return target_idx

    try:
        lname = board.GetLayerName(pad_lid).upper()
    except Exception:
        lname = ""
    return len(copper_ids) - 1 if ("B." in lname or "BOT" in lname) else 0


def _build_sparse_pad_contributions(board, copper_ids, pads_list, rows, cols, x_min, y_min, res):
    """
    Build sparse per-pad unit power distributions.

    Parameters
    ----------
    board : pcbnew.BOARD
        The active board object.
    copper_ids : list of int
        Copper layer IDs in stackup order.
    pads_list : list
        Selected pads used as heat sources.
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    x_min : float
        Grid origin x coordinate in millimeters.
    y_min : float
        Grid origin y coordinate in millimeters.
    res : float
        Grid resolution in millimeters.

    Returns
    -------
    list of SparsePadContribution
        Sparse power distributions normalized to 1 W per pad.
    """
    rc = rows * cols
    lid_to_idx = {lid: idx for idx, lid in enumerate(copper_ids)}
    contributions = []

    for pad in pads_list:
        target_idx = _pad_target_layer_index(board, copper_ids, pad, lid_to_idx)
        indices = _bbox_to_power_indices(
            pad.GetBoundingBox(),
            target_idx=target_idx,
            rows=rows,
            cols=cols,
            x_min=x_min,
            y_min=y_min,
            res=res,
            rc=rc,
        )
        if indices.size:
            weights = np.full(indices.shape, 1.0 / float(indices.size), dtype=np.float64)
        else:
            weights = np.empty(0, dtype=np.float64)
        contributions.append(SparsePadContribution(indices=indices, weights=weights))

    return contributions


def _build_power_vector(pad_sources, pad_contributions, total_nodes):
    """
    Build the constant power vector and optional time-varying callback.

    Parameters
    ----------
    pad_sources : list
        Parsed pad source descriptors: ('const', value) or ('pwl', (times, powers)).
    pad_contributions : list of SparsePadContribution
        Sparse per-pad unit distributions.
    total_nodes : int
        Total number of thermal nodes.

    Returns
    -------
    tuple
        (Q, Q_func) where Q is the initial dense power vector and Q_func is an
        optional callback for time-varying inputs.
    """
    q_const = np.zeros(total_nodes, dtype=np.float64)
    pwl_terms = []

    for idx, (source_type, source_value) in enumerate(pad_sources):
        if idx >= len(pad_contributions):
            break
        contribution = pad_contributions[idx]
        if contribution.indices.size == 0:
            continue
        if source_type == 'const':
            q_const[contribution.indices] += float(source_value) * contribution.weights
        else:
            times, powers = source_value
            pwl_terms.append((
                np.asarray(times, dtype=np.float64),
                np.asarray(powers, dtype=np.float64),
                contribution.indices,
                contribution.weights,
            ))

    if not pwl_terms:
        return q_const, None

    q_initial = q_const.copy()
    for times, powers, indices, weights in pwl_terms:
        q_initial[indices] += float(np.interp(0.0, times, powers)) * weights

    def q_func(t, _q_const=q_const, _pwl_terms=tuple(pwl_terms)):
        q_t = _q_const.copy()
        for times, powers, indices, weights in _pwl_terms:
            q_t[indices] += float(np.interp(t, times, powers)) * weights
        return q_t

    return q_initial, q_func


def _format_timing_summary(timings):
    """Format initialization timings for console output."""
    parts = []
    for key in (
        "zone_refill_s",
        "geometry_maps_s",
        "capacity_build_s",
        "power_vector_build_s",
        "electrical_solve_s",
        "stiffness_matrix_s",
    ):
        if key in timings:
            parts.append(f"{key}={float(timings[key]):.4f}s")
    return ", ".join(parts)


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
        self.icon_file_name = os.path.join(os.path.dirname(__file__), "ThermalSim_icon.png")

        # Store references for preview
        self.board = None
        self.copper_ids = []
        self.bbox = None
        self.pads_list = []
        self.stack_info = None
        self.last_zone_refill_s = 0.0
        self.settings_dialog = None

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
            from .capabilities import get_missing_packages
            from .dependency_installer import DependencyInstallDialog
            missing = get_missing_packages()
            if missing:
                dlg = DependencyInstallDialog(None, missing, [get_pypardiso_optional_dependency()])
                dlg.ShowModal()
                dlg.Destroy()
            else:
                wx.MessageBox(
                    "All packages appear installed. Please restart KiCad.",
                    "ThermalSim"
                )
            return

        board = pcbnew.GetBoard()

        # Keep zone fills up-to-date
        zone_refill_start = time.perf_counter()
        try:
            pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        except Exception:
            pass
        zone_refill_s = time.perf_counter() - zone_refill_start

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

        # --- 3. Initial Pad Selection ---
        selected_pads = self._get_selected_pads(board)
        pads_list = [p[1] for p in selected_pads]

        # Store for preview
        self.board = board
        self.copper_ids = copper_ids
        self.bbox = bbox
        self.pads_list = pads_list
        self.stack_info = stack_info
        self.last_zone_refill_s = zone_refill_s

        # --- 4. Show Dialog ---
        stackup_details = format_stackup_report_um(stack_info) if stack_info else ""
        pad_names = self._format_pad_names(selected_pads)
        default_output_dir = os.path.dirname(__file__)
        last_settings = self._load_settings()
        if last_settings.get("output_dir") and os.path.isdir(last_settings.get("output_dir")):
            default_output_dir = last_settings.get("output_dir")

        if self.settings_dialog is not None:
            try:
                self.settings_dialog.Raise()
                return
            except Exception:
                self.settings_dialog = None

        def selection_provider():
            return self._get_selected_pad_descriptors(board)

        def run_callback(settings):
            self.settings_dialog = None
            self._save_settings(settings)
            current_pads = self._resolve_current_pad_objects(board, settings)
            focus_pads = self._unique_pads(pads_list + current_pads)
            self._run_simulation(
                board, copper_ids, layer_names, bbox, pads_list,
                settings, stack_info, pad_names, zone_refill_s=zone_refill_s,
                focus_pads=focus_pads
            )

        def close_callback():
            self.settings_dialog = None

        dlg = SettingsDialog(
            None, len(pads_list), suggested_res, layer_names,
            preview_callback=self.generate_preview,
            selection_provider=selection_provider,
            run_callback=run_callback,
            close_callback=close_callback,
            stackup_details=stackup_details,
            pad_names=pad_names,
            default_output_dir=default_output_dir,
            defaults=last_settings
        )
        self.settings_dialog = dlg
        try:
            dlg.Show()
        except Exception:
            try:
                if dlg.ShowModal() == wx.ID_OK:
                    settings = dlg.get_values()
                    if settings:
                        self._save_settings(settings)
                        current_pads = self._resolve_current_pad_objects(board, settings)
                        focus_pads = self._unique_pads(pads_list + current_pads)
                        self._run_simulation(
                            board, copper_ids, layer_names, bbox, pads_list,
                            settings, stack_info, pad_names, zone_refill_s=zone_refill_s,
                            focus_pads=focus_pads
                        )
            finally:
                self.settings_dialog = None
                dlg.Destroy()

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

    def _pad_key(self, fp_ref, pad):
        """Build a stable key for a pad in settings."""
        try:
            pos = pad.GetPosition()
            px, py = int(pos.x), int(pos.y)
        except Exception:
            px, py = 0, 0
        try:
            net_code = int(pad.GetNetCode())
        except Exception:
            net_code = 0
        try:
            number = pad.GetNumber()
        except Exception:
            number = ""
        return f"{fp_ref}:{number}:{net_code}:{px}:{py}"

    def _pad_net_name(self, pad):
        """Return the display net name for a pad."""
        try:
            net = pad.GetNetname()
        except Exception:
            try:
                net = pad.GetNet().GetNetname()
            except Exception:
                net = ""
        return net or ""

    def _pad_net_code(self, pad):
        """Return the KiCad net code for a pad."""
        try:
            return int(pad.GetNetCode())
        except Exception:
            try:
                return int(pad.GetNet().GetNetCode())
            except Exception:
                return 0

    def _pad_layer_name(self, board, pad):
        """Return the KiCad layer name for a pad."""
        try:
            return board.GetLayerName(pad.GetLayer())
        except Exception:
            return str(pad.GetLayer())

    def _pad_descriptor(self, board, fp_ref, pad):
        """Return a serializable pad descriptor for current groups."""
        try:
            number = pad.GetNumber()
        except Exception:
            number = ""
        name = f"{fp_ref}-{number}"
        net_name = self._pad_net_name(pad)
        return {
            "pad_key": self._pad_key(fp_ref, pad),
            "name": f"{name} [{net_name}]" if net_name else name,
            "net_name": net_name,
            "net_code": self._pad_net_code(pad),
            "layer": self._pad_layer_name(board, pad),
            "current_a": 0.0,
        }

    def _get_selected_pad_descriptors(self, board):
        """Return serializable descriptors for currently selected KiCad pads."""
        descriptors = []
        try:
            footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
            for fp in footprints:
                fp_ref = fp.GetReference()
                for pad in fp.Pads():
                    if pad.IsSelected():
                        descriptors.append(self._pad_descriptor(board, fp_ref, pad))
        except Exception:
            return []
        descriptors.sort(key=lambda item: item.get("name", ""))
        return descriptors

    def _build_pad_lookup(self, board):
        """Build lookup maps from persisted pad descriptors to live pad objects."""
        by_key = {}
        by_name = {}
        try:
            footprints = board.Footprints() if hasattr(board, 'Footprints') else board.GetFootprints()
        except Exception:
            footprints = []
        for fp in footprints:
            fp_ref = fp.GetReference()
            for pad in fp.Pads():
                descriptor = self._pad_descriptor(board, fp_ref, pad)
                by_key[descriptor["pad_key"]] = (descriptor, pad)
                by_name[descriptor["name"]] = (descriptor, pad)
        return by_key, by_name

    def _resolve_current_pad_objects(self, board, settings):
        """Resolve current-group pad descriptors to live pad objects."""
        if not settings.get("current_enabled"):
            return []
        by_key, by_name = self._build_pad_lookup(board)
        pads = []
        seen = set()
        for group in prepare_current_groups(settings.get("current_groups", [])):
            for pad_info in group.get("pads", []):
                match = by_key.get(pad_info.get("pad_key")) or by_name.get(pad_info.get("name"))
                if not match:
                    continue
                _, pad = match
                if id(pad) not in seen:
                    pads.append(pad)
                    seen.add(id(pad))
        return pads

    def _resolve_current_terminals(self, board, settings):
        """Resolve current-group settings into CurrentTerminal objects."""
        terminals = []
        missing = []
        if not settings.get("current_enabled"):
            return terminals, missing
        by_key, by_name = self._build_pad_lookup(board)
        for group in prepare_current_groups(settings.get("current_groups", [])):
            for pad_info in group.get("pads", []):
                current = float(pad_info.get("current_a", 0.0) or 0.0)
                if abs(current) <= 0.0:
                    continue
                match = by_key.get(pad_info.get("pad_key")) or by_name.get(pad_info.get("name"))
                if not match:
                    missing.append(pad_info.get("name", pad_info.get("pad_key", "<unknown>")))
                    continue
                descriptor, pad = match
                terminals.append(CurrentTerminal(
                    pad=pad,
                    name=descriptor.get("name", pad_info.get("name", "")),
                    net_name=descriptor.get("net_name", pad_info.get("net_name", "")),
                    net_code=int(descriptor.get("net_code", pad_info.get("net_code", 0)) or 0),
                    current_a=current,
                ))
        return terminals, missing

    def _unique_pads(self, pads):
        """Return pads without duplicate object identities."""
        result = []
        seen = set()
        for pad in pads or []:
            ident = id(pad)
            if ident in seen:
                continue
            result.append(pad)
            seen.add(ident)
        return result

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
        current_pads = self._resolve_current_pad_objects(self.board, settings)
        preview_pads = self._unique_pads(self.pads_list + current_pads)
        output_file = save_preview_image(
            self.board, self.copper_ids, self.bbox, preview_pads,
            settings, layer_names,
            self.stack_info if self.stack_info is not None else parse_stackup_from_board_file(self.board),
            get_pad_pixels,
            create_multilayer_maps,
            self._derive_stackup_thicknesses,
            open_file=True
        )
        if not output_file:
            wx.MessageBox("Board data missing for preview", "Error")

    def _run_simulation(self, board, copper_ids, layer_names, bbox, pads_list,
                        settings, stack_info, pad_names, zone_refill_s=0.0,
                        focus_pads=None):
        """Execute the thermal simulation."""
        focus_pads = focus_pads if focus_pads is not None else pads_list
        if settings.get("current_enabled") and settings.get("limit_area"):
            settings = dict(settings)
            settings["limit_area"] = False
            wx.MessageBox(
                "Limit Area was disabled for current-flow simulation so copper paths are not cut off.",
                "ThermalSim"
            )
        # Derive thicknesses
        stackup_derived = self._derive_stackup_thicknesses(board, copper_ids, stack_info, settings)
        total_thick_mm = stackup_derived["total_thick_mm_used"]
        init_timings = {"zone_refill_s": float(zone_refill_s)}

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
            pad_xs = [pad.GetPosition().x * 1e-6 for pad in focus_pads]
            pad_ys = [pad.GetPosition().y * 1e-6 for pad in focus_pads]
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
        via_factor = 390.0 / 0.3
        k_cu = 390.0
        k_fr4 = 0.3
        rho_cu, cp_cu = 8960.0, 385.0
        rho_fr4, cp_fr4 = 1850.0, 1100.0

        cu_thick_mm_used = stackup_derived["copper_thickness_mm_used"]
        gap_mm_used = stackup_derived["gap_mm_used"]
        cu_thick_m = [max(1e-9, th * 1e-3) for th in cu_thick_mm_used]
        gap_m = [max(1e-9, g * 1e-3) for g in gap_mm_used]

        # Build internal geometry state
        geometry_start = time.perf_counter()
        try:
            geometry_state = build_geometry_state(
                board=board,
                copper_ids=copper_ids,
                rows=rows,
                cols=cols,
                x_min=x_min,
                y_min=y_min,
                res=res,
                settings=settings,
                via_factor=via_factor,
                pads_list=focus_pads,
            )
        except Exception as e:
            wx.MessageBox(f"Error mapping geometry: {e}", "Error")
            return
        init_timings["geometry_maps_s"] = time.perf_counter() - geometry_start
        copper_mask = geometry_state.copper_mask
        V_map = geometry_state.via_map
        H_map = geometry_state.heatsink_mask.astype(np.float64, copy=False)

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
        capacity_start = time.perf_counter()
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
        init_timings["capacity_build_s"] = time.perf_counter() - capacity_start

        # Power injection (supports constant values and PWL file paths)
        RC = rows * cols
        N = RC * layer_count
        power_start = time.perf_counter()

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

        pad_contributions = _build_sparse_pad_contributions(
            board=board,
            copper_ids=copper_ids,
            pads_list=pads_list,
            rows=rows,
            cols=cols,
            x_min=x_min,
            y_min=y_min,
            res=res,
        )
        Q, Q_func = _build_power_vector(pad_sources, pad_contributions, N)
        init_timings["power_vector_build_s"] = time.perf_counter() - power_start

        electrical_summary = None
        if settings.get("current_enabled"):
            electrical_start = time.perf_counter()
            terminals, missing_pads = self._resolve_current_terminals(board, settings)
            if missing_pads:
                wx.MessageBox(
                    "Current simulation references pads that were not found on the board:\n"
                    + "\n".join(str(name) for name in missing_pads),
                    "Current Path Error"
                )
                return
            if terminals:
                electrical_config = ElectricalConfig(
                    copper_ids=list(copper_ids),
                    rows=rows,
                    cols=cols,
                    x_min=x_min,
                    y_min=y_min,
                    res=res,
                    t_cu=np.asarray(t_cu, dtype=np.float64),
                )
                electrical_result = solve_electrical_heating(board, terminals, electrical_config)
                if not electrical_result.valid:
                    wx.MessageBox(
                        "Current simulation validation failed:\n\n"
                        + "\n".join(electrical_result.errors),
                        "Current Path Error"
                    )
                    return
                q_joule = electrical_result.q_joule
                if np.any(q_joule):
                    Q = Q + q_joule
                    if Q_func is not None:
                        base_q_func = Q_func

                        def q_func_with_joule(t, _base=base_q_func, _q_joule=q_joule):
                            return _base(t) + _q_joule

                        Q_func = q_func_with_joule
                electrical_summary = {
                    "total_loss_w": electrical_result.total_loss_w,
                    "warnings": electrical_result.warnings,
                    "nets": [
                        {
                            "net": item.net_name,
                            "terminal_count": item.terminal_count,
                            "total_abs_current_a": item.total_abs_current_a,
                            "total_loss_w": item.total_loss_w,
                            "max_node_power_w": item.max_node_power_w,
                            "connected_component_count": item.connected_component_count,
                        }
                        for item in electrical_result.net_summaries
                    ],
                }
            else:
                electrical_summary = {
                    "total_loss_w": 0.0,
                    "warnings": ["Current simulation enabled but no non-zero pad currents were configured."],
                    "nets": [],
                }
            init_timings["electrical_solve_s"] = time.perf_counter() - electrical_start

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
        stiffness_start = time.perf_counter()
        K_matrix, b, hA, _ = build_stiffness_matrix(
            layer_count, rows, cols, copper_mask, t_cu, t_fr4_eff,
            k_cu, k_fr4, dx, dy, V_map, gap_m, H_map, settings, amb
        )
        init_timings["stiffness_matrix_s"] = time.perf_counter() - stiffness_start
        print(f"[ThermalSim] init timings: {_format_timing_summary(init_timings)}")

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
            "init_zone_refill_s": init_timings.get("zone_refill_s"),
            "init_geometry_maps_s": init_timings.get("geometry_maps_s"),
            "init_capacity_build_s": init_timings.get("capacity_build_s"),
            "init_power_vector_build_s": init_timings.get("power_vector_build_s"),
            "init_electrical_solve_s": init_timings.get("electrical_solve_s"),
            "init_stiffness_matrix_s": init_timings.get("stiffness_matrix_s"),
            "electrical_summary": electrical_summary,
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
            board, copper_ids, bbox, focus_pads,
            settings, layer_names, stack_info,
            get_pad_pixels, create_multilayer_maps,
            self._derive_stackup_thicknesses,
            open_file=False, out_dir=run_dir
        )

        interactive_heatmap = build_interactive_heatmap_payload(
            result.T,
            amb=amb,
            layer_names=layer_names,
            res_mm=res,
            x_min_mm=x_min,
            y_min_mm=y_min,
            show_all=bool(settings.get('show_all', True))
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
            "init_zone_refill_s": init_timings.get("zone_refill_s"),
            "init_geometry_maps_s": init_timings.get("geometry_maps_s"),
            "init_capacity_build_s": init_timings.get("capacity_build_s"),
            "init_power_vector_build_s": init_timings.get("power_vector_build_s"),
            "init_electrical_solve_s": init_timings.get("electrical_solve_s"),
            "init_stiffness_matrix_s": init_timings.get("stiffness_matrix_s"),
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
            snapshot_files=result.snapshot_files,
            interactive_heatmap=interactive_heatmap,
            electrical_summary=electrical_summary
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
