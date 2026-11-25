"""
ThermalSim - KiCad PCB thermal simulation plugin.

This is the main controller module that orchestrates the thermal simulation
workflow using the specialized sub-modules.
"""

import os
import re
import time
import math
import json
import inspect
import tempfile
import threading
import traceback
from dataclasses import asdict

import pcbnew
import numpy as np
import wx

from .capabilities import HAS_LIBS
try:
    from .capabilities import HAS_NATIVE_MG
except ImportError:
    HAS_NATIVE_MG = False
try:
    from .native_core import HAS_NATIVE_COMPACT, clear_native_scratch
except ImportError:
    HAS_NATIVE_COMPACT = False

    def clear_native_scratch():
        return None
from .stackup_parser import parse_stackup_from_board_file, format_stackup_report_um
from .gui_dialogs import (
    DEFAULT_GRID_MAX_CELLS,
    DEFAULT_GRID_TARGET_CELLS,
    MAX_CUSTOM_GRID_NODES,
    SettingsDialog,
    prepare_current_groups,
    prepare_power_pads,
    resolve_heatsink_cp,
)
from .electrical_solver import (
    CurrentTerminal,
    ElectricalConfig,
    net_key_from_obj,
    net_key_from_values,
    solve_electrical_heating,
)
from .geometry_mapper import (
    build_geometry_state,
    create_multilayer_maps,
    get_pad_copper_layer_indices,
    get_pad_pixels,
    get_pad_power_indices,
    get_pad_primary_layer_id,
)
try:
    from .geometry_mapper import GeometryCancelledError
except ImportError:
    class GeometryCancelledError(RuntimeError):
        pass
from .thermal_solver import (
    HeatsinkModel,
    SolverConfig,
    build_heatsink_model,
    build_stiffness_matrix,
    build_structured_operator,
    compute_heatsink_capacity_j_k,
    run_simulation,
    run_simulation_matrix_free,
)
try:
    from .thermal_solver import (
        CompactStructuredThermalOperator,
        SparsePowerVector,
        build_compact_structured_operator,
    )
except ImportError:
    CompactStructuredThermalOperator = None
    SparsePowerVector = None
    build_compact_structured_operator = None

_ORIG_BUILD_STRUCTURED_OPERATOR = build_structured_operator
_ORIG_BUILD_HEATSINK_MODEL = build_heatsink_model


def _call_with_optional_progress_kwargs(
    fn, *args, progress_callback=None, cancel_check=None, optional_kwargs=None, **kwargs
):
    """Invoke fn passing progress_callback/cancel_check and optional kwargs only when supported."""
    merged_optional = dict(optional_kwargs or {})
    if progress_callback is not None:
        merged_optional["progress_callback"] = progress_callback
    if cancel_check is not None:
        merged_optional["cancel_check"] = cancel_check
    return call_with_supported_kwargs(
        fn, *args, optional_kwargs=merged_optional, **kwargs
    )


def _message_phase(message):
    """Extract stable stage prefix so step-by-step updates stay throttled."""
    text = str(message or "")
    for sep in (" - ", " (", ":"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip()


def _format_eta_seconds(seconds: float) -> str:
    """Format an estimated remaining duration in concise English."""
    total_sec = max(1, int(round(max(0.0, float(seconds)))))
    if total_sec < 60:
        return f"~{total_sec}s remaining"
    if total_sec < 3600:
        mins, secs = divmod(total_sec, 60)
        return f"~{mins}m {secs:02d}s remaining"
    hours, rem = divmod(total_sec, 3600)
    mins = rem // 60
    return f"~{hours}h {mins:02d}m remaining"


class _UiProgressReporter:
    """Cancellable wx.ProgressDialog wrapper with throttled updates and background pumping."""

    def __init__(self, title="ThermalSim", initial_message="Preparing...", parent=None,
                 throttle_s=0.04, poll_interval_s=0.05):
        self.cancel_token = CancellationToken()
        self.throttle_s = float(throttle_s)
        self.poll_interval_s = float(poll_interval_s)
        self._last_update_t = 0.0
        self._last_message = str(initial_message or "")
        self._last_phase = _message_phase(self._last_message)
        self._last_percent = 0
        self._closed = False
        self.pd = self._create_dialog(title, self._last_message, parent=parent)

    @staticmethod
    def _create_dialog(title, message, parent=None):
        pd_cls = getattr(wx, "ProgressDialog", None)
        if not callable(pd_cls):
            return None
        style = (
            getattr(wx, "PD_CAN_ABORT", 0)
            | getattr(wx, "PD_APP_MODAL", 0)
            | getattr(wx, "PD_ELAPSED_TIME", 0)
            | getattr(wx, "PD_AUTO_HIDE", 0)
        )
        kwargs = {"style": style}
        if parent is not None:
            kwargs["parent"] = parent
        try:
            return pd_cls(title, message, 100, **kwargs)
        except TypeError:
            kwargs.pop("parent", None)
            try:
                return pd_cls(title, message, 100, **kwargs)
            except Exception:
                return None
        except Exception:
            return None

    @property
    def cancelled(self):
        return self.cancel_token.cancelled

    @staticmethod
    def _yield_wx():
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

    def update(self, percent, message="", force=False):
        """Update progress dialog and pump wx events; return False if cancelled."""
        if self.cancel_token.cancelled:
            return False
        clamped_pct = max(0, min(100, int(percent)))
        msg = str(message) if message else self._last_message
        phase = _message_phase(msg)
        phase_changed = phase != self._last_phase
        now = time.monotonic()
        if not force and not phase_changed and (now - self._last_update_t) < self.throttle_s:
            return True
        self._last_update_t = now
        self._last_message = msg
        self._last_phase = phase
        self._last_percent = clamped_pct
        if self.pd is not None:
            try:
                update_result = self.pd.Update(clamped_pct, msg)
                keep_going = (
                    update_result[0]
                    if isinstance(update_result, tuple)
                    else update_result
                )
                was_cancelled = (
                    hasattr(self.pd, "WasCancelled") and bool(self.pd.WasCancelled())
                )
                if (keep_going is not None and not keep_going) or was_cancelled:
                    self.cancel_token.cancel()
            except Exception:
                self.cancel_token.cancel()
        self._yield_wx()
        return not self.cancel_token.cancelled

    def make_stage_callback(self, start_pct, end_pct, default_message=""):
        """Create a (done, total, message) -> bool callback mapped to [start_pct, end_pct]."""
        base = int(start_pct)
        span = max(0, int(end_pct) - base)

        def _callback(done, total, message=None):
            denom = max(1, int(total))
            frac = max(0.0, min(1.0, float(done) / float(denom)))
            pct = base + int(round(frac * span))
            return self.update(pct, message or default_message)

        return _callback

    def make_cancel_check(self):
        return lambda: self.cancel_token.cancelled

    def run_in_background(self, fn, start_pct, end_pct, message,
                          progress_state=None, message_fn=None):
        """Execute fn in a daemon thread while pumping wx events and checking cancel."""
        if not self.update(start_pct, message, force=True):
            raise GeometryCancelledError("Operation cancelled by user.")

        box = {}

        def _worker():
            try:
                box["value"] = fn()
            except BaseException as exc:
                box["exc"] = exc

        worker = threading.Thread(
            target=_worker, name="ThermalSimWorker", daemon=True
        )
        started_at = time.monotonic()
        base = int(start_pct)
        span = max(0, int(end_pct) - base)
        worker.start()
        while worker.is_alive():
            if progress_state is not None:
                cur = int(progress_state.get("current", 0))
                tot = max(1, int(progress_state.get("total", 1)))
                frac = max(0.0, min(1.0, float(cur) / float(tot)))
                pct = base + int(round(frac * span))
                msg = (
                    message_fn(progress_state)
                    if message_fn is not None
                    else (progress_state.get("message") or message)
                )
            else:
                elapsed = max(0.0, time.monotonic() - started_at)
                max_idle_pct = max(base, int(end_pct) - 1)
                idle_span = max(0, max_idle_pct - base)
                pct = base + int(idle_span * (1.0 - math.exp(-elapsed / 2.5)))
                msg = message
            self.update(pct, msg)
            worker.join(self.poll_interval_s)
        worker.join()

        if "exc" in box:
            raise box["exc"]
        if not self.cancel_token.cancelled:
            final_msg = (
                message_fn(progress_state)
                if (progress_state is not None and message_fn is not None)
                else message
            )
            self.update(end_pct, final_msg, force=True)
        return box.get("value")

    def close(self):
        """Hide and destroy the underlying ProgressDialog safely."""
        if self._closed:
            return
        self._closed = True
        if self.pd is not None:
            try:
                self.pd.Hide()
            except Exception:
                pass
            try:
                self.pd.Destroy()
            except Exception:
                pass
            self._yield_wx()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
from .pwl_parser import parse_pwl_file
from .visualization import (
    save_snapshot, show_results_top_bot, show_results_all_layers, save_preview_image,
    build_interactive_heatmap_payload, save_joule_loss_map
)
from .thermal_report import write_html_report
from .workflow import (
    AreaEstimate,
    BoardSnapshot,
    CancellationToken,
    GeometryCache,
    ThermalFactorizationCache,
    ThermalOperatorCache,
    GridEstimate,
    PreflightResult,
    SimulationArtifacts,
    get_process_memory_mb,
    geometry_cache_key,
    stable_fingerprint,
)


from .simulation_core import (
    DEFAULT_GRID_DETAIL,
    DEFAULT_GRID_MAX_CELLS,
    DEFAULT_GRID_NODE_BUDGET,
    DEFAULT_GRID_TARGET_CELLS,
    GRID_DETAIL_PRESETS,
    MAX_CUSTOM_GRID_NODES,
    SparsePadContribution,
    auto_fast_engine_threshold,
    bbox_bounds_mm as _bbox_bounds_mm,
    build_compact_heatsink_model,
    build_dense_capacity as _build_dense_capacity,
    build_power_vector as _build_power_vector,
    build_sparse_pad_contributions as _build_sparse_pad_contributions,
    build_sparse_power_vector,
    calculate_suggested_resolution,
    call_with_supported_kwargs,
    coarsen_grid_resolution,
    derive_stackup_thicknesses,
    effective_fr4_control_volume_thicknesses as _effective_fr4_control_volume_thicknesses,
    estimate_simulation_area as _estimate_simulation_area,
    estimate_solver_cost,
    estimate_solver_grid,
    pad_target_layer_index as _pad_target_layer_index,
    resolve_convection_settings,
    resolve_grid_limits as _resolve_grid_limits,
    resolve_grid_policy as _resolve_grid_policy,
    resolve_mesh_mode,
    resolve_via_settings,
)


def _auto_fast_engine_threshold():
    """Return the node threshold for switching Auto compute_engine to matrix-free."""
    return auto_fast_engine_threshold(HAS_NATIVE_MG)


def _resolve_mesh_mode(settings, use_fast_engine, node_count):
    """Return the active spatial mesh mode (always ``'uniform'`` since v0.4.7)."""
    return resolve_mesh_mode(
        settings, use_fast_engine, node_count, has_native_mg=HAS_NATIVE_MG
    )


def _coarsen_grid_resolution(w_mm, h_mm, requested_res, settings, layer_count=1):
    """Apply automatic grid coarsening for large boards."""
    return coarsen_grid_resolution(
        w_mm, h_mm, requested_res, settings, layer_count=layer_count
    )


def _build_sparse_power_vector(pad_sources, pad_contributions, total_nodes):
    """Build a SparsePowerVector for constant pad sources without dense N-float allocation."""
    return build_sparse_power_vector(
        pad_sources,
        pad_contributions,
        total_nodes,
        sparse_power_cls=SparsePowerVector,
    )


def _build_compact_heatsink_model(dx, dy, heatsink_mask, settings, amb):
    """Build a 2D bottom-layer HeatsinkModel without allocating a full 3D vector."""
    return build_compact_heatsink_model(
        dx,
        dy,
        heatsink_mask,
        settings,
        amb,
        heatsink_model_cls=HeatsinkModel,
        capacity_fn=compute_heatsink_capacity_j_k,
    )


def _estimate_solver_cost(nodes, settings):
    """Return calibrated memory bounds (MB) and a relative runtime class."""
    return estimate_solver_cost(
        nodes,
        settings,
        has_native_mg=HAS_NATIVE_MG,
        has_native_compact=HAS_NATIVE_COMPACT,
    )


def _estimate_solver_grid(
    bbox, requested_res, settings, layer_count, focus_pads=None, area=None
):
    """Compute the final solver grid after area limiting and coarsening."""
    return estimate_solver_grid(
        bbox,
        requested_res,
        settings,
        layer_count,
        focus_pads=focus_pads,
        area=area,
        has_native_mg=HAS_NATIVE_MG,
        has_native_compact=HAS_NATIVE_COMPACT,
    )


def _build_operator_cache_key(
    cache_key,
    resolved_engine,
    t_cu,
    gap_m,
    t_fr4_eff,
    dx,
    amb,
    settings,
):
    """Return stable operator cache key including thermal-pad mode and contact settings."""
    pad_sink_mode = str(settings.get("pad_sink_mode", "fixed") or "fixed").strip().lower()
    pad_contact_factor = settings.get("pad_contact_factor", 0.2)
    pad_sink_temp_c = settings.get("pad_sink_temp_c", 25.0)
    h_top, h_bottom = resolve_convection_settings(settings)
    return stable_fingerprint({
        "geometry": cache_key,
        "compute_engine": resolved_engine,
        "copper_thickness_m": [float(value) for value in t_cu],
        "dielectric_gap_m": [float(value) for value in gap_m],
        "fr4_effective_m": [float(value) for value in t_fr4_eff],
        "cell_size_m": float(dx),
        "ambient_c": float(amb),
        "h_conv": h_top,
        "h_conv_bottom": h_bottom,
        "pad_th": float(settings.get("pad_th", 1.0)),
        "pad_k": float(settings.get("pad_k", 3.0)),
        "pad_cap_areal": float(settings.get("pad_cap_areal", 0.0) or 0.0),
        "pad_sink_mode": pad_sink_mode,
        "pad_contact_factor": float(
            0.2 if pad_contact_factor is None else pad_contact_factor
        ),
        "pad_sink_temp_use_ambient": bool(
            settings.get("pad_sink_temp_use_ambient", True)
        ),
        "pad_sink_temp_c": float(
            25.0 if pad_sink_temp_c is None else pad_sink_temp_c
        ),
    })


def _build_factorization_cache_key(operator_key, config, steady_state, settings):
    """Return stable factorization cache key including 0D heatsink parameters."""
    rth_kw = settings.get("heatsink_rth_kw", 5.0)
    mass_g = settings.get("heatsink_mass_g", 50.0)
    cp_j_kgk = resolve_heatsink_cp(
        settings.get("heatsink_material", "aluminium"),
        settings.get("heatsink_cp_j_kgk", 897.0),
    )
    return stable_fingerprint({
        "operator": operator_key,
        "time_stepping": config.time_stepping,
        "sim_time": float(config.sim_time),
        "dt_base": float(config.dt_base),
        "steps_target": int(config.steps_target),
        "use_multi_phase": bool(config.use_multi_phase),
        "steady_state": bool(steady_state),
        "heatsink_rth_kw": float(5.0 if rth_kw is None else rth_kw),
        "heatsink_mass_g": float(50.0 if mass_g is None else mass_g),
        "heatsink_cp_j_kgk": float(cp_j_kgk),
    })


def _electrical_net_summary_dict(summary):
    """Convert electrical net diagnostics to report-friendly dictionaries."""
    data = asdict(summary)
    data["net"] = data.get("net_name", "")
    return data


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


def _write_run_manifest(run_dir, status, **details):
    """Persist the lifecycle state of a result directory."""
    payload = {"schema_version": 1, "status": str(status), "updated_at": time.time()}
    payload.update(details)
    try:
        with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    except Exception:
        pass


def _find_pcb_editor_parent(board=None):
    """Return the PCB Editor top-level window for parenting plugin dialogs.

    KiCad can run the project manager and PCB Editor in the same wx
    application. A dialog created with ``parent=None`` may then be attached to
    the project manager even when the plugin was launched from PCB Editor.
    """
    try:
        active = wx.GetActiveWindow()
    except Exception:
        active = None

    def top_level(window):
        current = window
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            try:
                parent = current.GetParent()
            except Exception:
                parent = None
            if parent is None:
                break
            current = parent
        return current

    active_top = top_level(active)
    candidates = []
    try:
        candidates.extend(list(wx.GetTopLevelWindows()))
    except Exception:
        pass
    if active_top is not None and active_top not in candidates:
        candidates.append(active_top)

    board_tokens = []
    if board is not None:
        try:
            board_name = os.path.basename(str(board.GetFileName() or "")).lower()
            board_stem = os.path.splitext(board_name)[0]
            board_tokens = [token for token in (board_name, board_stem) if token]
        except Exception:
            pass

    editor_markers = (
        "pcb editor", "pcbnew", "pcb_edit_frame", "pcb editor frame",
        "leiterplatteneditor", "platteneditor",
    )
    manager_markers = ("project manager", "projektmanager")

    def window_text(window):
        parts = [type(window).__name__]
        for getter_name in ("GetTitle", "GetName", "GetClassName"):
            try:
                value = getattr(window, getter_name)()
                if value:
                    parts.append(str(value))
            except Exception:
                continue
        try:
            parts.append(str(window.GetClassInfo().GetClassName()))
        except Exception:
            pass
        return " ".join(parts).lower()

    best = None
    best_score = -10_000
    for window in candidates:
        if window is None:
            continue
        try:
            if hasattr(window, "IsBeingDeleted") and window.IsBeingDeleted():
                continue
        except Exception:
            continue
        text = window_text(window)
        score = 100 if window is active_top else 0
        if any(marker in text for marker in editor_markers):
            score += 1_000
        if any(token in text for token in board_tokens):
            score += 500
        if any(marker in text for marker in manager_markers):
            score -= 1_000
        if score > best_score:
            best = window
            best_score = score

    return best if best_score > 0 else active_top


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
        self.geometry_cache = GeometryCache(persistent=True)
        self.operator_cache = ThermalOperatorCache()
        self.electrical_cache = ThermalOperatorCache()
        self.factorization_cache = ThermalFactorizationCache()
        self.cancel_token = None
        self.last_artifacts = None
        self.startup_dialog = None
        self.host_window = None

    def _show_startup_progress(self, percent, message):
        """Show immediate feedback while the settings dialog is prepared."""
        try:
            if self.startup_dialog is None:
                self.startup_dialog = wx.ProgressDialog(
                    "ThermalSim",
                    message,
                    maximum=100,
                    parent=self.host_window,
                    style=getattr(wx, "PD_APP_MODAL", 0),
                )
            self.startup_dialog.Update(int(percent), message)
            app = wx.GetApp()
            if app is not None:
                app.Yield()
        except Exception:
            pass

    def _close_startup_progress(self):
        """Close the temporary startup progress window."""
        if self.startup_dialog is None:
            return
        try:
            self.startup_dialog.Destroy()
        except Exception:
            pass
        self.startup_dialog = None

    def _count_unfilled_copper_zones(self, board):
        """Return the number of normal copper zones without a valid fill."""
        try:
            zones = list(board.Zones() if hasattr(board, "Zones") else board.GetZones())
        except Exception:
            return 0
        unfilled = 0
        for zone in zones:
            try:
                is_rule_area = bool(
                    getattr(zone, "GetIsRuleArea", lambda: False)()
                )
                is_keepout = bool(
                    getattr(zone, "GetIsKeepout", lambda: False)()
                )
                if is_rule_area or is_keepout:
                    continue
                if hasattr(zone, "IsFilled") and not zone.IsFilled():
                    unfilled += 1
            except Exception:
                continue
        return unfilled

    def _require_filled_zones(self, board):
        """Block geometry work safely when KiCad copper zones are stale."""
        unfilled = self._count_unfilled_copper_zones(board)
        if unfilled <= 0:
            return True
        wx.MessageBox(
            f"{unfilled} copper zone(s) are not filled.\n\n"
            "Please return to PCB Editor, press B to refill all zones, "
            "then start Preview or Run again.\n\n"
            "ThermalSim no longer refills zones automatically because that "
            "can block the UI for a long time and can crash some KiCad builds.",
            "ThermalSim - Refill Copper Zones",
        )
        return False

    def _capture_board_snapshot(self, board, copper_ids, bbox, progress_callback=None):
        """Capture a deterministic identity for safe persistent geometry reuse."""
        tracks = list(board.Tracks() if hasattr(board, "Tracks") else board.GetTracks())
        footprints = list(board.Footprints() if hasattr(board, "Footprints") else board.GetFootprints())
        zones = list(board.Zones() if hasattr(board, "Zones") else board.GetZones())
        try:
            drawings = list(board.GetDrawings() if hasattr(board, "GetDrawings") else [])
        except Exception:
            drawings = []

        eco1_layer = getattr(pcbnew, "Eco1_User", None)
        eco1_layer_id = int(eco1_layer) if eco1_layer is not None else None
        total_items = max(
            1,
            len(tracks)
            + (len(drawings) if eco1_layer_id is not None else 0)
            + len(zones)
            + len(footprints),
        )
        done_items = 0
        last_progress_t = time.monotonic()

        def _emit_snapshot_progress(message, force=False):
            nonlocal last_progress_t
            if progress_callback is None:
                return
            now = time.monotonic()
            if force or (now - last_progress_t) >= 0.05:
                last_progress_t = now
                if progress_callback(done_items, total_items, message) is False:
                    raise GeometryCancelledError("Board snapshot cancelled.")

        _emit_snapshot_progress("Capturing board snapshot...", force=True)

        def _extract_layer_set_tuple(obj):
            get_layer_set = getattr(obj, "GetLayerSet", None)
            if get_layer_set is None:
                return None
            try:
                lset = get_layer_set()
                if hasattr(lset, "CuStack"):
                    return tuple(int(v) for v in lset.CuStack())
                if hasattr(lset, "IntSeq"):
                    return tuple(int(v) for v in lset.IntSeq())
            except Exception:
                return None
            return None

        primitive_rows = []
        track_getters = ("GetStart", "GetMid", "GetEnd", "GetPosition")
        for item in tracks:
            done_items += 1
            if progress_callback is not None and (done_items & 63) == 0:
                _emit_snapshot_progress(
                    f"Scanning tracks ({done_items}/{len(tracks)})..."
                )
            try:
                rect = item.GetBoundingBox()
                type_name = type(item).__name__
                row = [
                    type_name, int(item.GetLayer()), rect.GetX(), rect.GetY(),
                    rect.GetWidth(), rect.GetHeight(),
                ]
                for getter in track_getters:
                    fn = getattr(item, getter, None)
                    if fn is not None:
                        try:
                            point = fn()
                            row.extend((getter, int(point.x), int(point.y)))
                        except Exception:
                            continue
                is_via = "VIA" in type_name.upper()
                try:
                    if is_via:
                        if copper_ids:
                            try:
                                width_val = int(item.GetWidth(int(copper_ids[0])))
                            except TypeError:
                                width_val = int(item.GetWidth())
                        else:
                            width_val = int(rect.GetWidth())
                    else:
                        width_val = int(item.GetWidth())
                    row.extend(("width", width_val))
                except Exception:
                    pass
                if is_via:
                    drill_val = None
                    for drill_getter in ("GetDrillValue", "GetDrill"):
                        fn = getattr(item, drill_getter, None)
                        if fn is not None:
                            try:
                                cand = int(fn())
                                if cand > 0:
                                    drill_val = cand
                                    break
                            except Exception:
                                continue
                    if drill_val is not None:
                        row.extend(("drill", drill_val))
                    via_layers = _extract_layer_set_tuple(item)
                    if via_layers is not None:
                        row.extend(("layers", via_layers))
                primitive_rows.append(tuple(row))
            except Exception:
                continue
        if eco1_layer_id is not None:
            drawing_getters = ("GetShape", "GetStart", "GetEnd", "GetCenter", "GetPosition")
            for drawing in drawings:
                done_items += 1
                if progress_callback is not None and (done_items & 63) == 0:
                    _emit_snapshot_progress("Scanning board drawings...")
                try:
                    layer = int(drawing.GetLayer())
                    if layer != eco1_layer_id:
                        continue
                    rect = drawing.GetBoundingBox()
                    drawing_row = [
                        "drawing",
                        type(drawing).__name__,
                        layer,
                        rect.GetX(),
                        rect.GetY(),
                        rect.GetWidth(),
                        rect.GetHeight(),
                    ]
                    for getter in drawing_getters:
                        fn = getattr(drawing, getter, None)
                        if fn is not None:
                            try:
                                value = fn()
                                if hasattr(value, "x") and hasattr(value, "y"):
                                    drawing_row.extend((getter, int(value.x), int(value.y)))
                                else:
                                    drawing_row.extend((getter, str(value)))
                            except Exception:
                                continue
                    primitive_rows.append(tuple(drawing_row))
                except Exception:
                    continue
        zone_getters = ("GetNetCode", "GetNetname", "GetAssignedPriority", "IsFilled")
        for zone in zones:
            done_items += 1
            if progress_callback is not None and (done_items & 15) == 0:
                _emit_snapshot_progress("Scanning copper zones...")
            try:
                rect = zone.GetBoundingBox()
                zone_row = [
                    type(zone).__name__, int(zone.GetLayer()),
                    rect.GetX(), rect.GetY(), rect.GetWidth(), rect.GetHeight(),
                ]
                get_layer_set = getattr(zone, "GetLayerSet", None)
                if get_layer_set is not None:
                    try:
                        zone_row.extend(
                            ("layers", tuple(int(value) for value in get_layer_set().IntSeq()))
                        )
                    except Exception:
                        pass
                for getter in zone_getters:
                    fn = getattr(zone, getter, None)
                    if fn is not None:
                        try:
                            zone_row.extend((getter, str(fn())))
                        except Exception:
                            continue
                primitive_rows.append(tuple(zone_row))
            except Exception:
                continue
        pad_getters = (
            "GetShape",
            "GetSize",
            "GetOrientationDegrees",
            "GetShapePos",
            "GetOffset",
            "GetAttribute",
            "GetDrillSize",
        )
        for fp in footprints:
            done_items += 1
            if progress_callback is not None and (done_items & 31) == 0:
                _emit_snapshot_progress("Scanning footprints and pads...")
            try:
                reference = fp.GetReference()
            except Exception:
                reference = ""
            for pad in fp.Pads():
                try:
                    pos = pad.GetPosition()
                    rect = pad.GetBoundingBox()
                    row = [
                        "pad", reference, pad.GetNumber(), pos.x, pos.y,
                        pad.GetLayer(), rect.GetX(), rect.GetY(),
                        rect.GetWidth(), rect.GetHeight(),
                    ]
                    pad_layers = _extract_layer_set_tuple(pad)
                    if pad_layers is not None:
                        row.extend(("layers", pad_layers))
                    for getter in pad_getters:
                        fn = getattr(pad, getter, None)
                        if fn is not None:
                            try:
                                value = fn()
                                if hasattr(value, "x") and hasattr(value, "y"):
                                    row.extend((getter, int(value.x), int(value.y)))
                                else:
                                    row.extend((getter, str(value)))
                            except Exception:
                                continue
                    primitive_rows.append(tuple(row))
                except Exception:
                    continue
        _emit_snapshot_progress("Board snapshot captured.", force=True)
        bbox_mm = (
            bbox.GetX() * 1e-6,
            bbox.GetY() * 1e-6,
            bbox.GetWidth() * 1e-6,
            bbox.GetHeight() * 1e-6,
        )
        filename = str(board.GetFileName() or "")
        try:
            file_identity = (
                os.path.getsize(filename),
                os.stat(filename).st_mtime_ns,
            )
        except OSError:
            file_identity = None
        return BoardSnapshot(
            filename=filename,
            fingerprint=stable_fingerprint(
                (bbox_mm, file_identity, tuple(primitive_rows))
            ),
            bbox_mm=bbox_mm,
            copper_layers=tuple(int(x) for x in copper_ids),
            track_count=len(tracks),
            footprint_count=len(footprints),
            zone_count=len(zones),
        )

    def _geometry_key(self, board, copper_ids, bbox, grid, settings, pads, progress_callback=None):
        snapshot = self._capture_board_snapshot(
            board, copper_ids, bbox, progress_callback=progress_callback
        )
        # Bump pad_raster_v to invalidate stale v0.4.4 .npz caches on upgrade.
        pad_keys = [("pad_raster_v", 2)]
        for pad in pads or []:
            try:
                pos = pad.GetPosition()
                primary_lid = get_pad_primary_layer_id(pad, copper_ids or None)
                pad_keys.append((pad.GetNumber(), primary_lid, pos.x, pos.y))
            except Exception:
                pad_keys.append(id(pad))
        return geometry_cache_key(snapshot, grid, settings, pad_keys)

    def _settings_path(self):
        """Return path to settings persistence file."""
        try:
            base_dir = wx.StandardPaths.Get().GetUserConfigDir()
        except Exception:
            base_dir = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
        return os.path.join(base_dir, "ThermalSim", "thermal_sim_last_settings.json")

    def _load_settings(self, path=None):
        """Load settings from a JSON file."""
        settings_path = path or self._settings_path()
        if path is None and not os.path.isfile(settings_path):
            legacy_path = os.path.join(os.path.dirname(__file__), "thermal_sim_last_settings.json")
            if os.path.isfile(legacy_path):
                try:
                    with open(legacy_path, "r", encoding="utf-8") as f:
                        legacy = json.load(f)
                    if isinstance(legacy, dict):
                        legacy["schema_version"] = 2
                        self._save_settings(legacy)
                except Exception:
                    pass
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_settings(self, settings, path=None):
        """Save settings to a JSON file."""
        settings_path = path or self._settings_path()
        try:
            parent_dir = os.path.dirname(settings_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            payload = dict(settings)
            payload["schema_version"] = 2
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            return True
        except Exception:
            return False

    def Run(self):
        """Plugin entry point with error handling."""
        try:
            self.RunSafe()
        except Exception:
            wx.MessageBox(traceback.format_exc(), "Thermal Sim Error")
        finally:
            self._close_startup_progress()

    def RunSafe(self):
        """Main plugin execution logic."""
        board = pcbnew.GetBoard()
        # Capture the editor before a progress/dependency dialog can become
        # wx's active window and obscure the actual plugin host.
        self.host_window = _find_pcb_editor_parent(board)

        if not HAS_LIBS:
            from .capabilities import get_missing_packages
            from .dependency_installer import DependencyInstallDialog
            missing = get_missing_packages()
            if missing:
                dlg = DependencyInstallDialog(self.host_window, missing)
                dlg.ShowModal()
                dlg.Destroy()
            else:
                wx.MessageBox(
                    "All packages appear installed. Please restart KiCad.",
                    "ThermalSim"
                )
            return

        self._show_startup_progress(5, "Reading active PCB...")

        # Zone filling is intentionally not performed here. It can take
        # minutes on large boards and some KiCad builds let C++ map lookup
        # exceptions escape the Python boundary. Preview/Run validate the
        # existing fills and ask the user to press B when needed.
        zone_refill_s = 0.0

        # --- 1. Layer Detection ---
        self._show_startup_progress(20, "Detecting copper layers...")
        copper_ids, layer_names = self._detect_copper_layers(board)
        self._show_startup_progress(35, "Reading PCB stackup...")
        stack_info = parse_stackup_from_board_file(board)

        # Use stackup order if available
        copper_ids_stack = stack_info.get("copper_ids") if isinstance(stack_info, dict) else None
        detected_names = dict(zip(copper_ids, layer_names))
        valid_stack_ids = [
            layer_id for layer_id in (copper_ids_stack or [])
            if layer_id in detected_names
        ]
        if len(valid_stack_ids) >= 2:
            copper_ids = valid_stack_ids
            layer_names = [detected_names[layer_id] for layer_id in copper_ids]

        # --- 2. Auto-Resolution ---
        self._show_startup_progress(50, "Measuring board dimensions...")
        try:
            bbox = board.GetBoundingBox()
        except:
            bbox = board.ComputeBoundingBox(True)
        w_mm = bbox.GetWidth() * 1e-6
        h_mm = bbox.GetHeight() * 1e-6
        suggested_res = self._calculate_suggested_resolution(w_mm, h_mm, len(copper_ids))

        # --- 3. Initial Pad Selection ---
        self._show_startup_progress(65, "Reading selected heat-source pads...")
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
        self._show_startup_progress(78, "Preparing simulation settings...")
        stackup_details = format_stackup_report_um(stack_info) if stack_info else ""
        pad_names = self._format_pad_names(selected_pads)
        initial_power_pads = self._descriptors_from_selected_pads(
            board, selected_pads
        )
        board_path = str(board.GetFileName() or "")
        board_dir = os.path.dirname(board_path) if board_path else ""
        default_output_dir = (
            os.path.join(board_dir, "ThermalSim_results")
            if board_dir else os.path.join(os.path.expanduser("~"), "Documents", "ThermalSim_results")
        )
        last_settings = self._load_settings()
        if last_settings.get("output_dir") and os.path.isdir(last_settings.get("output_dir")):
            default_output_dir = last_settings.get("output_dir")

        if self.settings_dialog is not None:
            try:
                self._close_startup_progress()
                self.settings_dialog.Raise()
                return
            except Exception:
                self.settings_dialog = None

        def selection_provider():
            return self._get_selected_pad_descriptors(board)

        def run_callback(settings):
            self._save_settings(settings)
            pad_lookup = self._maybe_build_pad_lookup(board, settings)
            power_pads = self._resolve_power_pad_objects(
                board, settings, legacy_pads=pads_list, pad_lookup=pad_lookup
            )
            current_pads = self._resolve_current_pad_objects(
                board, settings, pad_lookup=pad_lookup
            )
            focus_pads = self._unique_pads(power_pads + current_pads)
            try:
                self._run_simulation(
                    board, copper_ids, layer_names, bbox, pads_list,
                    settings, stack_info, pad_names, zone_refill_s=zone_refill_s,
                    focus_pads=focus_pads
                )
            except Exception:
                error_text = traceback.format_exc()
                if self.settings_dialog is not None:
                    try:
                        self.settings_dialog.set_run_state(
                            "failed",
                            "The simulation failed. See the error dialog for details.",
                        )
                    except Exception:
                        pass
                wx.MessageBox(error_text, "Thermal Sim Error")

        def close_callback():
            self.settings_dialog = None
            self.operator_cache.clear()
            self.factorization_cache.clear()
            self.electrical_cache.clear()
            self.geometry_cache.key = None
            self.geometry_cache.value = None

        def preflight_callback(settings):
            return self._preflight(board, copper_ids, bbox, settings)

        self._show_startup_progress(92, "Creating ThermalSim dialog...")
        stackup_thickness_mm = (
            stack_info.get("board_thickness_mm")
            if isinstance(stack_info, dict)
            else None
        )
        dlg = SettingsDialog(
            self.host_window, len(pads_list), suggested_res, layer_names,
            preview_callback=self.generate_preview,
            selection_provider=selection_provider,
            run_callback=run_callback,
            close_callback=close_callback,
            preflight_callback=preflight_callback,
            load_settings_callback=self._load_settings,
            save_settings_callback=self._save_settings,
            stackup_details=stackup_details,
            pad_names=pad_names,
            initial_power_pads=initial_power_pads,
            default_output_dir=default_output_dir,
            defaults=last_settings,
            board_name=os.path.basename(board_path) if board_path else "Unsaved board",
            board_size_mm=(w_mm, h_mm),
            defer_initial_preflight=True,
            stackup_thickness_mm=stackup_thickness_mm,
        )
        self.settings_dialog = dlg
        self._close_startup_progress()
        try:
            dlg.Show()
        except Exception:
            try:
                if dlg.ShowModal() == wx.ID_OK:
                    settings = dlg.get_values()
                    if settings:
                        self._save_settings(settings)
                        pad_lookup = self._maybe_build_pad_lookup(board, settings)
                        power_pads = self._resolve_power_pad_objects(
                            board, settings, legacy_pads=pads_list, pad_lookup=pad_lookup
                        )
                        current_pads = self._resolve_current_pad_objects(
                            board, settings, pad_lookup=pad_lookup
                        )
                        focus_pads = self._unique_pads(power_pads + current_pads)
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
        return calculate_suggested_resolution(w_mm, h_mm, layer_count)

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

    def _pad_layer_name(self, board, pad, copper_ids=None):
        """Return the KiCad layer name for a pad."""
        c_ids = copper_ids if copper_ids else (getattr(self, "copper_ids", None) or None)
        lid = get_pad_primary_layer_id(pad, c_ids)
        try:
            return board.GetLayerName(lid)
        except Exception:
            return str(lid)

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

    def _descriptors_from_selected_pads(self, board, selected_pads):
        """Build descriptors without scanning every footprint a second time."""
        descriptors = []
        for name, pad in selected_pads or []:
            try:
                number = str(pad.GetNumber())
                suffix = f"-{number}"
                fp_ref = name[:-len(suffix)] if name.endswith(suffix) else name
                descriptors.append(self._pad_descriptor(board, fp_ref, pad))
            except Exception:
                continue
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

    def _maybe_build_pad_lookup(self, board, settings):
        """Build a shared pad lookup once when settings require pad resolution."""
        cfg = settings or {}
        needs_power = bool(cfg.get("power_pads"))
        needs_current = bool(cfg.get("current_enabled"))
        if not needs_power and not needs_current:
            return None
        return self._build_pad_lookup(board)

    def _resolve_power_pad_entries(self, board, settings, pad_lookup=None):
        """Resolve manual power-pad settings to live pad objects and power strings."""
        raw_power_pads = prepare_power_pads(
            settings.get("power_pads", []),
            settings.get("power_str", "1.0"),
        )
        if not raw_power_pads:
            return [], []
        by_key, by_name = (
            pad_lookup if pad_lookup is not None else self._build_pad_lookup(board)
        )
        entries = []
        missing = []
        for pad_info in raw_power_pads:
            match = by_key.get(pad_info.get("pad_key")) or by_name.get(pad_info.get("name"))
            if not match:
                missing.append(pad_info.get("name", pad_info.get("pad_key", "<unknown>")))
                continue
            descriptor, pad = match
            merged = dict(descriptor)
            merged["power"] = str(pad_info.get("power", "0.0")).strip()
            entries.append((merged, pad, merged["power"]))
        return entries, missing

    def _resolve_power_pad_objects(self, board, settings, legacy_pads=None, pad_lookup=None):
        """Resolve manual power pads to live pad objects for preview/focus area."""
        if "power_pads" not in (settings or {}):
            return list(legacy_pads or [])
        entries, _ = self._resolve_power_pad_entries(board, settings, pad_lookup=pad_lookup)
        return self._unique_pads([pad for _, pad, _ in entries])

    def _resolve_current_pad_objects(self, board, settings, pad_lookup=None):
        """Resolve current-group pad descriptors to live pad objects."""
        if not settings.get("current_enabled"):
            return []
        by_key, by_name = (
            pad_lookup if pad_lookup is not None else self._build_pad_lookup(board)
        )
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

    def _resolve_current_terminals(self, board, settings, pad_lookup=None):
        """Resolve current-group settings into CurrentTerminal objects."""
        terminals = []
        missing = []
        if not settings.get("current_enabled"):
            return terminals, missing
        by_key, by_name = (
            pad_lookup if pad_lookup is not None else self._build_pad_lookup(board)
        )
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

    def _preflight(self, board, copper_ids, bbox, settings):
        """Validate settings and estimate the exact grid before execution."""
        result = PreflightResult()
        pad_lookup = self._maybe_build_pad_lookup(board, settings)
        power_entries, missing_power = self._resolve_power_pad_entries(
            board, settings, pad_lookup=pad_lookup
        )
        power_pads = [pad for _, pad, _ in power_entries]
        terminals = []
        missing_current = []
        if settings.get("current_enabled"):
            terminals, missing_current = self._resolve_current_terminals(
                board, settings, pad_lookup=pad_lookup
            )
        current_pads = [item.pad for item in terminals]
        focus_pads = self._unique_pads(power_pads + current_pads)
        result.area = _estimate_simulation_area(
            board, bbox, settings, power_pads=power_pads, terminals=terminals
        )
        result.grid = _estimate_solver_grid(
            bbox, float(settings.get("res", 0.5)), settings, len(copper_ids),
            focus_pads, area=result.area,
        )
        result.warnings.extend(result.area.warnings)

        if missing_power:
            result.errors.append(f"{len(missing_power)} heat-source pad(s) are missing from the board.")
        has_power = False
        has_pwl = False
        for _, _, entry in power_entries:
            value = str(entry or "").strip()
            if not value:
                continue
            try:
                has_power = has_power or abs(float(value)) > 0.0
            except ValueError:
                has_pwl = True
                try:
                    _, pwl_values = parse_pwl_file(value)
                    has_power = has_power or bool(np.any(np.asarray(pwl_values) != 0.0))
                except Exception as exc:
                    result.errors.append(f"Invalid PWL source '{value}': {exc}")

        if settings.get("steady_state") and has_pwl:
            result.warnings.append(
                "Steady-state mode ignores PWL time variation; "
                "disable Steady-state for time-varying PWL profiles."
            )

        has_current = False
        if settings.get("current_enabled"):
            if missing_current:
                result.errors.append(f"{len(missing_current)} current-terminal pad(s) are missing from the board.")
            has_current = any(abs(float(item.current_a)) > 0.0 for item in terminals)
            totals = {}
            for item in terminals:
                net = item.net_name or f"net:{item.net_code}"
                totals[net] = totals.get(net, 0.0) + float(item.current_a)
            unbalanced = [name for name, total in totals.items() if abs(total) > 1e-9]
            if unbalanced:
                result.errors.append("Current is not balanced for: " + ", ".join(unbalanced[:3]))

        if not has_power and not has_current:
            result.errors.append("Configure at least one non-zero heat source or balanced current path.")

        output_dir = str(settings.get("output_dir", "") or "").strip()
        writable_parent = output_dir
        while writable_parent and not os.path.exists(writable_parent):
            parent = os.path.dirname(writable_parent)
            if parent == writable_parent:
                break
            writable_parent = parent
        if not output_dir or not writable_parent or not os.access(writable_parent, os.W_OK):
            result.errors.append("The output folder is not writable.")

        if result.grid.auto_coarsened:
            result.warnings.append(
                f"Resolution will be coarsened from {result.grid.requested_res_mm:.3f} "
                f"to {result.grid.actual_res_mm:.3f} mm."
            )
        if result.grid.nodes > 10_000_000:
            result.warnings.append(
                f"The requested grid contains {result.grid.nodes:,} equivalent "
                "uniform nodes. Adaptive reduction depends on the actual PCB "
                "geometry; verify the displayed RAM estimate before running."
            )
        return result

    def _derive_stackup_thicknesses(self, board, copper_ids, stack_info, settings):
        """Derive thickness values from stackup or defaults."""
        return derive_stackup_thicknesses(board, copper_ids, stack_info, settings)

    def generate_preview(self, settings, layer_names):
        """Generate geometry preview image."""
        def preview_status(message):
            if self.settings_dialog is None:
                return
            try:
                self.settings_dialog.lbl_preflight.SetLabel(message)
                app = wx.GetApp()
                if app is not None:
                    app.Yield()
            except Exception:
                pass

        preview_status("Checking existing KiCad copper-zone fills...")
        if not self._require_filled_zones(self.board):
            return None

        reporter = _UiProgressReporter(
            title="ThermalSim - Preview",
            initial_message="Resolving heat sources and simulation area...",
            parent=self.settings_dialog or self.host_window,
        )
        self.cancel_token = reporter.cancel_token
        try:
            if not reporter.update(2, "Resolving heat sources and simulation area...", force=True):
                raise GeometryCancelledError("Preview cancelled.")
            preview_status("Resolving heat sources and simulation area...")
            pad_lookup = self._maybe_build_pad_lookup(self.board, settings)
            power_pads = self._resolve_power_pad_objects(
                self.board, settings, legacy_pads=self.pads_list, pad_lookup=pad_lookup
            )
            terminals, _ = self._resolve_current_terminals(
                self.board, settings, pad_lookup=pad_lookup
            )
            current_pads = [item.pad for item in terminals]
            preview_pads = self._unique_pads(power_pads + current_pads)
            effective_settings = dict(settings)
            area = _estimate_simulation_area(
                self.board, self.bbox, effective_settings,
                power_pads=power_pads, terminals=terminals,
            )
            effective_settings["_preview_area_summary"] = (
                f"{area.width_mm:.1f} x {area.height_mm:.1f} mm / "
                f"{area.area_fraction * 100.0:.0f}% of board"
            )
            effective_settings["_preview_area_limited"] = area.limited
            grid = _estimate_solver_grid(
                self.bbox, float(effective_settings["res"]), effective_settings,
                len(self.copper_ids), preview_pads, area=area,
            )
            cache_key = self._geometry_key(
                self.board,
                self.copper_ids,
                self.bbox,
                grid,
                effective_settings,
                preview_pads,
                progress_callback=reporter.make_stage_callback(
                    2, 7, "Capturing board snapshot..."
                ),
            )
            geometry_state = reporter.run_in_background(
                lambda: self.geometry_cache.get(cache_key),
                7,
                10,
                "Checking geometry cache...",
            )
            if reporter.cancelled:
                raise GeometryCancelledError("Preview cancelled.")
            if geometry_state is None:
                preview_status("Rasterizing exact tracks, pads, vias, and zones...")
                if not reporter.update(10, "Mapping exact PCB geometry...", force=True):
                    raise GeometryCancelledError("Preview cancelled.")
                via_plating_um, via_fill = resolve_via_settings(effective_settings)
                geometry_state = _call_with_optional_progress_kwargs(
                    build_geometry_state,
                    board=self.board,
                    copper_ids=self.copper_ids,
                    rows=grid.rows,
                    cols=grid.cols,
                    x_min=grid.x_min_mm,
                    y_min=grid.y_min_mm,
                    res=grid.actual_res_mm,
                    settings=effective_settings,
                    via_factor=390.0 / 0.3,
                    pads_list=preview_pads,
                    progress_callback=reporter.make_stage_callback(
                        10, 70, "Rasterizing board geometry..."
                    ),
                    cancel_check=reporter.make_cancel_check(),
                    optional_kwargs={
                        "via_plating_um": via_plating_um,
                        "via_fill": via_fill,
                        "k_cu": 390.0,
                        "k_fr4": 0.3,
                    },
                )
                if reporter.cancelled:
                    raise GeometryCancelledError("Preview cancelled.")
                reporter.run_in_background(
                    lambda: self.geometry_cache.put(cache_key, geometry_state),
                    70,
                    75,
                    "Caching board geometry...",
                )
                if reporter.cancelled:
                    raise GeometryCancelledError("Preview cancelled.")

            preview_status("Rendering preview layers...")
            if not reporter.update(85, "Rendering preview layers...", force=True):
                raise GeometryCancelledError("Preview cancelled.")
            output_file = save_preview_image(
                self.board, self.copper_ids, self.bbox, preview_pads,
                effective_settings, layer_names,
                self.stack_info if self.stack_info is not None else parse_stackup_from_board_file(self.board),
                get_pad_pixels,
                create_multilayer_maps,
                self._derive_stackup_thicknesses,
                open_file=True,
                geometry_state=geometry_state,
                grid_spec=grid,
            )
            reporter.update(100, "Done", force=True)
            reporter.close()
            if not output_file:
                wx.MessageBox("Board data missing for preview", "Error")
            return output_file
        except GeometryCancelledError:
            reporter.close()
            preview_status("Preview cancelled.")
            if self.settings_dialog is not None and hasattr(self.settings_dialog, "_set_footer_status"):
                try:
                    self.settings_dialog._set_footer_status("Cancelled", "Preview cancelled.")
                except Exception:
                    pass
            return None
        finally:
            self.cancel_token = None
            reporter.close()

    def _run_simulation(self, board, copper_ids, layer_names, bbox, pads_list,
                        settings, stack_info, pad_names, zone_refill_s=0.0,
                        focus_pads=None):
        """Execute the thermal simulation."""
        run_started_at = time.perf_counter()

        def set_dialog_state(status, message=""):
            if self.settings_dialog is not None:
                try:
                    wx.CallAfter(self.settings_dialog.set_run_state, status, message)
                except Exception:
                    pass

        set_dialog_state("running", "Checking existing KiCad copper-zone fills...")
        if not self._require_filled_zones(board):
            set_dialog_state(
                "failed",
                "Copper zones must be refilled in KiCad before simulation.",
            )
            return

        # Output folder setup
        base_output_dir = settings.get('output_dir') or os.path.dirname(__file__)
        try:
            os.makedirs(base_output_dir, exist_ok=True)
            run_dir = os.path.join(base_output_dir, time.strftime("Thermalsim_%Y%m%d_%H%M%S"))
            os.makedirs(run_dir, exist_ok=True)
            test_path = os.path.join(run_dir, ".write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
        except Exception:
            run_dir = tempfile.mkdtemp(prefix="ThermalSim_")
        _write_run_manifest(run_dir, "running", settings=settings)

        reporter = _UiProgressReporter(
            title="ThermalSim",
            initial_message="Preparing thermal simulation...",
            parent=self.settings_dialog or self.host_window,
        )
        cancel_token = reporter.cancel_token
        self.cancel_token = cancel_token

        try:
            focus_pads = focus_pads if focus_pads is not None else pads_list
            pad_lookup = self._maybe_build_pad_lookup(board, settings)
            area_power_pads = self._resolve_power_pad_objects(
                board, settings, legacy_pads=pads_list, pad_lookup=pad_lookup
            )
            area_terminals = []
            if settings.get("current_enabled"):
                area_terminals, _ = self._resolve_current_terminals(
                    board, settings, pad_lookup=pad_lookup
                )
            area = _estimate_simulation_area(
                board, bbox, settings,
                power_pads=area_power_pads,
                terminals=area_terminals,
            )
            # Derive thicknesses
            stackup_derived = self._derive_stackup_thicknesses(board, copper_ids, stack_info, settings)
            total_thick_mm = stackup_derived["total_thick_mm_used"]
            init_timings = {"zone_refill_s": float(zone_refill_s)}

            requested_res = float(settings['res'])
            layer_count = len(copper_ids)
            grid = _estimate_solver_grid(
                bbox, requested_res, settings, layer_count, focus_pads, area=area
            )
            res = grid.actual_res_mm
            x_min = grid.x_min_mm
            y_min = grid.y_min_mm
            rows = grid.rows
            cols = grid.cols
            grid_info = {
                "grid_requested_res_mm": requested_res,
                "grid_res_mm": float(res),
                "grid_auto_coarsened": grid.auto_coarsened,
                "grid_expert_limits": grid.expert_limits,
                "grid_max_cells": grid.max_cells,
                "grid_target_cells": grid.target_cells,
                "grid_rows": int(rows),
                "grid_cols": int(cols),
                "grid_x_min_mm": float(x_min),
                "grid_y_min_mm": float(y_min),
                "grid_width_mm": float(cols * res),
                "grid_height_mm": float(rows * res),
                "grid_cell_area_mm2": float(res * res),
                "grid_detail_level": grid.detail_level,
                "grid_node_budget": grid.node_budget,
                "grid_memory_mb_low": grid.memory_mb_low,
                "grid_memory_mb_high": grid.memory_mb_high,
                "grid_runtime_class": grid.runtime_class,
                "area_mode": area.mode,
                "area_fraction": area.area_fraction,
                "area_margin_mm": area.margin_mm,
                "area_active_nets": list(area.active_net_names),
            }

            # Physical parameters
            k_fr4_rel = 1.0
            via_factor = 390.0 / 0.3
            k_cu = 390.0
            k_fr4 = 0.3
            rho_cu, cp_cu = 8960.0, 385.0
            rho_fr4, cp_fr4 = 1850.0, 1100.0
            via_plating_um, via_fill = resolve_via_settings(settings)
            h_top, h_bottom = resolve_convection_settings(settings)

            cu_thick_mm_used = stackup_derived["copper_thickness_mm_used"]
            gap_mm_used = stackup_derived["gap_mm_used"]
            cu_thick_m = [max(1e-9, th * 1e-3) for th in cu_thick_mm_used]
            gap_m = [max(1e-9, g * 1e-3) for g in gap_mm_used]

            # Build internal geometry state (0-36 %)
            set_dialog_state("running", "Mapping exact PCB geometry...")
            geometry_start = time.perf_counter()
            cache_key = self._geometry_key(
                board,
                copper_ids,
                bbox,
                grid,
                settings,
                focus_pads,
                progress_callback=reporter.make_stage_callback(
                    0, 3, "Capturing board snapshot..."
                ),
            )
            geometry_state = reporter.run_in_background(
                lambda: self.geometry_cache.get(cache_key),
                3,
                5,
                "Checking geometry cache...",
            )
            if reporter.cancelled:
                raise GeometryCancelledError("Simulation cancelled.")
            geometry_cache_hit = geometry_state is not None
            try:
                if geometry_state is None:
                    if not reporter.update(5, "Mapping exact PCB geometry...", force=True):
                        raise GeometryCancelledError("Simulation cancelled.")
                    geometry_state = _call_with_optional_progress_kwargs(
                        build_geometry_state,
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
                        progress_callback=reporter.make_stage_callback(
                            5, 32, "Mapping exact PCB geometry..."
                        ),
                        cancel_check=reporter.make_cancel_check(),
                        optional_kwargs={
                            "via_plating_um": via_plating_um,
                            "via_fill": via_fill,
                            "k_cu": k_cu,
                            "k_fr4": k_fr4,
                        },
                    )
                    if reporter.cancelled:
                        raise GeometryCancelledError("Simulation cancelled.")
                    reporter.run_in_background(
                        lambda: self.geometry_cache.put(cache_key, geometry_state),
                        32,
                        36,
                        "Caching board geometry...",
                    )
                    if reporter.cancelled:
                        raise GeometryCancelledError("Simulation cancelled.")
            except GeometryCancelledError:
                raise
            except Exception as e:
                reporter.close()
                _write_run_manifest(run_dir, "error", traceback=traceback.format_exc())
                wx.MessageBox(f"Error mapping geometry: {e}", "Error")
                set_dialog_state("failed", "Board geometry could not be mapped.")
                return
            init_timings["geometry_maps_s"] = time.perf_counter() - geometry_start
            init_timings["geometry_cache_hit"] = geometry_cache_hit
            copper_mask = geometry_state.copper_mask
            via_map_3d = getattr(geometry_state, "via_map_3d", None)
            has_via_map_3d = via_map_3d is not None
            V_map = via_map_3d if has_via_map_3d else geometry_state.via_map
            H_map = geometry_state.heatsink_mask

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

            # Effective FR4 thickness per copper-plane control volume.
            t_fr4_eff = _effective_fr4_control_volume_thicknesses(
                gap_m,
                total_thick_mm * 1e-3,
                layer_count,
            )
            t_fr4_eff_mm = (t_fr4_eff * 1e3).tolist()
            engine_setting = str(
                settings.get("compute_engine", "auto") or "auto"
            ).strip().lower()
            if engine_setting not in {"auto", "fast_native", "legacy"}:
                engine_setting = "auto"
            use_fast_engine = (
                engine_setting == "fast_native"
                or (engine_setting == "auto" and grid.nodes >= _auto_fast_engine_threshold())
            )
            resolved_engine = "fast_native" if use_fast_engine else "legacy"

            use_compact_uniform = (
                use_fast_engine
                and not has_via_map_3d
                and build_compact_structured_operator is not None
                and build_structured_operator is _ORIG_BUILD_STRUCTURED_OPERATOR
            )
            use_compact_path = use_compact_uniform

            operator_key = _build_operator_cache_key(
                cache_key,
                resolved_engine,
                t_cu,
                gap_m,
                t_fr4_eff,
                dx,
                amb,
                settings,
            )
            operator_cache_lookup_key = (
                f"{operator_key}:compact_uniform" if use_compact_uniform else operator_key
            )
            cached_operator = self.operator_cache.get(operator_cache_lookup_key)
            operator_cache_hit = cached_operator is not None

            # Heat capacity (deferred/omitted on zero-peak compact paths)
            set_dialog_state("running", "Preparing material capacities...")
            capacity_start = time.perf_counter()
            pad_cap_areal = float(settings.get('pad_cap_areal', 0.0) or 0.0)
            C = None
            K_matrix = None
            b = None
            hA = None
            if operator_cache_hit:
                C, K_matrix, b, hA = cached_operator
            elif not use_compact_path:
                C = reporter.run_in_background(
                    lambda: _build_dense_capacity(
                        layer_count,
                        rows,
                        cols,
                        copper_mask,
                        t_cu,
                        t_fr4_eff,
                        pixel_area,
                        rho_cu,
                        cp_cu,
                        rho_fr4,
                        cp_fr4,
                        pad_cap_areal,
                        H_map,
                    ),
                    36,
                    37,
                    "Preparing material capacities...",
                )
                if reporter.cancelled:
                    raise GeometryCancelledError("Simulation cancelled.")
            init_timings["capacity_build_s"] = time.perf_counter() - capacity_start
            init_timings["operator_cache_hit"] = operator_cache_hit

            # Power injection (supports constant values and PWL file paths)
            RC = rows * cols
            N = RC * layer_count
            power_start = time.perf_counter()

            has_power_pad_settings = "power_pads" in settings
            missing_power_pads = []
            if has_power_pad_settings:
                power_pad_entries, missing_power_pads = self._resolve_power_pad_entries(
                    board, settings, pad_lookup=pad_lookup
                )
                power_pads = [pad for _, pad, _ in power_pad_entries]
                power_pad_names = [descriptor.get("name", "") for descriptor, _, _ in power_pad_entries]
                entries = [power for _, _, power in power_pad_entries if str(power).strip()]
            else:
                power_pads = list(pads_list or [])
                power_pad_names = list(pad_names or [])
                entries = [x.strip() for x in settings.get('power_str', '').split(',') if x.strip()]
                if len(entries) == 1:
                    entries = entries * len(power_pads)

            if missing_power_pads:
                reporter.close()
                wx.MessageBox(
                    "Manual power references pads that were not found on the board:\n"
                    + "\n".join(str(name) for name in missing_power_pads),
                    "Power Pad Error"
                )
                set_dialog_state("failed", "One or more heat-source pads were not found.")
                return

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
                        reporter.close()
                        wx.MessageBox(
                            f"Error reading PWL file:\n{entry}\n\n{e}",
                            "PWL Error"
                        )
                        set_dialog_state("failed", "A PWL power profile could not be read.")
                        return

            if len(pad_sources) == 1 and len(power_pads) > 1:
                pad_sources = pad_sources * len(power_pads)
                entries = entries * len(power_pads)

            if len(pad_sources) != len(power_pads):
                wx.MessageBox(
                    f"Number of power entries ({len(pad_sources)}) does not match "
                    f"number of power pads ({len(power_pads)}).",
                    "Warning"
                )

            pad_contributions = _build_sparse_pad_contributions(
                board=board,
                copper_ids=copper_ids,
                pads_list=power_pads,
                rows=rows,
                cols=cols,
                x_min=x_min,
                y_min=y_min,
                res=res,
                area_mask=getattr(geometry_state, "area_mask", None),
            )
            has_pwl_sources = any(source_type == 'pwl' for source_type, _ in pad_sources)
            Q = None
            Q_func = None
            Q_sparse = None
            if not use_compact_path or has_pwl_sources or SparsePowerVector is None:
                Q, Q_func = _build_power_vector(pad_sources, pad_contributions, N)
            init_timings["power_vector_build_s"] = time.perf_counter() - power_start

            # Electrical solve (37-46 %)
            electrical_summary = None
            q_joule = None
            if settings.get("current_enabled"):
                set_dialog_state("running", "Solving active electrical paths...")
                electrical_start = time.perf_counter()
                electrical_cache_hit = False
                terminals, missing_pads = self._resolve_current_terminals(
                    board, settings, pad_lookup=pad_lookup
                )
                if missing_pads:
                    reporter.close()
                    wx.MessageBox(
                        "Current simulation references pads that were not found on the board:\n"
                        + "\n".join(str(name) for name in missing_pads),
                        "Current Path Error"
                    )
                    set_dialog_state("failed", "One or more current-terminal pads were not found.")
                    return
                if terminals:
                    electrical_config = call_with_supported_kwargs(
                        ElectricalConfig,
                        copper_ids=list(copper_ids),
                        rows=rows,
                        cols=cols,
                        x_min=x_min,
                        y_min=y_min,
                        res=res,
                        t_cu=np.asarray(t_cu, dtype=np.float64),
                        layer_names={lid: layer_names[idx] for idx, lid in enumerate(copper_ids) if idx < len(layer_names)},
                        optional_kwargs={
                            "via_plating_um": via_plating_um,
                            "via_fill": via_fill,
                            "ambient_c": float(amb),
                        },
                    )
                    electrical_key = stable_fingerprint({
                        "geometry": cache_key,
                        "copper_thickness_m": [float(value) for value in t_cu],
                        "via_plating_um": float(via_plating_um),
                        "via_fill": via_fill,
                        "ambient_c": float(amb),
                        "terminals": [
                            (item.name, item.net_name, int(item.net_code), float(item.current_a))
                            for item in terminals
                        ],
                    })
                    electrical_result = reporter.run_in_background(
                        lambda: self.electrical_cache.get(electrical_key),
                        37,
                        38,
                        "Checking electrical cache...",
                    )
                    if reporter.cancelled:
                        raise GeometryCancelledError("Simulation cancelled.")
                    electrical_cache_hit = electrical_result is not None
                    if electrical_result is None:
                        if not reporter.update(38, "Solving active electrical paths...", force=True):
                            raise GeometryCancelledError("Simulation cancelled.")
                        electrical_result = _call_with_optional_progress_kwargs(
                            solve_electrical_heating,
                            board,
                            terminals,
                            electrical_config,
                            progress_callback=reporter.make_stage_callback(
                                38, 46, "Solving active electrical paths..."
                            ),
                            cancel_check=reporter.make_cancel_check(),
                        )
                        if reporter.cancelled:
                            raise GeometryCancelledError("Simulation cancelled.")
                        if electrical_result.valid:
                            self.electrical_cache.put(electrical_key, electrical_result)
                    if not electrical_result.valid:
                        reporter.close()
                        wx.MessageBox(
                            "Current simulation validation failed:\n\n"
                            + "\n".join(electrical_result.errors),
                            "Current Path Error"
                        )
                        set_dialog_state("failed", "Current-path validation failed.")
                        return
                    q_joule = electrical_result.q_joule
                    if np.any(q_joule):
                        if Q is None:
                            Q, Q_func = _build_power_vector(pad_sources, pad_contributions, N)
                        Q = Q + q_joule
                        if Q_func is not None:
                            base_q_func = Q_func

                            def q_func_with_joule(t, _base=base_q_func, _q_joule=q_joule):
                                return _base(t) + _q_joule

                            Q_func = q_func_with_joule
                    electrical_summary = {
                        "total_loss_w": electrical_result.total_loss_w,
                        "warnings": electrical_result.warnings,
                        "nets": [_electrical_net_summary_dict(item) for item in electrical_result.net_summaries],
                    }
                else:
                    electrical_summary = {
                        "total_loss_w": 0.0,
                        "warnings": ["Current simulation enabled but no non-zero pad currents were configured."],
                        "nets": [],
                    }
                init_timings["electrical_solve_s"] = time.perf_counter() - electrical_start
                init_timings["electrical_cache_hit"] = electrical_cache_hit

            if Q is None and Q_func is None:
                if use_compact_path and SparsePowerVector is not None:
                    Q_sparse = _build_sparse_power_vector(pad_sources, pad_contributions, N)
                else:
                    Q, Q_func = _build_power_vector(pad_sources, pad_contributions, N)

            # Build pad_power for reporting
            pad_power = []
            for i, name in enumerate(power_pad_names):
                if i < len(pad_sources):
                    stype, sval = pad_sources[i]
                    if stype == 'const':
                        pad_power.append((name, sval))
                    else:
                        entry_label = entries[i] if i < len(entries) else ""
                        pad_power.append((name, f"PWL:{entry_label}"))
                else:
                    pad_power.append((name, None))

            # Build the thermal operator (46-54 %)
            set_dialog_state("running", "Building the thermal operator...")
            stiffness_start = time.perf_counter()
            if use_compact_uniform:
                if not operator_cache_hit:
                    K_matrix = reporter.run_in_background(
                        lambda: build_compact_structured_operator(
                            layer_count, rows, cols, copper_mask, t_cu, t_fr4_eff,
                            k_cu, k_fr4, dx, dy, V_map, gap_m, H_map, settings, amb,
                            rho_cu=rho_cu, cp_cu=cp_cu, rho_fr4=rho_fr4, cp_fr4=cp_fr4,
                            pad_cap_areal=pad_cap_areal,
                        ),
                        46,
                        54,
                        "Building thermal operator...",
                    )
                    if reporter.cancelled:
                        raise GeometryCancelledError("Simulation cancelled.")
                    self.operator_cache.put(operator_cache_lookup_key, (None, K_matrix, None, None))
                init_timings["stiffness_matrix_s"] = time.perf_counter() - stiffness_start
            else:
                if not operator_cache_hit:
                    H_map_float = H_map.astype(np.float64, copy=False)
                    builder = build_structured_operator if use_fast_engine else build_stiffness_matrix
                    K_matrix, b, hA, _ = reporter.run_in_background(
                        lambda: builder(
                            layer_count, rows, cols, copper_mask, t_cu, t_fr4_eff,
                            k_cu, k_fr4, dx, dy, V_map, gap_m, H_map_float, settings, amb
                        ),
                        46,
                        54,
                        "Building thermal operator...",
                    )
                    if reporter.cancelled:
                        raise GeometryCancelledError("Simulation cancelled.")
                    self.operator_cache.put(operator_cache_lookup_key, (C, K_matrix, b, hA))
                init_timings["stiffness_matrix_s"] = time.perf_counter() - stiffness_start
            init_timings["compute_engine"] = resolved_engine

            grid_info.update({
                "mesh_mode": "uniform",
                "equivalent_uniform_nodes": int(N),
            })
            if build_heatsink_model is not _ORIG_BUILD_HEATSINK_MODEL or not use_compact_path:
                H_map_float = H_map.astype(np.float64, copy=False)
                heatsink_model = build_heatsink_model(
                    layer_count, rows, cols, dx, dy, H_map_float, settings, amb
                )
            elif use_compact_uniform:
                heatsink_model = K_matrix.build_heatsink_model(settings, amb)
            else:
                heatsink_model = _build_compact_heatsink_model(
                    dx, dy, H_map, settings, amb
                )
            solver_heatsink_model = heatsink_model

            steady_state = bool(settings.get("steady_state", False)) and Q_func is None
            print(f"[ThermalSim] init timings: {_format_timing_summary(init_timings)}")
            solver_start_msg = (
                "Solving steady-state thermal model..."
                if steady_state
                else "Starting transient solver..."
            )
            set_dialog_state("running", solver_start_msg)

            # Snapshot configuration
            snap_times = []
            if settings.get('snapshots') and not steady_state:
                snap_count = max(1, min(50, int(settings.get('snap_count', 5))))
                snap_times = [sim_time * (k / (snap_count + 1)) for k in range(1, snap_count + 1)]
            snap_times = sorted({t for t in snap_times if 0.0 < t < sim_time})

            progress_state = {
                "current": 0,
                "total": 1,
                "message": "",
                "started_at": None,
            }

            def progress_callback(current, total, message=None):
                if progress_state["started_at"] is None:
                    progress_state["started_at"] = time.monotonic()
                progress_state["current"] = int(current)
                progress_state["total"] = max(1, int(total))
                if message is not None:
                    progress_state["message"] = str(message)
                return not cancel_token.cancelled

            def snapshot_callback(T_view, t_elapsed, snap_idx):
                return save_snapshot(T_view, H_map, amb, layer_names, snap_idx, t_elapsed, out_dir=run_dir)

            # Run solver (54-88 %)
            configured_time_stepping = settings.get("time_stepping", "auto")
            solver_config_fields = getattr(SolverConfig, "__dataclass_fields__", None)
            steady_state_kwargs = (
                {"steady_state": steady_state}
                if (solver_config_fields is None or "steady_state" in solver_config_fields)
                else {}
            )
            config = SolverConfig(
                sim_time=sim_time,
                amb=amb,
                dt_base=dt,
                steps_target=steps_target,
                use_multi_phase=True,
                time_stepping=configured_time_stepping,
                snapshots_enabled=bool(settings.get('snapshots', False)) and not steady_state,
                snap_times=snap_times,
                **steady_state_kwargs,
            )
            factorization_cache_key = _build_factorization_cache_key(
                operator_key, config, steady_state, settings
            )

            worker_result = {}

            def solver_worker():
                try:
                    if use_fast_engine:
                        solver_power = Q_sparse if Q_sparse is not None else Q
                        mf_kwargs = {}
                        if solver_heatsink_model is not None:
                            mf_kwargs["heatsink_model"] = solver_heatsink_model
                        worker_result["result"] = run_simulation_matrix_free(
                            config, K_matrix, C, solver_power,
                            b, hA,
                            progress_callback, snapshot_callback,
                            Q_func=Q_func,
                            cancel_check=lambda: cancel_token.cancelled,
                            **mf_kwargs,
                        )
                    else:
                        legacy_kwargs = {}
                        if heatsink_model is not None:
                            legacy_kwargs["heatsink_model"] = heatsink_model
                        worker_result["result"] = run_simulation(
                            config, K_matrix, C, Q, b, hA,
                            layer_count, rows, cols,
                            progress_callback, snapshot_callback,
                            Q_func=Q_func,
                            cancel_check=lambda: cancel_token.cancelled,
                            factorization_cache=self.factorization_cache,
                            factorization_cache_key=factorization_cache_key,
                            **legacy_kwargs,
                        )
                except Exception:
                    fast_traceback = traceback.format_exc()
                    if use_fast_engine and N <= 1_000_000 and not cancel_token.cancelled:
                        try:
                            print(
                                "[ThermalSim][WARN] Fast CPU solver failed; "
                                "retrying the safe legacy backend."
                            )
                            H_map_float = H_map.astype(np.float64, copy=False)
                            fallback_C = (
                                C
                                if C is not None
                                else _build_dense_capacity(
                                    layer_count,
                                    rows,
                                    cols,
                                    copper_mask,
                                    t_cu,
                                    t_fr4_eff,
                                    pixel_area,
                                    rho_cu,
                                    cp_cu,
                                    rho_fr4,
                                    cp_fr4,
                                    pad_cap_areal,
                                    H_map,
                                )
                            )
                            if Q is not None:
                                fallback_Q = Q
                                fallback_Q_func = Q_func
                            elif Q_sparse is not None:
                                fallback_Q = Q_sparse.to_dense()
                                fallback_Q_func = None
                            else:
                                fallback_Q, fallback_Q_func = _build_power_vector(
                                    pad_sources, pad_contributions, N
                                )
                            fallback_matrix, fallback_b, fallback_h_area, _ = (
                                build_stiffness_matrix(
                                    layer_count, rows, cols, copper_mask,
                                    t_cu, t_fr4_eff, k_cu, k_fr4,
                                    dx, dy, V_map, gap_m, H_map_float, settings, amb,
                                )
                            )
                            fallback_config = SolverConfig(
                                sim_time=sim_time,
                                amb=amb,
                                dt_base=dt,
                                steps_target=steps_target,
                                use_multi_phase=True,
                                time_stepping=settings.get("time_stepping", "auto"),
                                snapshots_enabled=(
                                    bool(settings.get("snapshots", False))
                                    and not steady_state
                                ),
                                snap_times=snap_times,
                                **steady_state_kwargs,
                            )
                            fallback_heatsink = (
                                heatsink_model
                                if (
                                    heatsink_model is not None
                                    and not getattr(heatsink_model, "bottom_layer_only", False)
                                )
                                else build_heatsink_model(
                                    layer_count, rows, cols, dx, dy, H_map_float, settings, amb
                                )
                            )
                            fallback_kwargs = {}
                            if fallback_heatsink is not None:
                                fallback_kwargs["heatsink_model"] = fallback_heatsink
                            worker_result["result"] = run_simulation(
                                fallback_config,
                                fallback_matrix,
                                fallback_C,
                                fallback_Q,
                                fallback_b,
                                fallback_h_area,
                                layer_count,
                                rows,
                                cols,
                                progress_callback,
                                snapshot_callback,
                                Q_func=fallback_Q_func,
                                cancel_check=lambda: cancel_token.cancelled,
                                **fallback_kwargs,
                            )
                            worker_result["result"].k_norm_info.update({
                                "compute_engine": "legacy_fallback",
                                "fast_engine_error": fast_traceback,
                            })
                            grid_info.update({
                                "mesh_mode": "uniform",
                                "equivalent_uniform_nodes": int(N),
                            })
                        except Exception:
                            worker_result["traceback"] = (
                                fast_traceback
                                + "\nLegacy fallback also failed:\n"
                                + traceback.format_exc()
                            )
                    else:
                        worker_result["traceback"] = fast_traceback

            def _solver_progress_message(state):
                msg = state.get("message")
                if msg:
                    return msg
                if steady_state:
                    return "Solving steady-state thermal model..."
                cur = int(state.get("current", 0))
                tot = max(1, int(state.get("total", 1)))
                text = f"Solving thermal model - step {cur}/{tot}"
                started_at = state.get("started_at")
                if started_at is not None and 3 <= cur < tot:
                    elapsed = time.monotonic() - float(started_at)
                    if elapsed >= 1.5:
                        eta_s = elapsed * float(tot - cur) / float(cur)
                        text += f" ({_format_eta_seconds(eta_s)})"
                return text

            try:
                reporter.run_in_background(
                    solver_worker,
                    54,
                    88,
                    solver_start_msg,
                    progress_state=progress_state,
                    message_fn=_solver_progress_message,
                )
            finally:
                if K_matrix is not None and hasattr(K_matrix, "clear_caches"):
                    K_matrix.clear_caches()
                clear_native_scratch()

            if "traceback" in worker_result:
                error_text = worker_result["traceback"]
                reporter.close()
                _write_run_manifest(run_dir, "error", traceback=error_text)
                wx.MessageBox(f"Solver failed:\n{error_text}", "Solver Error")
                set_dialog_state("failed", "The thermal solver failed. See the error dialog for details.")
                return
            result = worker_result["result"]

            if result.aborted or reporter.cancelled:
                raise GeometryCancelledError("Simulation cancelled.")

            # Add extra info to k_norm_info
            result.k_norm_info.update({
                **grid_info,
                "copper_threshold_rel": copper_threshold_rel,
                "t_fr4_eff_min": float(np.min(t_fr4_eff)),
                "t_fr4_eff_max": float(np.max(t_fr4_eff)),
                "t_fr4_eff_per_plane_mm": t_fr4_eff_mm,
                "pad_cap_input_areal": pad_cap_areal,
                "h_top": h_top,
                "h_air_bottom": h_bottom,
                "via_plating_um": via_plating_um,
                "via_fill": via_fill,
                "init_zone_refill_s": init_timings.get("zone_refill_s"),
                "init_geometry_maps_s": init_timings.get("geometry_maps_s"),
                "init_capacity_build_s": init_timings.get("capacity_build_s"),
                "init_power_vector_build_s": init_timings.get("power_vector_build_s"),
                "init_electrical_solve_s": init_timings.get("electrical_solve_s"),
                "init_electrical_cache_hit": init_timings.get("electrical_cache_hit"),
                "init_stiffness_matrix_s": init_timings.get("stiffness_matrix_s"),
                "init_adaptive_mesh_s": init_timings.get("adaptive_mesh_s"),
                "init_operator_cache_hit": init_timings.get("operator_cache_hit"),
                "compute_engine": result.k_norm_info.get(
                    "compute_engine", resolved_engine
                ),
                "process_working_set_mb": get_process_memory_mb(),
                "process_peak_working_set_mb": get_process_memory_mb(peak=True),
                "factorization_cache_hit": result.k_norm_info.get("factorization_cache_hit"),
                "electrical_summary": electrical_summary,
            })
            if settings.get("use_heatsink"):
                pad_sink_mode = str(
                    settings.get("pad_sink_mode", "fixed") or "fixed"
                ).strip().lower()
                if pad_sink_mode == "fixed":
                    pad_contact = settings.get("pad_contact_factor", 0.2)
                    result.k_norm_info.update({
                        "heatsink_mode": "fixed",
                        "pad_contact_factor": float(
                            0.2 if pad_contact is None else pad_contact
                        ),
                    })
                elif pad_sink_mode == "heatsink" and not np.any(H_map):
                    result.k_norm_info.update({
                        "heatsink_enabled": False,
                        "heatsink_mode": "heatsink",
                        "heatsink_inactive_reason": "No User.Eco1 thermal pad shapes found on board",
                    })

            # Save results & build HTML report (88-100 %)
            set_dialog_state("running", "Rendering layer heatmaps...")
            if not reporter.update(88, "Rendering layer heatmaps...", force=True):
                raise GeometryCancelledError("Simulation cancelled.")
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
            if reporter.cancelled:
                raise GeometryCancelledError("Simulation cancelled.")

            joule_map_path = None
            if q_joule is not None and np.any(q_joule):
                if not reporter.update(90, "Rendering Joule loss map...", force=True):
                    raise GeometryCancelledError("Simulation cancelled.")
                joule_map_path = save_joule_loss_map(
                    q_joule,
                    layer_count=layer_count,
                    rows=rows,
                    cols=cols,
                    layer_names=layer_names,
                    x_min_mm=x_min,
                    y_min_mm=y_min,
                    res_mm=res,
                    electrical_summary=electrical_summary,
                    out_dir=run_dir,
                )
                if reporter.cancelled:
                    raise GeometryCancelledError("Simulation cancelled.")

            if not reporter.update(92, "Generating board preview...", force=True):
                raise GeometryCancelledError("Simulation cancelled.")
            preview_path = save_preview_image(
                board, copper_ids, bbox, focus_pads,
                settings, layer_names, stack_info,
                get_pad_pixels, create_multilayer_maps,
                self._derive_stackup_thicknesses,
                open_file=False, out_dir=run_dir,
                geometry_state=geometry_state,
                grid_spec=grid,
            )
            if reporter.cancelled:
                raise GeometryCancelledError("Simulation cancelled.")

            interactive_heatmap = reporter.run_in_background(
                lambda: build_interactive_heatmap_payload(
                    result.T,
                    amb=amb,
                    layer_names=layer_names,
                    res_mm=res,
                    x_min_mm=x_min,
                    y_min_mm=y_min,
                    show_all=bool(settings.get('show_all', True))
                ),
                94,
                97,
                "Building interactive heatmap...",
            )
            if reporter.cancelled:
                raise GeometryCancelledError("Simulation cancelled.")

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
                "init_electrical_cache_hit": init_timings.get("electrical_cache_hit"),
                "init_stiffness_matrix_s": init_timings.get("stiffness_matrix_s"),
                "init_operator_cache_hit": init_timings.get("operator_cache_hit"),
                "factorization_cache_hit": result.k_norm_info.get("factorization_cache_hit"),
            }

            report_path = reporter.run_in_background(
                lambda: write_html_report(
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
                    electrical_summary=electrical_summary,
                    joule_map_path=joule_map_path
                ),
                97,
                99,
                "Writing HTML report...",
            )
            if reporter.cancelled:
                raise GeometryCancelledError("Simulation cancelled.")

            reporter.update(100, "Done", force=True)
            reporter.close()

            _write_run_manifest(
                run_dir,
                "success",
                report_path=report_path,
                heatmap_path=heatmap_path,
                preview_path=preview_path,
            )
            elapsed_s = time.perf_counter() - run_started_at
            max_temp_c = float(np.max(result.T))
            self.last_artifacts = SimulationArtifacts(
                report_path=report_path,
                preview_path=preview_path,
                heatmap_path=heatmap_path,
                run_dir=run_dir,
                status="success",
                elapsed_s=elapsed_s,
                max_temp_c=max_temp_c,
            )
            if self.settings_dialog is not None:
                try:
                    wx.CallAfter(
                        self.settings_dialog.set_artifacts,
                        report_path,
                        run_dir,
                        elapsed_s,
                        max_temp_c,
                    )
                except Exception:
                    pass

            # Open outputs
            if report_path:
                def _open_outputs():
                    try:
                        import webbrowser
                        webbrowser.open("file://" + os.path.abspath(report_path))
                    except Exception:
                        pass
                wx.CallAfter(_open_outputs)
        except GeometryCancelledError:
            reporter.close()
            _write_run_manifest(run_dir, "cancelled")
            set_dialog_state("cancelled")
            return
        finally:
            self.cancel_token = None
            reporter.close()
