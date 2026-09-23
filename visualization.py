"""
Visualization functions for ThermalSim.

This module provides Matplotlib-based plotting functions for thermal
simulation results and geometry previews.
"""

import os
import sys
import math
import json
import tempfile
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

import pcbnew


def build_interactive_heatmap_payload(
    T,
    amb,
    layer_names,
    res_mm,
    x_min_mm=0.0,
    y_min_mm=0.0,
    show_all=True,
    max_delta_c=250.0
):
    """
    Build a JSON-safe payload for the interactive HTML heatmap viewer.

    Parameters
    ----------
    T : np.ndarray
        Temperature array, shape (layers, rows, cols).
    amb : float
        Ambient temperature in degrees Celsius.
    layer_names : list of str
        Names for each layer.
    res_mm : float
        Grid resolution in millimeters.
    x_min_mm : float, optional
        Minimum x coordinate of the simulated area in millimeters.
    y_min_mm : float, optional
        Minimum y coordinate of the simulated area in millimeters.
    show_all : bool, optional
        Whether all layers are visible in the interactive viewer.
    max_delta_c : float, optional
        Maximum color scale range above ambient.

    Returns
    -------
    dict
        JSON-safe payload for the HTML report viewer.
    """
    T = np.asarray(T)
    if T.ndim != 3:
        raise ValueError("Temperature array T must have shape (layers, rows, cols)")

    layer_count, rows, cols = T.shape
    if show_all or layer_count <= 1:
        visible_indices = list(range(layer_count))
    else:
        visible_indices = sorted({0, layer_count - 1})

    finite_mask = np.isfinite(T)
    if np.any(finite_mask):
        vmax = float(np.max(T[finite_mask]))
    else:
        vmax = float(amb)
    vmax = min(vmax, float(amb) + float(max_delta_c))
    vmax = max(vmax, float(amb))

    def _layer_name(index):
        if index < len(layer_names):
            return str(layer_names[index])
        if index == 0:
            return "Top (F.Cu)"
        if index == layer_count - 1:
            return "Bottom (B.Cu)"
        return f"Inner {index}"

    def _json_value(value):
        if value is None or not np.isfinite(value):
            return None
        return round(float(value), 3)

    layers = []
    for index in visible_indices:
        layer = np.asarray(T[index])
        finite_layer = np.isfinite(layer)
        if np.any(finite_layer):
            layer_min = float(np.min(layer[finite_layer]))
            layer_max = float(np.max(layer[finite_layer]))
        else:
            layer_min = float(amb)
            layer_max = float(amb)
        flat_data = [_json_value(val) for val in layer.ravel(order='C')]
        layers.append({
            "index": int(index),
            "name": _layer_name(index),
            "rows": int(rows),
            "cols": int(cols),
            "min_c": round(layer_min, 3),
            "max_c": round(layer_max, 3),
            "data": flat_data,
        })

    return {
        "ambient_c": round(float(amb), 3),
        "vmin_c": round(float(amb), 3),
        "vmax_c": round(vmax, 3),
        "res_mm": round(float(res_mm), 6),
        "x_min_mm": round(float(x_min_mm), 6),
        "y_min_mm": round(float(y_min_mm), 6),
        "visible_layer_indices": visible_indices,
        "layers": layers,
    }


def save_stackup_plot(T, H, amb, layer_names, fname, t_elapsed=None):
    """
    Save a multi-layer temperature plot to file.

    Parameters
    ----------
    T : np.ndarray
        Temperature array, shape (layers, rows, cols).
    H : np.ndarray
        Heatsink mask, shape (rows, cols).
    amb : float
        Ambient temperature for color scale minimum.
    layer_names : list of str
        Names for each layer.
    fname : str
        Output filename.
    t_elapsed : float, optional
        Elapsed simulation time for title annotation.
    """
    vmax = np.max(T)
    if vmax > amb + 250:
        vmax = amb + 250

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
        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(12, 4 * rows_grid))
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
        if i >= len(axes):
            break
        ax = axes[i]
        name = labels[i]
        max_temp = np.max(T[i])
        if t_elapsed is not None:
            ax.set_title(f"{name} - t = {t_elapsed:.1f} s - Max: {max_temp:.1f}C")
        else:
            ax.set_title(f"{name} - Max: {max_temp:.1f}C")
        im = ax.imshow(
            T[i], cmap='inferno', origin='upper',
            vmin=amb, vmax=vmax, interpolation='bilinear'
        )
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        if i == count - 1 and np.max(H) > 0:
            ax.contour(H, levels=[0.5], colors='white', linewidths=2, linestyles='--')

    for j in range(count, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def save_snapshot(T, H, amb, layer_names, idx, t_elapsed, out_dir=None):
    """
    Save a time-series snapshot to file.

    Parameters
    ----------
    T : np.ndarray
        Temperature array, shape (layers, rows, cols).
    H : np.ndarray
        Heatsink mask.
    amb : float
        Ambient temperature.
    layer_names : list of str
        Layer names.
    idx : int
        Snapshot index number.
    t_elapsed : float
        Elapsed simulation time.
    out_dir : str, optional
        Output directory. Defaults to module directory.

    Returns
    -------
    str
        Path to saved snapshot file.
    """
    out_dir = out_dir or os.path.dirname(__file__)
    try:
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"snap_{idx:02d}_t{t_elapsed:.1f}.png")
        save_stackup_plot(T, H, amb, layer_names, fname, t_elapsed=t_elapsed)
        return fname
    except Exception:
        tmp = tempfile.gettempdir()
        fname = os.path.join(tmp, f"snap_{idx:02d}_t{t_elapsed:.1f}.png")
        save_stackup_plot(T, H, amb, layer_names, fname, t_elapsed=t_elapsed)
        return fname


def save_joule_loss_map(
    q_joule,
    layer_count,
    rows,
    cols,
    layer_names,
    x_min_mm=0.0,
    y_min_mm=0.0,
    res_mm=1.0,
    electrical_summary=None,
    out_dir=None,
):
    """
    Save a per-layer Joule-loss map for current-path diagnostics.

    Parameters
    ----------
    q_joule : np.ndarray
        Flattened Joule heat source vector in watts per thermal node.
    layer_count : int
        Number of copper layers.
    rows, cols : int
        Grid dimensions.
    layer_names : list of str
        Names for each layer.
    x_min_mm, y_min_mm : float, optional
        Grid origin in millimeters.
    res_mm : float, optional
        Grid resolution in millimeters.
    electrical_summary : dict, optional
        Current-path diagnostics containing terminal positions.
    out_dir : str, optional
        Output directory.

    Returns
    -------
    str or None
        Path to saved image, or None when no Joule data is available.
    """
    q_arr = np.asarray(q_joule, dtype=np.float64)
    expected = int(layer_count) * int(rows) * int(cols)
    if q_arr.size != expected or expected <= 0:
        return None
    q_layers = q_arr.reshape((layer_count, rows, cols))
    if not np.any(np.isfinite(q_layers)) or float(np.nanmax(q_layers)) <= 0.0:
        return None

    out_dir = out_dir or os.path.dirname(__file__)
    output_file = os.path.join(out_dir, "joule_loss_map.png")
    vmax = float(np.nanmax(q_layers))
    vmax = max(vmax, 1e-18)

    count = int(layer_count)
    if count == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    elif count == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = axes.flatten()
    else:
        cols_grid = 2
        rows_grid = math.ceil(count / 2)
        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(12, 4 * rows_grid))
        axes = axes.flatten()

    terminals = []
    for net in (electrical_summary or {}).get("nets", []) or []:
        terminals.extend(net.get("terminal_diagnostics", []) or [])

    for i in range(count):
        ax = axes[i]
        name = layer_names[i] if i < len(layer_names) else f"Layer {i}"
        layer = q_layers[i]
        im = ax.imshow(
            layer,
            cmap="magma",
            origin="upper",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        max_mw = float(np.nanmax(layer)) * 1e3 if np.any(np.isfinite(layer)) else 0.0
        ax.set_title(f"{name} Joule Loss - Max: {max_mw:.3f} mW/cell")
        for term in terminals:
            term_layer = str(term.get("layer", ""))
            if term_layer not in (name, "All copper (PTH)", "All copper"):
                continue
            try:
                col = (float(term["x_mm"]) - float(x_min_mm)) / float(res_mm)
                row = (float(term["y_mm"]) - float(y_min_mm)) / float(res_mm)
            except Exception:
                continue
            current = float(term.get("current_a", 0.0) or 0.0)
            marker = "^" if current >= 0.0 else "v"
            color = "#e31a1c" if current >= 0.0 else "#1f78b4"
            ax.scatter([col], [row], marker=marker, s=52, c=color, edgecolors="white", linewidths=0.8)
            ax.text(col + 1.5, row + 1.5, str(term.get("name", "")), color="white", fontsize=7)
        plt.colorbar(im, ax=ax, label="Cell loss (W)")
        ax.axis("off")

    for j in range(count, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    return output_file


def show_results_top_bot(T, H, amb, open_file=True, t_elapsed=None, out_dir=None,
                         steady_state=False):
    """
    Save and optionally display top/bottom layer temperature results.

    Parameters
    ----------
    T : np.ndarray
        Temperature array, shape (layers, rows, cols).
    H : np.ndarray
        Heatsink mask.
    amb : float
        Ambient temperature.
    open_file : bool, optional
        Whether to open the file in default viewer.
    t_elapsed : float, optional
        Elapsed simulation time for annotation.
    out_dir : str, optional
        Output directory.

    Returns
    -------
    str
        Path to saved file.
    """
    out_dir = out_dir or os.path.dirname(__file__)
    output_file = os.path.join(out_dir, "thermal_final.png")
    vmax = np.max(T)
    if vmax > amb + 250:
        vmax = amb + 250

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    time_label = "Steady State - " if steady_state else (
        f"t = {t_elapsed:.1f} s - " if t_elapsed is not None else ""
    )
    ax1.set_title(f"TOP Layer ({time_label}Max: {np.max(T[0]):.1f} C)")
    im1 = ax1.imshow(
        T[0], cmap='inferno', origin='upper',
        vmin=amb, vmax=vmax, interpolation='bilinear'
    )
    plt.colorbar(im1, ax=ax1)
    ax2.set_title(f"BOTTOM Layer ({time_label}Max: {np.max(T[-1]):.1f} C)")
    im2 = ax2.imshow(
        T[-1], cmap='inferno', origin='upper',
        vmin=amb, vmax=vmax, interpolation='bilinear'
    )
    plt.colorbar(im2, ax=ax2)
    if np.max(H) > 0:
        ax2.contour(H, levels=[0.5], colors='white', linewidths=2, linestyles='--')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    if open_file:
        _open_file(output_file)
    return output_file


def show_results_all_layers(T, H, amb, layer_names, open_file=True, t_elapsed=None,
                            out_dir=None, steady_state=False):
    """
    Save and optionally display all-layer temperature results.

    Parameters
    ----------
    T : np.ndarray
        Temperature array, shape (layers, rows, cols).
    H : np.ndarray
        Heatsink mask.
    amb : float
        Ambient temperature.
    layer_names : list of str
        Names for each layer.
    open_file : bool, optional
        Whether to open the file in default viewer.
    t_elapsed : float, optional
        Elapsed simulation time for annotation.
    out_dir : str, optional
        Output directory.

    Returns
    -------
    str
        Path to saved file.
    """
    out_dir = out_dir or os.path.dirname(__file__)
    output_file = os.path.join(out_dir, "thermal_stackup.png")
    vmax = np.max(T)
    if vmax > amb + 250:
        vmax = amb + 250

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
        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(12, 4 * rows_grid))
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
        if i >= len(axes):
            break
        ax = axes[i]
        name = labels[i]
        max_temp = np.max(T[i])
        time_label = "Steady State - " if steady_state else (
            f"t = {t_elapsed:.1f} s - " if t_elapsed is not None else ""
        )
        ax.set_title(f"{name} - {time_label}Max: {max_temp:.1f}C")
        im = ax.imshow(
            T[i], cmap='inferno', origin='upper',
            vmin=amb, vmax=vmax, interpolation='bilinear'
        )
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        if i == count - 1 and np.max(H) > 0:
            ax.contour(H, levels=[0.5], colors='white', linewidths=2, linestyles='--')

    for j in range(count, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    if open_file:
        _open_file(output_file)
    return output_file


def save_preview_image(
    board,
    copper_ids,
    bbox,
    pads_list,
    settings,
    layer_names,
    stack_info,
    get_pad_pixels_func,
    create_maps_func,
    derive_stackup_func,
    open_file=False,
    out_dir=None,
    geometry_state=None,
    grid_spec=None,
    adaptive_mesh=None,
):
    """
    Save a geometry preview image showing copper, vias, and heat sources.

    Parameters
    ----------
    board : pcbnew.BOARD
        The KiCad board object.
    copper_ids : list of int
        Copper layer IDs in stackup order.
    bbox : pcbnew.EDA_RECT
        Board bounding box.
    pads_list : list
        List of selected pad objects.
    settings : dict
        Simulation settings.
    layer_names : list of str
        Names of copper layers.
    stack_info : dict
        Stackup information from parser.
    get_pad_pixels_func : callable
        Function to get pad pixel coordinates.
    create_maps_func : callable
        Function to create conductivity maps.
    derive_stackup_func : callable
        Function to derive stackup thicknesses.
    open_file : bool, optional
        Whether to open the file in default viewer.
    out_dir : str, optional
        Output directory.

    Returns
    -------
    str or None
        Path to saved file, or None if failed.
    """
    if not board or not bbox:
        return None

    if grid_spec is None:
        res = settings['res']
        w_mm = bbox.GetWidth() * 1e-6
        h_mm = bbox.GetHeight() * 1e-6
        x_min = bbox.GetX() * 1e-6
        y_min = bbox.GetY() * 1e-6
        cols = int(w_mm / res) + 4
        rows = int(h_mm / res) + 4
    else:
        res = float(grid_spec.actual_res_mm)
        x_min = float(grid_spec.x_min_mm)
        y_min = float(grid_spec.y_min_mm)
        rows = int(grid_spec.rows)
        cols = int(grid_spec.cols)

    # Physics constants for mapping
    k_fr4_rel = 1.0
    k_cu_rel = 400.0
    via_factor = 390.0 / 0.3
    ref_cu_thick_m = 35e-6
    layer_count = len(copper_ids)

    stackup_derived = derive_stackup_func(board, copper_ids, stack_info, settings)
    cu_thick_m = [max(1e-9, th * 1e-3) for th in stackup_derived["copper_thickness_mm_used"]]
    k_cu_layers = [k_cu_rel * (th / ref_cu_thick_m) for th in cu_thick_m]

    try:
        if geometry_state is None:
            K, V_map, H_map = create_maps_func(
                board, copper_ids, rows, cols, x_min, y_min, res,
                settings, k_fr4_rel, k_cu_layers, via_factor, pads_list
            )
        else:
            K = np.empty((layer_count, rows, cols), dtype=np.float64)
            for idx, k_cu_layer in enumerate(k_cu_layers):
                K[idx] = np.where(geometry_state.copper_mask[idx], k_cu_layer, k_fr4_rel)
            V_map = geometry_state.via_map
            H_map = geometry_state.heatsink_mask.astype(np.float64, copy=False)

        out_dir = out_dir or settings.get('output_dir') or os.path.dirname(__file__)
        if not os.path.isdir(out_dir):
            out_dir = os.path.dirname(__file__)
        output_file = os.path.join(out_dir, "thermal_preview.png")
        count = len(K)
        cols_grid = 2
        rows_grid = math.ceil(count / 2)

        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(12, 4 * rows_grid), squeeze=False)
        axes = axes.flatten()
        area_summary = str(settings.get("_preview_area_summary", "") or "")
        if area_summary:
            prefix = "Limited simulation area" if settings.get("_preview_area_limited") else "Full simulation area"
            fig.suptitle(f"{prefix}: {area_summary}", fontsize=11)

        # Build pad masks per layer
        pad_masks = [np.zeros((rows, cols), dtype=bool) for _ in range(count)]
        pad_labels = []
        label_limit = 10

        for pad in pads_list or []:
            pad_lid = pad.GetLayer()
            target_indices = []
            if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                target_indices = list(range(count))
            elif pad_lid in copper_ids:
                target_indices = [copper_ids.index(pad_lid)]
            else:
                lname = board.GetLayerName(pad_lid).upper()
                target_indices = [count - 1 if ("B." in lname or "BOT" in lname) else 0]

            pixels = get_pad_pixels_func(pad, rows, cols, x_min, y_min, res)
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

            if adaptive_mesh is not None:
                leaf_sizes = np.maximum(
                    adaptive_mesh.leaves[:, 1] - adaptive_mesh.leaves[:, 0],
                    adaptive_mesh.leaves[:, 3] - adaptive_mesh.leaves[:, 2],
                ).astype(np.uint8)
                cell_sizes = leaf_sizes[adaptive_mesh.leaf_map]
                refinement = np.ma.masked_where(
                    cell_sizes <= 1,
                    adaptive_mesh.max_cell_ratio - cell_sizes + 1,
                )
                ax.imshow(
                    refinement,
                    cmap='Purples',
                    origin='upper',
                    interpolation='none',
                    alpha=0.16,
                )

            # Heatsink overlay (board-level)
            if settings.get('use_heatsink'):
                is_bottom = (i == count - 1) or (name == "B.Cu")
                if is_bottom:
                    ax.imshow(
                        np.ma.masked_where(H_map <= 0, H_map),
                        cmap='Blues', origin='upper', interpolation='none', alpha=0.45
                    )

            # Overlay vias in red
            v_mask = V_map > 1.0
            if np.any(v_mask):
                ax.imshow(
                    np.ma.masked_where(~v_mask, v_mask),
                    cmap='Reds', origin='upper', alpha=0.5, interpolation='none'
                )

            # Overlay pads (heat sources)
            pad_mask = pad_masks[i]
            if np.any(pad_mask):
                ax.imshow(
                    np.ma.masked_where(~pad_mask, pad_mask),
                    cmap='autumn', origin='upper', alpha=0.6, interpolation='none'
                )
                for layer_idx, cx, cy, label in pad_labels:
                    if layer_idx == i:
                        ax.text(cx, cy, str(label), color='black', fontsize=8, ha='center', va='center')

            if settings.get("_preview_area_limited"):
                ax.add_patch(Rectangle(
                    (-0.5, -0.5), cols, rows,
                    fill=False, edgecolor="#d62728", linewidth=1.5,
                ))

            ax.axis('off')

        for j in range(count, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(rect=(0, 0, 1, 0.96) if area_summary else None)
        plt.savefig(output_file, dpi=120)
        plt.close()

        if open_file:
            _open_file(output_file)
        return output_file
    except Exception:
        return None


def save_electrical_connectivity_preview(prepared, config, terminals, layer_names,
                                        out_dir=None, open_file=False, connectivity_report=None):
    """Render three complementary electrical diagnostics.

    ``electrical_connectivity_preview.png`` is the primary view and shows
    primitive centreline/contact topology.  Routed tracks are shown as their
    physical centrelines and filled zones are shown using a visualization-only
    raster skeleton, so closely-spaced planar-transformer turns remain visibly
    separate while zone continuity and accepted primitive junctions can also be
    inspected.

    ``electrical_graph_raster_preview.png`` shows the exact directional grid
    edges accepted by the resistor-network solver.

    ``electrical_copper_occupancy_preview.png`` shows cells containing copper.
    It is a geometry/raster diagnostic only; touching occupancy pixels do not
    by themselves imply an electrical graph connection.
    """
    out_dir = out_dir or os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    topology_path = os.path.join(out_dir, "electrical_connectivity_preview.png")
    graph_path = os.path.join(out_dir, "electrical_graph_raster_preview.png")
    occupancy_path = os.path.join(out_dir, "electrical_copper_occupancy_preview.png")

    if connectivity_report is not None:
        with open(os.path.join(out_dir, "electrical_connectivity_report.json"),
                  "w", encoding="utf-8") as report_file:
            json.dump(connectivity_report, report_file, indent=2, sort_keys=True)

    count = len(config.copper_ids)
    rows = int(config.rows)
    cols = int(config.cols)
    dx = float(config.res)
    x_min = float(config.x_min)
    y_min = float(config.y_min)
    x_max = x_min + cols * dx
    y_max = y_min + rows * dx

    raster_items = sorted(prepared.rasters.items())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(raster_items))))
    color_by_raw_key = {
        raw_key: colors[index]
        for index, (raw_key, _) in enumerate(raster_items)
    }

    def edge_masks(raster, layer_idx):
        """Return the exact in-plane edge masks accepted by _build_net_edges()."""
        mask = np.asarray(raster.copper_mask[layer_idx], dtype=bool)
        if mask.shape != (rows, cols):
            raise ValueError(
                f"Electrical raster has shape {mask.shape}; expected {(rows, cols)}."
            )
        right = (
            mask[:, :-1]
            & mask[:, 1:]
            & np.asarray(raster.connect_right[layer_idx], dtype=bool)
        )
        down = (
            mask[:-1, :]
            & mask[1:, :]
            & np.asarray(raster.connect_down[layer_idx], dtype=bool)
        )
        diag_right_candidates = mask[:-1, :-1] & mask[1:, 1:]
        diag_right_cardinal = mask[:-1, 1:] | mask[1:, :-1]
        diag_right = (
            diag_right_candidates
            & np.asarray(raster.connect_down_right[layer_idx], dtype=bool)
            & ~diag_right_cardinal
        )
        diag_left_candidates = mask[:-1, 1:] & mask[1:, :-1]
        diag_left_cardinal = mask[:-1, :-1] | mask[1:, 1:]
        diag_left = (
            diag_left_candidates
            & np.asarray(raster.connect_down_left[layer_idx], dtype=bool)
            & ~diag_left_cardinal
        )
        return mask, right, down, diag_right, diag_left

    def add_links(ax, links, dr, dc, color, linewidth=0.38, alpha=0.9):
        rr, cc = np.nonzero(links)
        if rr.size == 0:
            return
        x0 = x_min + (cc.astype(np.float64) + 0.5) * dx
        y0 = y_min + (rr.astype(np.float64) + 0.5) * dx
        x1 = x_min + (cc.astype(np.float64) + float(dc) + 0.5) * dx
        y1 = y_min + (rr.astype(np.float64) + float(dr) + 0.5) * dx
        segments = np.stack(
            (
                np.column_stack((x0, y0)),
                np.column_stack((x1, y1)),
            ),
            axis=1,
        )
        ax.add_collection(
            LineCollection(segments, colors=[color], linewidths=linewidth, alpha=alpha)
        )

    def _circle_from_points(a, b, c):
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        det = 2.0 * (
            x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        )
        if abs(det) <= 1e-18:
            return None
        u1 = x1 * x1 + y1 * y1
        u2 = x2 * x2 + y2 * y2
        u3 = x3 * x3 + y3 * y3
        cx = (u1 * (y2 - y3) + u2 * (y3 - y1) + u3 * (y1 - y2)) / det
        cy = (u1 * (x3 - x2) + u2 * (x1 - x3) + u3 * (x2 - x1)) / det
        return cx, cy, math.hypot(x1 - cx, y1 - cy)

    def primitive_centerline_segments(primitive):
        """Return physical track/arc centreline segments in board millimetres."""
        if getattr(primitive, "kind", None) != "Track":
            return []
        obj = primitive.obj
        try:
            start = obj.GetStart()
            end = obj.GetEnd()
            start_xy = (float(start.x) * 1e-6, float(start.y) * 1e-6)
            end_xy = (float(end.x) * 1e-6, float(end.y) * 1e-6)
        except Exception:
            return []
        if not hasattr(obj, "GetMid"):
            return [(start_xy, end_xy)]
        try:
            mid = obj.GetMid()
            mid_xy = (float(mid.x) * 1e-6, float(mid.y) * 1e-6)
            circle = _circle_from_points(start_xy, mid_xy, end_xy)
            if circle is None:
                return [(start_xy, end_xy)]
            cx, cy, radius = circle
            angles = [
                math.atan2(y - cy, x - cx)
                for x, y in (start_xy, mid_xy, end_xy)
            ]
            tau = 2.0 * math.pi
            ccw = (angles[2] - angles[0]) % tau
            mid_ccw = (angles[1] - angles[0]) % tau
            span = ccw if mid_ccw <= ccw else -((angles[0] - angles[2]) % tau)
            steps = max(8, int(math.ceil(abs(span) / (math.pi / 48.0))))
            points = []
            for index in range(steps + 1):
                angle = angles[0] + span * index / steps
                points.append((
                    cx + radius * math.cos(angle),
                    cy + radius * math.sin(angle),
                ))
            points[0] = start_xy
            points[-1] = end_xy
            return list(zip(points, points[1:]))
        except Exception:
            return [(start_xy, end_xy)]

    def unpack_contact(contact):
        """Accept current six-field contacts and older five-field reports."""
        if len(contact) >= 6:
            return contact[0], contact[1], contact[2], contact[3], contact[4], contact[5]
        left, right, layer_idx, x_mm, y_mm = contact
        if getattr(left, "kind", None) == "Track" and getattr(right, "kind", None) == "Track":
            kind = "track_centerline_topology"
        elif getattr(left, "kind", None) == "Track" or getattr(right, "kind", None) == "Track":
            kind = "track_to_shape"
        else:
            kind = "shape_overlap"
        return left, right, layer_idx, x_mm, y_mm, kind

    def primitive_on_layer(primitive, layer_idx):
        try:
            return bool(np.any(primitive.copper_mask[layer_idx]))
        except Exception:
            return False

    zone_skeleton_cache = {}

    def _thin_binary_mask(mask):
        """Return a one-cell Zhang-Suen skeleton using only NumPy.

        This is intentionally a visualization-only helper.  It avoids a
        SciPy/scikit-image dependency in KiCad's Python environment while
        providing a stable centreline for filled zone copper.
        """
        image = np.asarray(mask, dtype=bool)
        if image.ndim != 2 or not np.any(image):
            return np.zeros_like(image, dtype=bool)

        work = np.pad(image.astype(np.uint8), 1, mode="constant")
        max_iterations = max(work.shape)
        for _iteration in range(max_iterations):
            changed = False
            for phase in (0, 1):
                c = work[1:-1, 1:-1]
                p2 = work[:-2, 1:-1]
                p3 = work[:-2, 2:]
                p4 = work[1:-1, 2:]
                p5 = work[2:, 2:]
                p6 = work[2:, 1:-1]
                p7 = work[2:, :-2]
                p8 = work[1:-1, :-2]
                p9 = work[:-2, :-2]

                neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                transitions = (
                    ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                    + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                    + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                    + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                    + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                    + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                    + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                    + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
                )
                if phase == 0:
                    keep_a = (p2 * p4 * p6) == 0
                    keep_b = (p4 * p6 * p8) == 0
                else:
                    keep_a = (p2 * p4 * p8) == 0
                    keep_b = (p2 * p6 * p8) == 0
                remove = (
                    (c == 1)
                    & (neighbours >= 2)
                    & (neighbours <= 6)
                    & (transitions == 1)
                    & keep_a
                    & keep_b
                )
                if np.any(remove):
                    c[remove] = 0
                    changed = True
            if not changed:
                break
        return work[1:-1, 1:-1].astype(bool)

    def _zone_skeleton(primitive, layer_idx):
        key = (id(primitive), int(layer_idx))
        cached = zone_skeleton_cache.get(key)
        if cached is not None:
            return cached
        try:
            mask = np.asarray(primitive.copper_mask[layer_idx], dtype=bool)
        except Exception:
            mask = np.zeros((0, 0), dtype=bool)
        skeleton = _thin_binary_mask(mask)
        zone_skeleton_cache[key] = skeleton
        return skeleton

    def _mask_link_segments(mask, primitive):
        """Convert an 8-connected local raster skeleton into board-mm links."""
        if mask.ndim != 2 or not np.any(mask):
            return [], []
        row0 = int(getattr(primitive, "row0", 0))
        col0 = int(getattr(primitive, "col0", 0))
        segments = []

        def add_local_links(links, dr, dc):
            rr, cc = np.nonzero(links)
            if rr.size == 0:
                return
            global_r = rr.astype(np.float64) + row0
            global_c = cc.astype(np.float64) + col0
            x0 = x_min + (global_c + 0.5) * dx
            y0 = y_min + (global_r + 0.5) * dx
            x1 = x_min + (global_c + float(dc) + 0.5) * dx
            y1 = y_min + (global_r + float(dr) + 0.5) * dx
            segments.extend(
                np.stack(
                    (np.column_stack((x0, y0)), np.column_stack((x1, y1))),
                    axis=1,
                ).tolist()
            )

        add_local_links(mask[:, :-1] & mask[:, 1:], 0, 1)
        add_local_links(mask[:-1, :] & mask[1:, :], 1, 0)
        add_local_links(mask[:-1, :-1] & mask[1:, 1:], 1, 1)
        add_local_links(mask[:-1, 1:] & mask[1:, :-1], 1, -1)

        # Preserve a visible marker for very small zones whose skeleton reduces
        # to one isolated raster cell.
        degree = np.zeros_like(mask, dtype=np.uint8)
        degree[:, :-1] += mask[:, 1:]
        degree[:, 1:] += mask[:, :-1]
        degree[:-1, :] += mask[1:, :]
        degree[1:, :] += mask[:-1, :]
        degree[:-1, :-1] += mask[1:, 1:]
        degree[1:, 1:] += mask[:-1, :-1]
        degree[:-1, 1:] += mask[1:, :-1]
        degree[1:, :-1] += mask[:-1, 1:]
        rr, cc = np.nonzero(mask & (degree == 0))
        isolated = [
            (
                x_min + (float(c + col0) + 0.5) * dx,
                y_min + (float(r + row0) + 0.5) * dx,
            )
            for r, c in zip(rr, cc)
        ]
        return segments, isolated

    def add_topology_tracks(ax, layer_idx):
        """Draw routed track centrelines plus visualization-only zone skeletons."""
        for raw_key, raster in raster_items:
            track_segments = []
            zone_segments = []
            zone_points = []
            for primitive in raster.primitives:
                if not primitive_on_layer(primitive, layer_idx):
                    continue
                if getattr(primitive, "kind", None) == "Zone":
                    segments, isolated = _mask_link_segments(
                        _zone_skeleton(primitive, layer_idx), primitive
                    )
                    zone_segments.extend(segments)
                    zone_points.extend(isolated)
                else:
                    track_segments.extend(primitive_centerline_segments(primitive))
            if track_segments:
                ax.add_collection(LineCollection(
                    track_segments,
                    colors=[color_by_raw_key[raw_key]],
                    linewidths=0.65,
                    alpha=0.95,
                    zorder=2,
                ))
            if zone_segments:
                ax.add_collection(LineCollection(
                    zone_segments,
                    colors=[color_by_raw_key[raw_key]],
                    linewidths=0.85,
                    alpha=0.58,
                    zorder=1.8,
                ))
            if zone_points:
                xy = np.asarray(zone_points, dtype=np.float64)
                ax.scatter(
                    xy[:, 0], xy[:, 1],
                    marker=".", s=5,
                    color=color_by_raw_key[raw_key],
                    alpha=0.58, zorder=1.8,
                )

    def add_contact_markers(ax, layer_idx, topology_view=False):
        """Draw solver-accepted primitive contacts with contact-type markers."""
        buckets = {
            "track_centerline_topology": [],
            "track_endpoint_snap": [],
            "track_endpoint_copper_overlap": [],
            "track_to_shape": [],
            "track_to_shape_raster_overlap": [],
            "shape_overlap": [],
            "shape_raster_overlap": [],
        }
        for _raw_key, contacts in prepared.contact_points.items():
            for contact in contacts:
                _left, _right, contact_layer, x_mm, y_mm, kind = unpack_contact(contact)
                if int(contact_layer) != int(layer_idx):
                    continue
                buckets.setdefault(kind, []).append((x_mm, y_mm))

        # Exact segmented-track joints are numerous.  Keep them small so the
        # centreline itself remains visible; repaired/fallback contacts are
        # deliberately much more prominent.
        specs = {
            "track_centerline_topology": dict(marker=".", s=5, c="black", alpha=0.45),
            "track_endpoint_snap": dict(marker="o", s=34, facecolors="none", edgecolors="darkorange", linewidths=1.0),
            "track_endpoint_copper_overlap": dict(marker="^", s=42, facecolors="none", edgecolors="magenta", linewidths=1.1),
            "track_to_shape": dict(marker="o", s=22, facecolors="none", edgecolors="black", linewidths=0.7),
            "track_to_shape_raster_overlap": dict(marker="x", s=30, c="deepskyblue", linewidths=1.0),
            "shape_overlap": dict(marker="s", s=22, facecolors="none", edgecolors="black", linewidths=0.7),
            "shape_raster_overlap": dict(marker="+", s=32, c="royalblue", linewidths=1.0),
        }
        for kind, points in buckets.items():
            if not points:
                continue
            # In raster graph/occupancy views exact segment joints add little
            # information and clutter the plot; retain all contact classes in
            # the primitive-topology view.
            if not topology_view and kind == "track_centerline_topology":
                continue
            xy = np.asarray(points, dtype=np.float64)
            ax.scatter(xy[:, 0], xy[:, 1], zorder=5, **specs.get(kind, specs["shape_overlap"]))

    def add_net_ties_and_terminals(ax, layer_idx, annotate=False):
        for group_links in prepared.net_tie_links.values():
            for left_pad, right_pad in group_links:
                for pad in (left_pad, right_pad):
                    try:
                        pos = pad.GetPosition()
                        if (
                            pad.GetAttribute() != getattr(pcbnew, "PAD_ATTRIB_PTH", -1)
                            and not pad.GetLayerSet().Contains(config.copper_ids[layer_idx])
                        ):
                            continue
                        ax.scatter(
                            [pos.x * 1e-6], [pos.y * 1e-6],
                            marker="D", color="purple", s=24, zorder=6,
                        )
                    except Exception:
                        continue

        for terminal in terminals:
            layer_ids = getattr(terminal.pad, "GetLayerSet", lambda: None)()
            if (
                getattr(terminal.pad, "GetAttribute", lambda: -1)()
                != getattr(pcbnew, "PAD_ATTRIB_PTH", -1)
            ):
                try:
                    if not layer_ids.Contains(config.copper_ids[layer_idx]):
                        continue
                except Exception:
                    if terminal.pad.GetLayer() != config.copper_ids[layer_idx]:
                        continue
            try:
                pos = terminal.pad.GetPosition()
            except Exception:
                continue
            ax.scatter(
                [pos.x * 1e-6], [pos.y * 1e-6],
                marker="*",
                color="red" if terminal.current_a > 0 else "blue",
                s=55, zorder=7,
            )
            if annotate:
                short_name = str(terminal.name).split(" [", 1)[0]
                ax.annotate(short_name, (pos.x * 1e-6, pos.y * 1e-6), fontsize=5)

    def decorate_axes(ax):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.10)

    def new_figure():
        fig, axes = plt.subplots(
            max(1, math.ceil(count / 2)), 2,
            figsize=(14, max(4, 4 * math.ceil(count / 2))),
            squeeze=False,
        )
        return fig, axes.flatten()

    def finish_figure(fig, axes, title, path):
        for ax in axes[count:]:
            ax.axis("off")
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(path, dpi=200)
        plt.close(fig)

    # 1) Primitive centreline/contact topology -- primary diagnostic.
    fig, axes = new_figure()
    for layer_idx in range(count):
        ax = axes[layer_idx]
        ax.set_title(layer_names[layer_idx] if layer_idx < len(layer_names) else f"Layer {layer_idx}")
        add_topology_tracks(ax, layer_idx)
        add_contact_markers(ax, layer_idx, topology_view=True)
        add_net_ties_and_terminals(ax, layer_idx, annotate=True)
        decorate_axes(ax)
    from matplotlib.lines import Line2D
    contact_handles = [
        Line2D([0], [0], linestyle="-", color="gray", alpha=0.58, linewidth=1.2, label="zone raster skeleton (visual only)"),
        Line2D([0], [0], marker=".", linestyle="None", color="black", label="exact centreline topology"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="darkorange", label="<=25 um endpoint snap"),
        Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="none", markeredgecolor="magenta", label="wide-track endpoint copper overlap"),
        Line2D([0], [0], marker="x", linestyle="None", color="deepskyblue", label="track/shape raster overlap"),
        Line2D([0], [0], marker="+", linestyle="None", color="royalblue", label="shape/shape raster overlap"),
        Line2D([0], [0], marker="D", linestyle="None", color="purple", label="net tie"),
    ]
    axes[0].legend(handles=contact_handles, fontsize=6, loc="best")
    finish_figure(
        fig, axes,
        "Electrical primitive centreline/contact topology - routed tracks + zone skeletons + accepted primitive junctions",
        topology_path,
    )

    # 2) Exact resistor-network raster edges.
    fig, axes = new_figure()
    for layer_idx in range(count):
        ax = axes[layer_idx]
        ax.set_title(layer_names[layer_idx] if layer_idx < len(layer_names) else f"Layer {layer_idx}")
        for raw_key, raster in raster_items:
            _mask, right, down, diag_right, diag_left = edge_masks(raster, layer_idx)
            color = color_by_raw_key[raw_key]
            add_links(ax, right, 0, 1, color)
            add_links(ax, down, 1, 0, color)
            add_links(ax, diag_right, 1, 1, color)
            add_links(ax, diag_left, 1, -1, color)
            via_r, via_c = np.nonzero(raster.via_mask)
            if via_r.size:
                ax.scatter(
                    x_min + (via_c + 0.5) * dx,
                    y_min + (via_r + 0.5) * dx,
                    marker="x", color="black", s=10, zorder=4,
                )
        add_contact_markers(ax, layer_idx, topology_view=False)
        add_net_ties_and_terminals(ax, layer_idx, annotate=False)
        decorate_axes(ax)
    finish_figure(
        fig, axes,
        f"Electrical resistor-network graph - accepted grid edges - actual cell {dx:.4f} mm",
        graph_path,
    )

    # 3) Copper occupancy only.
    fig, axes = new_figure()
    for layer_idx in range(count):
        ax = axes[layer_idx]
        ax.set_title(layer_names[layer_idx] if layer_idx < len(layer_names) else f"Layer {layer_idx}")
        for raw_key, raster in raster_items:
            mask = np.asarray(raster.copper_mask[layer_idx], dtype=bool)
            rr, cc = np.nonzero(mask)
            if rr.size:
                ax.scatter(
                    x_min + (cc + 0.5) * dx,
                    y_min + (rr + 0.5) * dx,
                    s=2.2, color=color_by_raw_key[raw_key], alpha=0.5,
                )
        add_net_ties_and_terminals(ax, layer_idx, annotate=False)
        decorate_axes(ax)
    finish_figure(
        fig, axes,
        f"Electrical copper occupancy - cell contains copper - actual cell {dx:.4f} mm",
        occupancy_path,
    )

    if open_file:
        _open_file(topology_path)
    return topology_path

def _open_file(filepath):
    """
    Open a file in the system default viewer.

    Parameters
    ----------
    filepath : str
        Path to the file to open.
    """
    try:
        if sys.platform == 'win32':
            os.startfile(filepath)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', filepath])
        else:
            subprocess.Popen(['xdg-open', filepath])
    except Exception:
        pass
