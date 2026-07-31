"""
Visualization functions for ThermalSim.

This module provides Matplotlib-based plotting functions for thermal
simulation results and geometry previews.
"""

import os
import sys
import math
import tempfile
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

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


def show_results_top_bot(T, H, amb, open_file=True, t_elapsed=None, out_dir=None):
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
    time_label = f"t = {t_elapsed:.1f} s - " if t_elapsed is not None else ""
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


def show_results_all_layers(T, H, amb, layer_names, open_file=True, t_elapsed=None, out_dir=None):
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
        time_label = f"t = {t_elapsed:.1f} s - " if t_elapsed is not None else ""
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
    heat_source_pads=None,
    current_terminals=None,
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
    heat_source_pads : list, optional
        Pads that receive explicit heat-source power.  When omitted, all
        ``pads_list`` entries retain the legacy heat-source highlighting.
    current_terminals : list, optional
        Current-terminal objects with ``pad``, ``current_a``, and ``name``
        attributes. Positive current is shown as entering copper and negative
        current as leaving copper.

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

        # Build role-specific pad masks per layer.  A pad can legitimately be
        # both a heat source and a current terminal, so the overlays are kept
        # independent instead of assigning one exclusive role.
        pad_masks = [np.zeros((rows, cols), dtype=bool) for _ in range(count)]
        current_source_masks = [np.zeros((rows, cols), dtype=bool) for _ in range(count)]
        current_sink_masks = [np.zeros((rows, cols), dtype=bool) for _ in range(count)]
        pad_labels = []
        terminal_labels = []
        label_limit = 10
        heat_source_pads = pads_list if heat_source_pads is None else heat_source_pads
        heat_pad_ids = {id(pad) for pad in (heat_source_pads or [])}
        terminal_by_pad = {}
        for terminal in current_terminals or []:
            if isinstance(terminal, dict):
                terminal_pad = terminal.get("pad")
                current_a = float(terminal.get("current_a", 0.0) or 0.0)
                terminal_name = str(terminal.get("name", "") or "")
            else:
                terminal_pad = getattr(terminal, "pad", None)
                current_a = float(getattr(terminal, "current_a", 0.0) or 0.0)
                terminal_name = str(getattr(terminal, "name", "") or "")
            if terminal_pad is not None:
                terminal_by_pad.setdefault(id(terminal_pad), []).append(
                    (current_a, terminal_name)
                )

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
                pad_terminals = terminal_by_pad.get(id(pad), [])
                for idx in target_indices:
                    for r, c in pixels:
                        if r < rows and c < cols:
                            if id(pad) in heat_pad_ids:
                                pad_masks[idx][r, c] = True
                            if any(current > 0.0 for current, _ in pad_terminals):
                                current_source_masks[idx][r, c] = True
                            if any(current < 0.0 for current, _ in pad_terminals):
                                current_sink_masks[idx][r, c] = True
                if id(pad) in heat_pad_ids and len(pad_labels) < label_limit:
                    try:
                        pos = pad.GetPosition()
                        cx = int((pos.x * 1e-6 - x_min) / res)
                        cy = int((pos.y * 1e-6 - y_min) / res)
                    except Exception:
                        cx, cy = None, None
                    if cx is not None and cy is not None:
                        label = pad.GetNumber() if hasattr(pad, "GetNumber") else ""
                        pad_labels.append((target_indices[0], cx, cy, label))
                if pad_terminals:
                    try:
                        pos = pad.GetPosition()
                        cx = int((pos.x * 1e-6 - x_min) / res)
                        cy = int((pos.y * 1e-6 - y_min) / res)
                    except Exception:
                        cx, cy = None, None
                    if cx is not None and cy is not None:
                        for current_a, terminal_name in pad_terminals:
                            for layer_idx in target_indices:
                                terminal_labels.append(
                                    (layer_idx, cx, cy, current_a, terminal_name)
                                )

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

            source_mask = current_source_masks[i]
            if np.any(source_mask):
                ax.imshow(
                    np.ma.masked_where(~source_mask, source_mask),
                    cmap='Reds', origin='upper', alpha=0.46, interpolation='none'
                )
            sink_mask = current_sink_masks[i]
            if np.any(sink_mask):
                ax.imshow(
                    np.ma.masked_where(~sink_mask, sink_mask),
                    cmap='Blues', origin='upper', alpha=0.46, interpolation='none'
                )
            for layer_idx, cx, cy, current_a, terminal_name in terminal_labels:
                if layer_idx != i:
                    continue
                if current_a == 0.0:
                    continue
                marker = '^' if current_a > 0.0 else 'v'
                color = '#d62728' if current_a > 0.0 else '#1f77b4'
                ax.scatter(
                    [cx], [cy], marker=marker, s=58, c=color,
                    edgecolors='white', linewidths=0.8,
                )
                if terminal_name:
                    ax.text(cx + 1.2, cy + 1.2, terminal_name, color=color, fontsize=7)

            if settings.get("_preview_area_limited"):
                ax.add_patch(Rectangle(
                    (-0.5, -0.5), cols, rows,
                    fill=False, edgecolor="#d62728", linewidth=1.5,
                ))

            ax.axis('off')

        for j in range(count, len(axes)):
            axes[j].axis('off')

        legend_handles = []
        if heat_pad_ids:
            legend_handles.append(Patch(facecolor='#ffb000', alpha=0.65, label='Heat source'))
        if any(np.any(mask) for mask in current_source_masks):
            legend_handles.append(Line2D(
                [0], [0], marker='^', color='none', markerfacecolor='#d62728',
                markeredgecolor='white', markersize=8, label='Current enters copper'
            ))
        if any(np.any(mask) for mask in current_sink_masks):
            legend_handles.append(Line2D(
                [0], [0], marker='v', color='none', markerfacecolor='#1f77b4',
                markeredgecolor='white', markersize=8, label='Current leaves copper'
            ))
        if legend_handles:
            fig.legend(
                handles=legend_handles, loc='upper center', ncol=len(legend_handles),
                bbox_to_anchor=(0.5, 0.965 if area_summary else 0.995), fontsize=8,
            )
        top = 0.90 if area_summary and legend_handles else (0.94 if area_summary or legend_handles else 1.0)
        plt.tight_layout(rect=(0, 0, 1, top))
        plt.savefig(output_file, dpi=120)
        plt.close()

        if open_file:
            _open_file(output_file)
        return output_file

    except Exception:
        return None


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
