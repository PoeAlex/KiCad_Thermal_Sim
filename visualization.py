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
from matplotlib.patches import Rectangle

import pcbnew


def _downsample_layer_block_max(layer, stride):
    """Downsample a 2D temperature array via hotspot-preserving block-max pooling."""
    arr = np.asarray(layer, dtype=np.float64)
    rows, cols = arr.shape
    if stride <= 1 or rows == 0 or cols == 0:
        return arr

    out_rows = (rows + stride - 1) // stride
    out_cols = (cols + stride - 1) // stride
    pad_rows = out_rows * stride - rows
    pad_cols = out_cols * stride - cols

    finite = np.isfinite(arr)
    all_finite = bool(np.all(finite))
    if all_finite and pad_rows == 0 and pad_cols == 0:
        return arr.reshape(out_rows, stride, out_cols, stride).max(axis=(1, 3))

    padded = np.full((out_rows * stride, out_cols * stride), -np.inf, dtype=np.float64)
    if all_finite:
        padded[:rows, :cols] = arr
    else:
        padded[:rows, :cols] = np.where(finite, arr, -np.inf)

    pooled = padded.reshape(out_rows, stride, out_cols, stride).max(axis=(1, 3))
    if not all_finite:
        pooled = np.where(np.isfinite(pooled), pooled, np.nan)
    return pooled


def _downsample_for_plot(layer, max_plot_dim=800):
    """Downsample a 2D array for Matplotlib imshow while preserving full-grid extent."""
    rows, cols = layer.shape[:2]
    max_axis = max(rows, cols)
    if max_axis > max_plot_dim:
        stride = int(math.ceil(max_axis / max_plot_dim))
        return _downsample_layer_block_max(layer, stride), [-0.5, cols - 0.5, rows - 0.5, -0.5]
    return layer, None


def _layer_to_json_list(layer):
    """Convert a 2D layer array into a row-major JSON-safe list rounded to 3 decimals."""
    arr = np.asarray(layer, dtype=np.float64)
    if arr.size == 0:
        return []
    finite = np.isfinite(arr)
    if np.all(finite):
        return np.round(arr, 3).ravel(order="C").tolist()

    safe = np.where(finite, arr, 0.0)
    result = np.round(safe, 3).ravel(order="C").tolist()
    non_finite_indices = np.flatnonzero(~finite.ravel(order="C")).tolist()
    for idx in non_finite_indices:
        result[idx] = None
    return result


def build_interactive_heatmap_payload(
    T,
    amb,
    layer_names,
    res_mm,
    x_min_mm=0.0,
    y_min_mm=0.0,
    show_all=True,
    max_delta_c=250.0,
    max_dim=320,
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
    max_dim : int or None, optional
        Maximum grid dimension along the longer axis before hotspot-preserving
        block-max downsampling is applied. Defaults to 320. Set to None or <= 0
        to disable downsampling.

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

    max_axis = max(rows, cols)
    if max_dim is not None and int(max_dim) > 0 and max_axis > int(max_dim):
        stride = int(math.ceil(max_axis / int(max_dim)))
    else:
        stride = 1
    downsampled = stride > 1

    orig_res_mm = float(res_mm)
    display_res_mm = orig_res_mm * stride

    def _layer_name(index):
        if index < len(layer_names):
            return str(layer_names[index])
        if index == 0:
            return "Top (F.Cu)"
        if index == layer_count - 1:
            return "Bottom (B.Cu)"
        return f"Inner {index}"

    layers = []
    for index in visible_indices:
        layer = np.asarray(T[index], dtype=np.float64)
        finite_layer = np.isfinite(layer)
        if np.any(finite_layer):
            layer_min = float(np.min(layer[finite_layer]))
            layer_max = float(np.max(layer[finite_layer]))
        else:
            layer_min = float(amb)
            layer_max = float(amb)
        display_layer = _downsample_layer_block_max(layer, stride) if downsampled else layer
        disp_rows, disp_cols = display_layer.shape
        flat_data = _layer_to_json_list(display_layer)
        layers.append({
            "index": int(index),
            "name": _layer_name(index),
            "rows": int(disp_rows),
            "cols": int(disp_cols),
            "min_c": round(layer_min, 3),
            "max_c": round(layer_max, 3),
            "data": flat_data,
        })

    return {
        "ambient_c": round(float(amb), 3),
        "vmin_c": round(float(amb), 3),
        "vmax_c": round(vmax, 3),
        "res_mm": round(display_res_mm, 6),
        "original_res_mm": round(orig_res_mm, 6),
        "x_min_mm": round(float(x_min_mm), 6),
        "y_min_mm": round(float(y_min_mm), 6),
        "original_rows": int(rows),
        "original_cols": int(cols),
        "downsample_stride": int(stride),
        "downsampled": bool(downsampled),
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
        plot_layer, extent = _downsample_for_plot(T[i])
        im = ax.imshow(
            plot_layer, cmap='inferno', origin='upper',
            vmin=amb, vmax=vmax, interpolation='bilinear', extent=extent
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
        plot_layer, extent = _downsample_for_plot(layer)
        im = ax.imshow(
            plot_layer,
            cmap="magma",
            origin="upper",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            extent=extent,
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
    top_plot, top_extent = _downsample_for_plot(T[0])
    im1 = ax1.imshow(
        top_plot, cmap='inferno', origin='upper',
        vmin=amb, vmax=vmax, interpolation='bilinear', extent=top_extent
    )
    plt.colorbar(im1, ax=ax1)
    ax2.set_title(f"BOTTOM Layer ({time_label}Max: {np.max(T[-1]):.1f} C)")
    bot_plot, bot_extent = _downsample_for_plot(T[-1])
    im2 = ax2.imshow(
        bot_plot, cmap='inferno', origin='upper',
        vmin=amb, vmax=vmax, interpolation='bilinear', extent=bot_extent
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
        plot_layer, extent = _downsample_for_plot(T[i])
        im = ax.imshow(
            plot_layer, cmap='inferno', origin='upper',
            vmin=amb, vmax=vmax, interpolation='bilinear', extent=extent
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

        max_axis = max(rows, cols)
        stride = int(math.ceil(max_axis / 800)) if max_axis > 800 else 1
        extent = [-0.5, cols - 0.5, rows - 0.5, -0.5] if stride > 1 else None

        for i in range(count):
            ax = axes[i]
            name = layer_names[i] if i < len(layer_names) else f"Layer {i}"
            ax.set_title(f"Preview: {name}")

            # Show copper as a mask overlay
            copper_mask = K[i] > k_fr4_rel
            copper_disp = (_downsample_layer_block_max(copper_mask, stride) > 0) if stride > 1 else copper_mask
            ax.imshow(copper_disp, cmap='Greens', origin='upper', interpolation='none', alpha=0.35, extent=extent)

            # Heatsink overlay (board-level)
            if settings.get('use_heatsink'):
                is_bottom = (i == count - 1) or (name == "B.Cu")
                if is_bottom:
                    h_disp = _downsample_layer_block_max(H_map, stride) if stride > 1 else H_map
                    ax.imshow(
                        np.ma.masked_where(h_disp <= 0, h_disp),
                        cmap='Blues', origin='upper', interpolation='none', alpha=0.45, extent=extent
                    )

            # Overlay vias in red
            v_mask = V_map > 1.0
            if np.any(v_mask):
                v_disp = (_downsample_layer_block_max(v_mask, stride) > 0) if stride > 1 else v_mask
                ax.imshow(
                    np.ma.masked_where(~v_disp, v_disp),
                    cmap='Reds', origin='upper', alpha=0.5, interpolation='none', extent=extent
                )

            # Overlay pads (heat sources)
            pad_mask = pad_masks[i]
            if np.any(pad_mask):
                pad_disp = (_downsample_layer_block_max(pad_mask, stride) > 0) if stride > 1 else pad_mask
                ax.imshow(
                    np.ma.masked_where(~pad_disp, pad_disp),
                    cmap='autumn', origin='upper', alpha=0.6, interpolation='none', extent=extent
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
