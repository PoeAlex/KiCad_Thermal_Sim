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

import pcbnew


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
    out_dir=None
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

    # Keep zone fills up-to-date
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
    k_cu_rel = 400.0
    via_factor = 390.0 / 0.3
    ref_cu_thick_m = 35e-6
    layer_count = len(copper_ids)

    stackup_derived = derive_stackup_func(board, copper_ids, stack_info, settings)
    cu_thick_m = [max(1e-9, th * 1e-3) for th in stackup_derived["copper_thickness_mm_used"]]
    k_cu_layers = [k_cu_rel * (th / ref_cu_thick_m) for th in cu_thick_m]

    try:
        K, V_map, H_map = create_maps_func(
            board, copper_ids, rows, cols, x_min, y_min, res,
            settings, k_fr4_rel, k_cu_layers, via_factor, pads_list
        )

        out_dir = out_dir or settings.get('output_dir') or os.path.dirname(__file__)
        if not os.path.isdir(out_dir):
            out_dir = os.path.dirname(__file__)
        output_file = os.path.join(out_dir, "thermal_preview.png")
        count = len(K)
        cols_grid = 2
        rows_grid = math.ceil(count / 2)

        fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(12, 4 * rows_grid), squeeze=False)
        axes = axes.flatten()

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

            ax.axis('off')

        for j in range(count, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
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
