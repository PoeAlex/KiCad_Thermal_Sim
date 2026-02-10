"""
HTML report generation for ThermalSim.

This module generates an HTML report summarizing the thermal simulation
results, including settings, stackup information, and embedded images.
"""

import json
import os
import re
import html


def _fmt(val, suffix=""):
    """
    Format a value for display in the report.

    Parameters
    ----------
    val : any
        Value to format. Floats get 4 decimal places.
    suffix : str, optional
        Suffix to append (e.g., " mm", " W").

    Returns
    -------
    str
        Formatted string representation.
    """
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.4f}{suffix}"
    return f"{val}{suffix}"


def _esc(text):
    """
    HTML-escape a text value.

    Parameters
    ----------
    text : str or None
        Text to escape.

    Returns
    -------
    str
        HTML-escaped string, or empty string if input was None.
    """
    return html.escape(text if text is not None else "")


# Inferno colormap LUT (256 RGB entries) — shared by interactive section and viewer
_INFERNO_LUT = ("[[0,0,4],[1,0,5],[1,1,6],[1,1,8],[2,1,10],[2,2,12],[2,2,14],"
    "[3,2,16],[4,3,18],[4,3,20],[5,4,23],[6,4,25],[7,5,27],[8,5,29],"
    "[9,6,31],[10,7,34],[11,7,36],[12,8,38],[13,8,41],[14,9,43],"
    "[16,9,45],[17,10,48],[18,10,50],[20,11,52],[21,11,55],[22,11,57],"
    "[24,12,60],[25,12,62],[27,12,65],[28,12,67],[30,12,69],[31,12,72],"
    "[33,12,74],[35,12,76],[36,12,79],[38,12,81],[40,11,83],[41,11,85],"
    "[43,11,87],[45,11,89],[47,10,91],[49,10,92],[50,10,94],[52,10,95],"
    "[54,9,97],[56,9,98],[57,9,99],[59,9,100],[61,9,101],[62,9,102],"
    "[64,10,103],[66,10,104],[68,10,104],[69,10,105],[71,11,106],"
    "[73,11,106],[74,12,107],[76,12,107],[77,13,108],[79,13,108],"
    "[81,14,108],[82,14,109],[84,15,109],[85,15,109],[87,16,110],"
    "[89,16,110],[90,17,110],[92,18,110],[93,18,110],[95,19,110],"
    "[97,19,110],[98,20,110],[100,21,110],[101,21,110],[103,22,110],"
    "[105,22,110],[106,23,110],[108,24,110],[109,24,110],[111,25,110],"
    "[113,25,110],[114,26,110],[116,26,110],[117,27,110],[119,28,109],"
    "[120,28,109],[122,29,109],[124,29,109],[125,30,109],[127,30,108],"
    "[128,31,108],[130,32,108],[132,32,107],[133,33,107],[135,33,107],"
    "[136,34,106],[138,34,106],[140,35,105],[141,35,105],[143,36,105],"
    "[144,37,104],[146,37,104],[147,38,103],[149,38,103],[151,39,102],"
    "[152,39,102],[154,40,101],[155,41,100],[157,41,100],[159,42,99],"
    "[160,42,99],[162,43,98],[163,44,97],[165,44,96],[166,45,96],"
    "[168,46,95],[169,46,94],[171,47,94],[173,48,93],[174,48,92],"
    "[176,49,91],[177,50,90],[179,50,90],[180,51,89],[182,52,88],"
    "[183,53,87],[185,53,86],[186,54,85],[188,55,84],[189,56,83],"
    "[191,57,82],[192,58,81],[193,58,80],[195,59,79],[196,60,78],"
    "[198,61,77],[199,62,76],[200,63,75],[202,64,74],[203,65,73],"
    "[204,66,72],[206,67,71],[207,68,70],[208,69,69],[210,70,68],"
    "[211,71,67],[212,72,66],[213,74,65],[215,75,63],[216,76,62],"
    "[217,77,61],[218,78,60],[219,80,59],[221,81,58],[222,82,56],"
    "[223,83,55],[224,85,54],[225,86,53],[226,87,52],[227,89,51],"
    "[228,90,49],[229,92,48],[230,93,47],[231,94,46],[232,96,45],"
    "[233,97,43],[234,99,42],[235,100,41],[235,102,40],[236,103,38],"
    "[237,105,37],[238,106,36],[239,108,35],[239,110,33],[240,111,32],"
    "[241,113,31],[241,115,29],[242,116,28],[243,118,27],[243,120,25],"
    "[244,121,24],[245,123,23],[245,125,21],[246,126,20],[246,128,19],"
    "[247,130,18],[247,132,16],[248,133,15],[248,135,14],[248,137,12],"
    "[249,139,11],[249,140,10],[249,142,9],[250,144,8],[250,146,7],"
    "[250,148,7],[251,150,6],[251,151,6],[251,153,6],[251,155,6],"
    "[251,157,7],[252,159,7],[252,161,8],[252,163,9],[252,165,10],"
    "[252,166,12],[252,168,13],[252,170,15],[252,172,17],[252,174,18],"
    "[252,176,20],[252,178,22],[252,180,24],[251,182,26],[251,184,29],"
    "[251,186,31],[251,188,33],[251,190,35],[250,192,38],[250,194,40],"
    "[250,196,42],[250,198,45],[249,199,47],[249,201,50],[249,203,53],"
    "[248,205,55],[248,207,58],[247,209,61],[247,211,64],[246,213,67],"
    "[246,215,70],[245,217,73],[245,219,76],[244,221,79],[244,223,83],"
    "[244,225,86],[243,227,90],[243,229,93],[242,230,97],[242,232,101],"
    "[242,234,105],[241,236,109],[241,237,113],[241,239,117],"
    "[241,241,121],[242,242,125],[242,244,130],[243,245,134],"
    "[243,246,138],[244,248,142],[245,249,146],[246,250,150],"
    "[248,251,154],[249,252,157],[250,253,161],[252,255,164]]")


def _build_interactive_section(T_data, ambient, layer_names):
    """
    Build an interactive heatmap HTML section with canvas and hover tooltips.

    Parameters
    ----------
    T_data : array-like
        Temperature data with shape (layers, rows, cols).
    ambient : float
        Ambient temperature in degrees Celsius.
    layer_names : list of str
        Names of copper layers in stackup order.

    Returns
    -------
    str
        HTML fragment containing the interactive heatmap section.
    """
    import numpy as np

    T_arr = np.asarray(T_data, dtype=float)
    n_layers, n_rows, n_cols = T_arr.shape

    # Round to 1 decimal place for compact JSON
    T_rounded = np.round(T_arr, 1)
    T_list = T_rounded.tolist()
    t_json = json.dumps(T_list)

    # Sanitize layer names for JS string literals
    js_layer_names = json.dumps([str(n) for n in layer_names])

    # vmin/vmax matching visualization.py logic (use rounded data for consistency)
    vmin = round(float(ambient), 1)
    vmax = round(float(np.max(T_rounded)), 1)
    if vmax > vmin + 250:
        vmax = round(vmin + 250, 1)

    # Per-layer max temperatures (from rounded data)
    layer_maxes = [round(float(np.max(T_rounded[i])), 1) for i in range(n_layers)]

    # Build canvas elements
    canvas_html = ""
    for i in range(n_layers):
        lname = _esc(str(layer_names[i]) if i < len(layer_names) else f"Layer {i}")
        canvas_html += f"""
    <div class="heatmap-layer">
      <h3>{lname} &mdash; Max: {layer_maxes[i]:.1f} &deg;C</h3>
      <div style="position:relative; display:inline-block;">
        <canvas id="layer-{i}" width="{n_cols}" height="{n_rows}"
                style="border:1px solid #ccc; image-rendering:pixelated;"></canvas>
        <div id="tooltip-{i}" class="hm-tooltip"></div>
      </div>
    </div>"""

    section = f"""
  <h2>Interactive Heatmap</h2>
  <style>
    .heatmap-layer {{ margin-bottom: 16px; }}
    .heatmap-layer canvas {{ width: 90vw; max-width: 1200px; cursor: crosshair; }}
    .hm-tooltip {{
      position: absolute; display: none; pointer-events: none;
      background: rgba(0,0,0,0.82); color: #fff; padding: 4px 8px;
      border-radius: 3px; font-size: 13px; white-space: nowrap;
      z-index: 10;
    }}
    .color-legend {{ display: inline-flex; align-items: center; gap: 6px; margin-top: 10px; }}
    .color-legend canvas {{ border: 1px solid #ccc; }}
    .color-legend span {{ font-size: 12px; }}
  </style>
  {canvas_html}
  <div class="color-legend">
    <span>{vmin:.1f} &deg;C</span>
    <canvas id="hm-legend" width="256" height="18"></canvas>
    <span>{vmax:.1f} &deg;C</span>
  </div>
  <script>
  (function() {{
    var T_DATA = {t_json};
    var LAYER_NAMES = {js_layer_names};
    var AMBIENT = {vmin};
    var VMIN = {vmin};
    var VMAX = {vmax};
    var INFERNO = {_INFERNO_LUT};

    function valToColor(v) {{
      var t = (v - VMIN) / (VMAX - VMIN || 1);
      if (t < 0) t = 0; if (t > 1) t = 1;
      var idx = Math.round(t * 255);
      return INFERNO[idx];
    }}

    function renderHeatmap(canvasId, layerData) {{
      var c = document.getElementById(canvasId);
      if (!c) return;
      var ctx = c.getContext('2d');
      var rows = layerData.length, cols = layerData[0].length;
      var img = ctx.createImageData(cols, rows);
      for (var r = 0; r < rows; r++) {{
        for (var co = 0; co < cols; co++) {{
          var rgb = valToColor(layerData[r][co]);
          var off = (r * cols + co) * 4;
          img.data[off] = rgb[0]; img.data[off+1] = rgb[1];
          img.data[off+2] = rgb[2]; img.data[off+3] = 255;
        }}
      }}
      ctx.putImageData(img, 0, 0);
    }}

    function renderLegend() {{
      var c = document.getElementById('hm-legend');
      if (!c) return;
      var ctx = c.getContext('2d');
      var img = ctx.createImageData(256, 18);
      for (var x = 0; x < 256; x++) {{
        var rgb = INFERNO[x];
        for (var y = 0; y < 18; y++) {{
          var off = (y * 256 + x) * 4;
          img.data[off] = rgb[0]; img.data[off+1] = rgb[1];
          img.data[off+2] = rgb[2]; img.data[off+3] = 255;
        }}
      }}
      ctx.putImageData(img, 0, 0);
    }}

    for (var i = 0; i < T_DATA.length; i++) {{
      renderHeatmap('layer-' + i, T_DATA[i]);
      (function(idx) {{
        var canvas = document.getElementById('layer-' + idx);
        var tip = document.getElementById('tooltip-' + idx);
        if (!canvas || !tip) return;
        var rows = T_DATA[idx].length, cols = T_DATA[idx][0].length;
        canvas.addEventListener('mousemove', function(e) {{
          var rect = canvas.getBoundingClientRect();
          var sx = canvas.width / rect.width;
          var sy = canvas.height / rect.height;
          var col = Math.floor((e.clientX - rect.left) * sx);
          var row = Math.floor((e.clientY - rect.top) * sy);
          if (row < 0 || row >= rows || col < 0 || col >= cols) {{
            tip.style.display = 'none'; return;
          }}
          var val = T_DATA[idx][row][col];
          tip.textContent = LAYER_NAMES[idx] + ' | (' + row + ', ' + col + ') | ' + val.toFixed(1) + ' \\u00b0C';
          tip.style.display = 'block';
          tip.style.left = (e.clientX - rect.left + 12) + 'px';
          tip.style.top = (e.clientY - rect.top - 24) + 'px';
        }});
        canvas.addEventListener('mouseout', function() {{
          tip.style.display = 'none';
        }});
      }})(i);
    }}
    renderLegend();
  }})();
  </script>
"""
    return section


def write_html_report(
    settings,
    stack_info,
    stackup_derived,
    pad_power,
    layer_names,
    preview_path,
    heatmap_path,
    k_norm_info=None,
    out_dir=None,
    snapshot_debug=None,
    snapshot_files=None,
    T_data=None,
    ambient=None,
):
    """
    Generate an HTML report for the thermal simulation.

    Parameters
    ----------
    settings : dict
        Simulation settings from the GUI dialog.
    stack_info : dict
        Raw stackup information from parse_stackup_from_board_file.
    stackup_derived : dict
        Derived thickness values used in simulation, containing:
        - total_thick_mm_used : float
        - stack_board_thick_mm : float or None
        - copper_thickness_mm_used : list of float
        - gap_mm_used : list of float
        - gap_fallback_used : bool
    pad_power : list of tuple
        List of (pad_name, power_watts) tuples.
    layer_names : list of str
        Names of copper layers in stackup order.
    preview_path : str or None
        Path to the preview image file.
    heatmap_path : str or None
        Path to the final heatmap image file.
    k_norm_info : dict, optional
        Solver normalization and debug information.
    out_dir : str, optional
        Output directory for the report. Defaults to module directory.
    snapshot_debug : dict, optional
        Debug information about snapshot generation.
    snapshot_files : list, optional
        List of (time, filename) tuples for snapshot images.
    T_data : numpy.ndarray, optional
        Temperature array with shape (layers, rows, cols) for interactive
        heatmap. If None, the interactive section is omitted.
    ambient : float, optional
        Ambient temperature in degrees Celsius. Required when T_data is
        provided.

    Returns
    -------
    str or None
        Path to the generated report file, or None if writing failed.

    Notes
    -----
    The report includes:
    - Thickness summary (board, copper layers, dielectric gaps)
    - Simulation settings table
    - Power per pad table
    - Embedded preview and heatmap images
    - Time-series snapshots if available
    - Debug information for troubleshooting
    """
    out_dir = out_dir or os.path.dirname(__file__)
    report_path = os.path.join(out_dir, "thermal_report.html")

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
        f"<tr><td>{_esc(layer_names[i])} -> {_esc(layer_names[i + 1])}</td><td>{_esc(_fmt(g, ' mm'))}</td></tr>"
        for i, g in enumerate(gaps_used)
    )

    t_fr4_eff_mm = []
    if isinstance(k_norm_info, dict):
        t_fr4_eff_mm = k_norm_info.get("t_fr4_eff_per_plane_mm") or []
    fr4_eff_rows = "\n".join(
        f"<tr><td>{_esc(layer_names[i])}</td><td>{_esc(_fmt(val, ' mm'))}</td></tr>"
        for i, val in enumerate(t_fr4_eff_mm)
    )

    # Process snapshot files
    snapshot_items = []
    if snapshot_files is not None:
        for item in snapshot_files:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                t_val, fname = item
            else:
                fname = os.path.basename(str(item))
                m = re.search(r"_t([0-9.]+)", fname)
                t_val = float(m.group(1)) if m else None
            snapshot_items.append((t_val, os.path.basename(fname)))
    else:
        try:
            import glob
            for path in glob.glob(os.path.join(out_dir, "snap_*.png")):
                fname = os.path.basename(path)
                m = re.search(r"_t([0-9.]+)", fname)
                t_val = float(m.group(1)) if m else None
                snapshot_items.append((t_val, fname))
        except Exception:
            snapshot_items = []
    snapshot_items.sort(key=lambda x: (x[0] if x[0] is not None else 1e9, x[1]))

    snapshots_html = ""
    for t_val, fname in snapshot_items:
        label = f"t = {t_val:.1f} s" if t_val is not None else fname
        snapshots_html += f"<div><p class='small'>{_esc(label)}</p><img src='{_esc(fname)}' alt='{_esc(label)}'></div>"

    if k_norm_info is None:
        k_norm_info = {}
    if snapshot_debug is None:
        snapshot_debug = {}

    k_norm_rows = "\n".join(
        f"<tr><td>{_esc(str(k))}</td><td>{_esc(_fmt(v))}</td></tr>"
        for k, v in k_norm_info.items()
    )
    snapshot_debug_rows = "\n".join(
        f"<tr><td>{_esc(str(k))}</td><td>{_esc(_fmt(v))}</td></tr>"
        for k, v in snapshot_debug.items()
    )

    # Build interactive heatmap section if T_data provided
    interactive_html = ""
    if T_data is not None and ambient is not None:
        try:
            interactive_html = _build_interactive_section(
                T_data, ambient, layer_names
            )
        except Exception:
            interactive_html = ""

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
  <h2>Effective Dielectric Thickness per Plane</h2>
  <p class="small">t_fr4_eff is per-plane effective dielectric thickness derived by averaging adjacent interface gaps; therefore its max may be lower than the maximum single interface gap.</p>
  <table>
    <tr><th>Plane</th><th>t_fr4_eff</th></tr>
    {fr4_eff_rows if fr4_eff_rows else "<tr><td colspan='2'>n/a</td></tr>"}
  </table>

  <h2>Debug</h2>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    {k_norm_rows}
  </table>

  <h2>Snapshot Debug</h2>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    {snapshot_debug_rows}
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

  <h2>Snapshots</h2>
  <div class="images">
    {snapshots_html if snapshots_html else "<p class='small'>No snapshots captured.</p>"}
  </div>
  {interactive_html}
</body>
</html>
"""

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_body)
    except Exception:
        return None
    return report_path


def write_interactive_viewer(
    T_snapshots,
    T_final,
    sim_time,
    ambient,
    layer_names,
    out_dir=None,
):
    """
    Write a standalone interactive HTML viewer with layer/time switching.

    Parameters
    ----------
    T_snapshots : list of (float, numpy.ndarray)
        Snapshot temperature arrays as (elapsed_time, T_array) tuples.
        Each T_array has shape (layers, rows, cols).
    T_final : numpy.ndarray
        Final temperature array with shape (layers, rows, cols).
    sim_time : float
        Total simulation time in seconds.
    ambient : float
        Ambient temperature in degrees Celsius.
    layer_names : list of str
        Names of copper layers in stackup order.
    out_dir : str, optional
        Output directory. Defaults to module directory.

    Returns
    -------
    str or None
        Path to the generated viewer file, or None if writing failed.
    """
    import numpy as np

    out_dir = out_dir or os.path.dirname(__file__)
    viewer_path = os.path.join(out_dir, "thermal_viewer.html")

    T_final_arr = np.asarray(T_final, dtype=float)
    n_layers, n_rows, n_cols = T_final_arr.shape

    # Build frames list: snapshots + final
    frames = []
    if T_snapshots:
        for t_elapsed, T_snap in T_snapshots:
            frames.append((f"t = {t_elapsed:.1f} s", np.asarray(T_snap, dtype=float)))

    # Size guard: if total cells > 5M, subsample snapshot frames
    max_cells = 5_000_000
    cells_per_frame = n_layers * n_rows * n_cols
    if cells_per_frame > 0 and len(frames) > 0:
        max_snap_frames = max(1, max_cells // cells_per_frame - 1)  # reserve 1 for Final
        if len(frames) > max_snap_frames:
            step = len(frames) / max_snap_frames
            indices = [int(i * step) for i in range(max_snap_frames)]
            frames = [frames[i] for i in indices]

    # Always add Final as last frame
    frames.append((f"Final ({sim_time:.1f} s)", T_final_arr))

    # Compute global vmin/vmax across ALL frames
    vmin = round(float(ambient), 1)
    global_max = float(max(np.max(f[1]) for f in frames))
    vmax = round(global_max, 1)
    if vmax > vmin + 250:
        vmax = round(vmin + 250, 1)

    # Build FRAMES JSON: round to 1dp for compact output
    frames_json_list = []
    for label, T_arr in frames:
        T_rounded = np.round(T_arr, 1).tolist()
        frames_json_list.append({"label": label, "data": T_rounded})
    frames_json = json.dumps(frames_json_list)

    js_layer_names = json.dumps([str(n) for n in layer_names])

    # Layer buttons (hidden if single layer)
    layer_buttons_style = "display:none;" if n_layers <= 1 else ""
    layer_buttons_html = ""
    for i, lname in enumerate(layer_names):
        layer_buttons_html += (
            f'<button class="btn layer-btn{" active" if i == 0 else ""}" '
            f'data-layer="{i}">{_esc(str(lname))}</button>\n'
        )

    # Frame buttons (hidden if single frame i.e. no snapshots)
    frame_buttons_style = "display:none;" if len(frames) <= 1 else ""
    frame_buttons_html = ""
    for i, (label, _) in enumerate(frames):
        is_last = (i == len(frames) - 1)
        frame_buttons_html += (
            f'<button class="btn frame-btn{" active" if is_last else ""}" '
            f'data-frame="{i}">{_esc(label)}</button>\n'
        )

    viewer_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ThermalSim Viewer</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #1a1a2e; color: #e0e0e0; font-family: Arial, sans-serif;
           display: flex; flex-direction: column; align-items: center; padding: 16px; }}
    h1 {{ font-size: 1.3em; margin-bottom: 10px; color: #f0f0f0; }}
    .toolbar {{ margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 6px;
               justify-content: center; }}
    .btn {{ background: #2a2a4a; color: #ccc; border: 1px solid #444; padding: 5px 12px;
           cursor: pointer; border-radius: 3px; font-size: 13px; }}
    .btn:hover {{ background: #3a3a5a; }}
    .btn.active {{ background: #4a6fa5; color: #fff; border-color: #6a8fc5; }}
    .viewer-main {{ display: flex; gap: 16px; align-items: flex-start;
                    width: 90vw; max-width: 1400px; margin: 8px auto; }}
    .canvas-wrap {{ position: relative; flex: 1; min-width: 0; }}
    #heatmap, #overlay {{ image-rendering: pixelated; border: 1px solid #555; width: 100%; }}
    #overlay {{ position: absolute; top: 0; left: 0; cursor: crosshair; }}
    #tooltip {{ position: absolute; display: none; pointer-events: none;
               background: rgba(0,0,0,0.85); color: #fff; padding: 4px 8px;
               border-radius: 3px; font-size: 13px; white-space: nowrap; z-index: 10; }}
    #stats-panel {{ width: 250px; min-height: 200px; background: #2a2a4a;
                    border-radius: 4px; padding: 12px; font-size: 13px;
                    max-height: 80vh; overflow-y: auto; flex-shrink: 0; }}
    #stats-panel h3 {{ font-size: 14px; margin-bottom: 8px; color: #f0f0f0; }}
    .stat-rect {{ margin-bottom: 10px; padding: 8px; background: #1a1a2e;
                  border-radius: 3px; border-left: 3px solid #888; }}
    .stat-rect .stat-hdr {{ display: flex; justify-content: space-between; align-items: center;
                            margin-bottom: 4px; }}
    .stat-rect .stat-label {{ font-weight: bold; }}
    .stat-rect .stat-del {{ cursor: pointer; color: #f66; font-size: 15px; line-height: 1; }}
    .stat-rect .stat-del:hover {{ color: #ff4444; }}
    .stat-rect .stat-row {{ color: #bbb; }}
    .clear-btn {{ background: #4a2a2a; color: #ccc; border: 1px solid #644; padding: 4px 10px;
                  cursor: pointer; border-radius: 3px; font-size: 12px; margin-top: 8px; width: 100%; }}
    .clear-btn:hover {{ background: #5a3a3a; }}
    .legend {{ display: flex; align-items: center; gap: 8px; margin: 8px 0; }}
    .legend canvas {{ border: 1px solid #555; }}
    .legend span {{ font-size: 12px; }}
    .info {{ font-size: 12px; color: #aaa; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>ThermalSim Interactive Viewer</h1>
  <div class="toolbar" style="{layer_buttons_style}">
    {layer_buttons_html}
  </div>
  <div class="toolbar" style="{frame_buttons_style}">
    {frame_buttons_html}
  </div>
  <div class="viewer-main">
    <div class="canvas-wrap">
      <canvas id="heatmap" width="{n_cols}" height="{n_rows}"></canvas>
      <canvas id="overlay" width="{n_cols}" height="{n_rows}"></canvas>
      <div id="tooltip"></div>
    </div>
    <div id="stats-panel">
      <h3>Selections</h3>
      <div id="stats-list"></div>
      <button class="clear-btn" id="clear-all-btn">Clear All</button>
    </div>
  </div>
  <div class="legend">
    <span>{vmin:.1f} &deg;C</span>
    <canvas id="legend-bar" width="256" height="18"></canvas>
    <span>{vmax:.1f} &deg;C</span>
  </div>
  <div class="info" id="info-line"></div>

  <script>
  (function() {{
    var FRAMES = {frames_json};
    var LAYER_NAMES = {js_layer_names};
    var VMIN = {vmin};
    var VMAX = {vmax};
    var INFERNO = {_INFERNO_LUT};

    var currentLayer = 0;
    var currentFrame = FRAMES.length - 1;

    // Rectangle selection state
    var rectangles = [];
    var drawing = false, startCol = 0, startRow = 0;
    var RECT_COLORS = ['#ff4444','#44ff44','#4488ff','#ffaa00','#ff44ff','#44ffff'];

    function valToColor(v) {{
      var t = (v - VMIN) / (VMAX - VMIN || 1);
      if (t < 0) t = 0; if (t > 1) t = 1;
      return INFERNO[Math.round(t * 255)];
    }}

    function render() {{
      var c = document.getElementById('heatmap');
      var ctx = c.getContext('2d');
      var data = FRAMES[currentFrame].data[currentLayer];
      var rows = data.length, cols = data[0].length;
      var img = ctx.createImageData(cols, rows);
      for (var r = 0; r < rows; r++) {{
        for (var co = 0; co < cols; co++) {{
          var rgb = valToColor(data[r][co]);
          var off = (r * cols + co) * 4;
          img.data[off] = rgb[0]; img.data[off+1] = rgb[1];
          img.data[off+2] = rgb[2]; img.data[off+3] = 255;
        }}
      }}
      ctx.putImageData(img, 0, 0);

      var layerData = FRAMES[currentFrame].data[currentLayer];
      var mx = -Infinity;
      for (var r2 = 0; r2 < layerData.length; r2++)
        for (var c2 = 0; c2 < layerData[r2].length; c2++)
          if (layerData[r2][c2] > mx) mx = layerData[r2][c2];
      var lname = currentLayer < LAYER_NAMES.length ? LAYER_NAMES[currentLayer] : 'Layer ' + currentLayer;
      document.getElementById('info-line').textContent =
        'Layer: ' + lname + ' | Frame: ' + FRAMES[currentFrame].label + ' | Max: ' + mx.toFixed(1) + ' \\u00b0C';
      recomputeAllStats();
      renderOverlay();
    }}

    function renderLegend() {{
      var c = document.getElementById('legend-bar');
      var ctx = c.getContext('2d');
      var img = ctx.createImageData(256, 18);
      for (var x = 0; x < 256; x++) {{
        var rgb = INFERNO[x];
        for (var y = 0; y < 18; y++) {{
          var off = (y * 256 + x) * 4;
          img.data[off] = rgb[0]; img.data[off+1] = rgb[1];
          img.data[off+2] = rgb[2]; img.data[off+3] = 255;
        }}
      }}
      ctx.putImageData(img, 0, 0);
    }}

    // --- Rectangle selection ---
    function computeStats(rect) {{
      var data = FRAMES[currentFrame].data[currentLayer];
      var rMin = Infinity, rMax = -Infinity, sum = 0, count = 0;
      var r1 = Math.min(rect.r1, rect.r2), r2 = Math.max(rect.r1, rect.r2);
      var c1 = Math.min(rect.c1, rect.c2), c2 = Math.max(rect.c1, rect.c2);
      for (var r = r1; r <= r2; r++) {{
        for (var c = c1; c <= c2; c++) {{
          var v = data[r][c];
          if (v < rMin) rMin = v;
          if (v > rMax) rMax = v;
          sum += v; count++;
        }}
      }}
      return {{ min: rMin, avg: sum / count, max: rMax, count: count }};
    }}

    function recomputeAllStats() {{
      for (var i = 0; i < rectangles.length; i++) {{
        rectangles[i].stats = computeStats(rectangles[i]);
      }}
      updateStatsPanel();
    }}

    function updateStatsPanel() {{
      var list = document.getElementById('stats-list');
      if (!list) return;
      if (rectangles.length === 0) {{
        list.innerHTML = '<div style="color:#888;font-size:12px;">Click and drag on the heatmap to measure a region.</div>';
        return;
      }}
      var html = '';
      for (var i = 0; i < rectangles.length; i++) {{
        var rc = rectangles[i];
        var s = rc.stats;
        html += '<div class="stat-rect" style="border-left-color:' + rc.color + ';">';
        html += '<div class="stat-hdr"><span class="stat-label" style="color:' + rc.color + ';">Rect ' + (i + 1) + '</span>';
        html += '<span class="stat-del" data-idx="' + i + '">&times;</span></div>';
        html += '<div class="stat-row">Min: ' + s.min.toFixed(1) + ' \\u00b0C</div>';
        html += '<div class="stat-row">Avg: ' + s.avg.toFixed(1) + ' \\u00b0C</div>';
        html += '<div class="stat-row">Max: ' + s.max.toFixed(1) + ' \\u00b0C</div>';
        html += '<div class="stat-row" style="color:#777;">' + s.count + ' cells</div>';
        html += '</div>';
      }}
      list.innerHTML = html;
      list.querySelectorAll('.stat-del').forEach(function(el) {{
        el.addEventListener('click', function() {{
          var idx = parseInt(el.getAttribute('data-idx'));
          rectangles.splice(idx, 1);
          recomputeAllStats();
          renderOverlay();
        }});
      }});
    }}

    function renderOverlay(tempRect) {{
      var ov = document.getElementById('overlay');
      if (!ov) return;
      var ctx = ov.getContext('2d');
      ctx.clearRect(0, 0, ov.width, ov.height);
      for (var i = 0; i < rectangles.length; i++) {{
        var rc = rectangles[i];
        var x = Math.min(rc.c1, rc.c2), y = Math.min(rc.r1, rc.r2);
        var w = Math.abs(rc.c2 - rc.c1) + 1, h = Math.abs(rc.r2 - rc.r1) + 1;
        ctx.strokeStyle = rc.color;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, w, h);
        ctx.font = '3px Arial';
        ctx.fillStyle = rc.color;
        ctx.fillText((i + 1).toString(), x + 0.5, y + 3);
      }}
      if (tempRect) {{
        var tx = Math.min(tempRect.c1, tempRect.c2), ty = Math.min(tempRect.r1, tempRect.r2);
        var tw = Math.abs(tempRect.c2 - tempRect.c1) + 1, th = Math.abs(tempRect.r2 - tempRect.r1) + 1;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 0.5;
        ctx.setLineDash([2, 2]);
        ctx.strokeRect(tx, ty, tw, th);
        ctx.setLineDash([]);
      }}
    }}

    function getDataCoords(e, canvasEl) {{
      var rect = canvasEl.getBoundingClientRect();
      var sx = canvasEl.width / rect.width;
      var sy = canvasEl.height / rect.height;
      var col = Math.floor((e.clientX - rect.left) * sx);
      var row = Math.floor((e.clientY - rect.top) * sy);
      var data = FRAMES[currentFrame].data[currentLayer];
      var rows = data.length, cols = data[0].length;
      col = Math.max(0, Math.min(cols - 1, col));
      row = Math.max(0, Math.min(rows - 1, row));
      return {{ row: row, col: col }};
    }}

    // Layer button handlers
    document.querySelectorAll('.layer-btn').forEach(function(btn) {{
      btn.addEventListener('click', function() {{
        document.querySelectorAll('.layer-btn').forEach(function(b) {{ b.classList.remove('active'); }});
        btn.classList.add('active');
        currentLayer = parseInt(btn.getAttribute('data-layer'));
        render();
      }});
    }});

    // Frame button handlers
    document.querySelectorAll('.frame-btn').forEach(function(btn) {{
      btn.addEventListener('click', function() {{
        document.querySelectorAll('.frame-btn').forEach(function(b) {{ b.classList.remove('active'); }});
        btn.classList.add('active');
        currentFrame = parseInt(btn.getAttribute('data-frame'));
        render();
      }});
    }});

    // Overlay canvas event handlers (hover + rectangle drawing)
    var overlay = document.getElementById('overlay');
    var tip = document.getElementById('tooltip');

    overlay.addEventListener('mousedown', function(e) {{
      var pos = getDataCoords(e, overlay);
      startCol = pos.col; startRow = pos.row;
      drawing = true;
    }});

    overlay.addEventListener('mousemove', function(e) {{
      var pos = getDataCoords(e, overlay);
      // Tooltip
      var data = FRAMES[currentFrame].data[currentLayer];
      var val = data[pos.row][pos.col];
      var lname = currentLayer < LAYER_NAMES.length ? LAYER_NAMES[currentLayer] : 'Layer ' + currentLayer;
      tip.textContent = lname + ' | (' + pos.row + ', ' + pos.col + ') | ' + val.toFixed(1) + ' \\u00b0C';
      tip.style.display = 'block';
      var rect = overlay.getBoundingClientRect();
      tip.style.left = (e.clientX - rect.left + 12) + 'px';
      tip.style.top = (e.clientY - rect.top - 24) + 'px';
      // Rubber-band
      if (drawing) {{
        renderOverlay({{ r1: startRow, c1: startCol, r2: pos.row, c2: pos.col }});
      }}
    }});

    overlay.addEventListener('mouseup', function(e) {{
      if (!drawing) return;
      drawing = false;
      var pos = getDataCoords(e, overlay);
      if (pos.row === startRow && pos.col === startCol) return;
      var color = RECT_COLORS[rectangles.length % RECT_COLORS.length];
      var newRect = {{ r1: startRow, c1: startCol, r2: pos.row, c2: pos.col, color: color, stats: {{}} }};
      newRect.stats = computeStats(newRect);
      rectangles.push(newRect);
      updateStatsPanel();
      renderOverlay();
    }});

    overlay.addEventListener('mouseout', function() {{
      tip.style.display = 'none';
      if (drawing) {{
        renderOverlay();
      }}
    }});

    // Clear all button
    document.getElementById('clear-all-btn').addEventListener('click', function() {{
      rectangles = [];
      updateStatsPanel();
      renderOverlay();
    }});

    renderLegend();
    render();
  }})();
  </script>
</body>
</html>
"""

    try:
        with open(viewer_path, "w", encoding="utf-8") as f:
            f.write(viewer_html)
    except Exception:
        return None
    return viewer_path
