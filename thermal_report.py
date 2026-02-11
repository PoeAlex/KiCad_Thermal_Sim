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

    if k_norm_info is None:
        k_norm_info = {}
    if snapshot_debug is None:
        snapshot_debug = {}

    # --- Summary card data ---
    pin_w = k_norm_info.get("pin_w")
    pout_w = k_norm_info.get("pout_final_w")
    steady_rel = k_norm_info.get("steady_rel_diff")
    total_power_w = sum(
        p for _, p in pad_power if isinstance(p, (int, float))
    ) if pad_power else None

    # Compute max temperatures from T_data if available
    max_temps = {}
    if T_data is not None:
        try:
            import numpy as np
            T_arr = np.asarray(T_data, dtype=float)
            for i in range(min(len(layer_names), T_arr.shape[0])):
                max_temps[layer_names[i]] = round(float(np.max(T_arr[i])), 1)
        except Exception:
            pass

    # Energy balance indicator
    if steady_rel is not None:
        if steady_rel < 0.01:
            balance_color = "#2ecc40"
            balance_label = "Excellent"
        elif steady_rel < 0.05:
            balance_color = "#ff851b"
            balance_label = "Acceptable"
        else:
            balance_color = "#ff4136"
            balance_label = "Poor"
        balance_dot = (
            f'<span class="balance-dot" style="background:{balance_color};" '
            f'title="Relative difference between input and output power"></span>'
        )
        balance_html = (
            f'<div class="summary-card">'
            f'<h3 title="Compares total injected power (Pin) against convective heat loss (Pout) at the final time step. '
            f'A small difference indicates the simulation has reached or is near thermal equilibrium.">'
            f'Energy Balance {balance_dot} {_esc(balance_label)}</h3>'
            f'<table class="summary-tbl">'
            f'<tr><td title="Total power injected into heat source pads">P<sub>in</sub></td>'
            f'<td>{_fmt(pin_w, " W")}</td></tr>'
            f'<tr><td title="Total convective heat loss from all surfaces at the final time step">P<sub>out</sub></td>'
            f'<td>{_fmt(pout_w, " W")}</td></tr>'
            f'<tr><td title="Relative difference: |Pin - Pout| / Pin">Rel. diff.</td>'
            f'<td>{steady_rel:.2%}</td></tr>'
            f'</table></div>'
        )
    else:
        balance_html = ""

    # Max temperature summary
    max_temp_rows = ""
    if max_temps:
        for lname, tmax in max_temps.items():
            max_temp_rows += f"<tr><td>{_esc(lname)}</td><td>{tmax:.1f} &deg;C</td></tr>"
        max_temp_html = (
            f'<div class="summary-card">'
            f'<h3 title="Peak temperature on each copper layer at the final simulation time step">'
            f'Peak Temperatures</h3>'
            f'<table class="summary-tbl">'
            f'<tr><th>Layer</th><th>T<sub>max</sub></th></tr>'
            f'{max_temp_rows}</table></div>'
        )
    else:
        max_temp_html = ""

    # Overview card
    sim_time = settings.get("time")
    amb_val = settings.get("amb", ambient)
    n_layers = len(layer_names)
    overview_html = (
        f'<div class="summary-card">'
        f'<h3>Overview</h3>'
        f'<table class="summary-tbl">'
        f'<tr><td title="Number of copper layers in the simulation">Layers</td>'
        f'<td>{n_layers} ({", ".join(_esc(n) for n in layer_names)})</td></tr>'
        f'<tr><td title="Ambient temperature used as initial condition and convective reference">Ambient</td>'
        f'<td>{_fmt(amb_val, " &deg;C") if amb_val is not None else "n/a"}</td></tr>'
        f'<tr><td title="Total simulation time">Sim. time</td>'
        f'<td>{_fmt(sim_time, " s") if sim_time is not None else "n/a"}</td></tr>'
        f'<tr><td title="Total power injected across all heat source pads">Total power</td>'
        f'<td>{_fmt(total_power_w, " W") if total_power_w is not None else "n/a"}</td></tr>'
        f'<tr><td title="Board thickness from KiCad stackup definition">Board thickness</td>'
        f'<td>{_esc(_fmt(board_thick_mm, " mm"))}</td></tr>'
        f'</table></div>'
    )

    # --- Settings table ---
    settings_rows = "\n".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(str(v))}</td></tr>"
        for k, v in settings.items()
    )

    # --- Power per pad ---
    pad_rows = "\n".join(
        f"<tr><td>{_esc(name)}</td><td>{_esc(_fmt(power, ' W'))}</td></tr>"
        for name, power in pad_power
    )

    # --- Combined stackup table ---
    t_fr4_eff_mm = []
    if isinstance(k_norm_info, dict):
        t_fr4_eff_mm = k_norm_info.get("t_fr4_eff_per_plane_mm") or []
    stackup_rows = ""
    for i, lname in enumerate(layer_names):
        cu_th = _fmt(copper_thicknesses[i], " mm") if i < len(copper_thicknesses) else "n/a"
        t_eff = _fmt(t_fr4_eff_mm[i], " mm") if i < len(t_fr4_eff_mm) else "n/a"
        stackup_rows += (
            f"<tr><td>{_esc(lname)}</td>"
            f"<td>{_esc(cu_th)}</td>"
            f"<td>{_esc(t_eff)}</td></tr>"
        )
        if i < len(gaps_used):
            stackup_rows += (
                f'<tr class="gap-row"><td class="gap-label">'
                f'{_esc(lname)} &rarr; {_esc(layer_names[i + 1])}</td>'
                f'<td colspan="2">{_esc(_fmt(gaps_used[i], " mm"))}</td></tr>'
            )

    fallback_note = ""
    if gap_fallback_used:
        fallback_note = (
            '<p class="note" title="The KiCad stackup did not contain per-interface '
            'dielectric gap data. A uniform gap was calculated by dividing the total '
            'board thickness by the number of copper layer interfaces.">'
            'Note: Uniform dielectric gap fallback was used (stackup gaps not available).</p>'
        )

    # --- Debug tables ---
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

    # Build snapshot gallery section
    snapshot_gallery_html = ""
    if snapshot_files:
        snap_items = ""
        for t_val, fpath in snapshot_files:
            if fpath and os.path.isfile(str(fpath)):
                rel_name = os.path.basename(str(fpath))
                snap_items += (
                    f'<div class="snap-item">'
                    f'<img src="{_esc(rel_name)}" loading="lazy">'
                    f'<p>t = {t_val:.1f} s</p></div>'
                )
        if snap_items:
            snapshot_gallery_html = (
                f'<h2>Time-Series Snapshots</h2>'
                f'<div class="snap-gallery">{snap_items}</div>'
            )

    html_body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ThermalSim Report</title>
  <style>
    :root {{
      --bg: #fafbfc; --card-bg: #ffffff; --border: #e1e4e8;
      --text: #24292e; --text-muted: #586069; --accent: #0366d6;
      --header-bg: #f6f8fa;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
           background: var(--bg); color: var(--text); line-height: 1.5;
           max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 1.6em; font-weight: 600; margin-bottom: 4px; color: var(--text); }}
    h2 {{ font-size: 1.2em; font-weight: 600; margin: 28px 0 12px 0; color: var(--text);
         padding-bottom: 6px; border-bottom: 1px solid var(--border); }}
    h3 {{ font-size: 1em; font-weight: 600; margin-bottom: 8px; color: var(--text); }}
    .subtitle {{ color: var(--text-muted); font-size: 0.85em; margin-bottom: 20px; }}
    .summary-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
    .summary-card {{ background: var(--card-bg); border: 1px solid var(--border);
                     border-radius: 6px; padding: 16px; flex: 1; min-width: 220px; }}
    .summary-tbl {{ width: 100%; border-collapse: collapse; }}
    .summary-tbl td, .summary-tbl th {{
      padding: 4px 8px; border: none; font-size: 0.9em;
      border-bottom: 1px solid #f0f0f0;
    }}
    .summary-tbl th {{ text-align: left; color: var(--text-muted); font-weight: 500; font-size: 0.85em; }}
    .summary-tbl td:first-child {{ color: var(--text-muted); white-space: nowrap; }}
    .summary-tbl td:last-child {{ font-weight: 500; }}
    .balance-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                    margin-right: 4px; vertical-align: middle; }}
    table.data-tbl {{ border-collapse: collapse; width: 100%; margin-bottom: 16px;
                      background: var(--card-bg); border: 1px solid var(--border);
                      border-radius: 6px; overflow: hidden; }}
    .data-tbl th {{ background: var(--header-bg); padding: 8px 12px; text-align: left;
                    font-weight: 600; font-size: 0.85em; color: var(--text-muted);
                    border-bottom: 1px solid var(--border); }}
    .data-tbl td {{ padding: 6px 12px; border-bottom: 1px solid #f0f0f0; font-size: 0.9em; }}
    .data-tbl tr:last-child td {{ border-bottom: none; }}
    .gap-row td {{ background: #f8f9fa; color: var(--text-muted); font-style: italic; font-size: 0.85em; }}
    .gap-label {{ padding-left: 24px !important; }}
    .note {{ color: var(--text-muted); font-size: 0.85em; font-style: italic; margin: 8px 0; }}
    .preview-section {{ margin: 16px 0; }}
    .preview-section img {{ max-width: 100%; height: auto; border: 1px solid var(--border);
                            border-radius: 4px; }}
    details {{ margin-bottom: 16px; }}
    details summary {{ cursor: pointer; font-weight: 600; font-size: 1.05em; color: var(--text);
                       padding: 8px 0; user-select: none; }}
    details summary:hover {{ color: var(--accent); }}
    details .debug-tbl {{ width: 100%; border-collapse: collapse; font-size: 0.85em;
                          background: var(--card-bg); border: 1px solid var(--border); }}
    details .debug-tbl td {{ padding: 4px 10px; border-bottom: 1px solid #f0f0f0;
                             font-family: 'SFMono-Regular', Consolas, monospace; }}
    details .debug-tbl td:first-child {{ color: var(--text-muted); white-space: nowrap; }}
    .snap-gallery {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:16px; }}
    .snap-item {{ flex:1; min-width:280px; max-width:48%; }}
    .snap-item img {{ max-width:100%; height:auto; border:1px solid var(--border); border-radius:4px; }}
    .snap-item p {{ font-size:.85em; color:var(--text-muted); margin-top:4px; text-align:center; }}
    [title] {{ cursor: help; border-bottom: 1px dotted var(--text-muted); }}
    table [title] {{ border-bottom: none; }}
    .summary-card h3[title] {{ border-bottom: none; cursor: help; }}
  </style>
</head>
<body>
  <h1>ThermalSim Report</h1>
  <p class="subtitle">2.5D transient thermal simulation &mdash; generated by ThermalSim plugin</p>

  <!-- Summary Cards -->
  <div class="summary-row">
    {overview_html}
    {max_temp_html}
    {balance_html}
  </div>

  <!-- Geometry Preview -->
  <h2>Geometry Preview</h2>
  <div class="preview-section">
    {"<img src='" + _esc(preview_rel) + "' alt='Geometry preview showing copper, vias, and heat sources'>" if preview_rel else "<p class='note'>Preview image not available.</p>"}
  </div>

  <!-- Interactive Heatmap -->
  {interactive_html}

  <!-- Time-Series Snapshots -->
  {snapshot_gallery_html}

  <!-- PCB Stackup -->
  <h2 title="Physical layer stack used in the finite-volume thermal model">PCB Stackup</h2>
  <table class="data-tbl">
    <tr>
      <th title="Copper layer name from KiCad board setup">Layer</th>
      <th title="Copper foil thickness (from stackup or default 35 &micro;m)">Cu Thickness</th>
      <th title="Effective dielectric thickness assigned to this copper plane, averaged from adjacent interface gaps">t<sub>fr4,eff</sub></th>
    </tr>
    {stackup_rows}
  </table>
  {fallback_note}

  <!-- Simulation Settings -->
  <h2>Simulation Settings</h2>
  <table class="data-tbl">
    <tr><th>Parameter</th><th>Value</th></tr>
    {settings_rows}
  </table>

  <!-- Power per Pad -->
  <h2 title="Thermal power injected at each selected pad">Heat Sources</h2>
  <table class="data-tbl">
    <tr>
      <th title="Pad reference (Footprint-PadNumber)">Pad</th>
      <th title="Injected power in watts (constant or PWL profile)">Power</th>
    </tr>
    {pad_rows if pad_rows else "<tr><td colspan='2' style='color:var(--text-muted)'>No heat sources defined.</td></tr>"}
  </table>

  <!-- Debug (collapsible) -->
  <details>
    <summary>Solver Debug Information</summary>
    <p class="note" style="margin:8px 0 12px 0;">Internal solver parameters, matrix dimensions, and performance counters.</p>
    <table class="debug-tbl">
      {k_norm_rows}
    </table>
    {"<h3 style='margin:16px 0 8px 0;'>Snapshot Debug</h3><table class='debug-tbl'>" + snapshot_debug_rows + "</table>" if snapshot_debug_rows else ""}
  </details>
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
