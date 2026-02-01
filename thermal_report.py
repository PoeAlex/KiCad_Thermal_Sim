"""
HTML report generation for ThermalSim.

This module generates an HTML report summarizing the thermal simulation
results, including settings, stackup information, and embedded images.
"""

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
    snapshot_files=None
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
</body>
</html>
"""

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_body)
    except Exception:
        return None
    return report_path
