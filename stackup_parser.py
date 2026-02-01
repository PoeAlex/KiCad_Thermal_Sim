"""
Stackup parsing from KiCad PCB files.

This module provides functions to parse the stackup information directly
from saved .kicad_pcb files using S-expression parsing. This approach
avoids limitations of the KiCad SWIG API for stackup access.

Functions
---------
parse_stackup_from_board_file
    Parse copper layers and dielectric thicknesses from a .kicad_pcb file.
format_stackup_report_um
    Generate a human-readable stackup report with thicknesses in micrometers.
"""

import re


def _sexpr_extract_from_index(s, i):
    """
    Extract a balanced S-expression block starting at index i.

    Parameters
    ----------
    s : str
        The input string containing S-expressions.
    i : int
        Starting index where '(' is expected.

    Returns
    -------
    tuple
        (block_str, next_index) where block_str is the matched '(...)'
        block and next_index is the position after the closing ')'.
        Returns (None, None) if no balanced block is found.
    """
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return s[i:j+1], j+1
    return None, None


def _sexpr_extract_block(s, token, start=0):
    """
    Find and extract the first S-expression block starting with a token.

    Parameters
    ----------
    s : str
        The input string to search.
    token : str
        The token to search for (e.g., "stackup", "layer").
    start : int, optional
        Starting position for the search. Default is 0.

    Returns
    -------
    str or None
        The complete '(token ...)' block if found, otherwise None.
    """
    key = "(" + token
    i = s.find(key, start)
    if i < 0:
        return None
    blk, _ = _sexpr_extract_from_index(s, i)
    return blk


def _sexpr_find_all_blocks(s, token):
    """
    Find all S-expression blocks starting with a given token.

    Parameters
    ----------
    s : str
        The input string to search.
    token : str
        The token to search for.

    Returns
    -------
    list of str
        List of all '(token ...)' blocks found in the string.
    """
    key = "(" + token
    blocks = []
    i = 0
    while True:
        i = s.find(key, i)
        if i < 0:
            break
        blk, end = _sexpr_extract_from_index(s, i)
        if not blk:
            break
        blocks.append(blk)
        i = end
    return blocks


def parse_stackup_from_board_file(board):
    """
    Parse stackup and thickness information from a saved .kicad_pcb file.

    This function reads the board file directly to extract stackup
    information, which is more reliable than using the SWIG API for
    older KiCad versions.

    Parameters
    ----------
    board : pcbnew.BOARD
        The KiCad board object. Must have been saved to disk.

    Returns
    -------
    dict
        Dictionary containing:
        - board_thickness_mm : float or None
            Overall board thickness from the general section.
        - copper : list of dict
            Copper layer information, each with keys:
            'order', 'name', 'layer_id', 'thickness_mm'.
        - dielectrics : list of dict
            Dielectric layer information, each with keys:
            'order', 'name', 'type', 'thickness_mm'.
        - copper_ids : list of int
            Layer IDs of copper layers in stackup order (top to bottom).
        - dielectric_gaps_mm : list of float
            Dielectric thickness between adjacent copper layers.
        - file_layer_count : int
            Total number of layer blocks found in stackup.
        - error : str (only if parsing failed)
            Error message if parsing was unsuccessful.

    Notes
    -----
    The function handles various KiCad file format variations including:
    - Sublayer thickness accumulation for prepreg layers
    - Both quoted and unquoted layer names
    - Dielectric layers specified as named or anonymous blocks
    """
    fn = ""
    try:
        fn = board.GetFileName()
    except Exception:
        fn = ""
    if not fn:
        return {"error": "Board has no filename (save board first)."}

    try:
        txt = open(fn, "r", encoding="utf-8", errors="ignore").read()
    except Exception as e:
        return {"error": f"Failed to read board file: {e}"}

    # General thickness
    board_thickness_mm = None
    general = _sexpr_extract_block(txt, "general")
    if general:
        m = re.search(r"\(thickness\s+([0-9.]+)\)", general)
        if m:
            try:
                board_thickness_mm = float(m.group(1))
            except Exception:
                board_thickness_mm = None

    stackup = _sexpr_extract_block(txt, "stackup")
    if not stackup:
        return {"error": "No (stackup ...) found in file. Save board after editing stackup."}

    layer_blocks = _sexpr_find_all_blocks(stackup, "layer")

    copper = []
    dielectrics = []

    for idx, lb in enumerate(layer_blocks):
        # Formats observed:
        # (layer "F.Cu" 3 (type "copper") (thickness 0.035) ...)
        # (layer "F.SilkS" (type "Top Silk Screen"))
        # (layer dielectric 4 (type "core") (thickness ...))
        name = None
        order = None

        m = re.match(r'\(layer\s+"([^"]+)"(?:\s+(\d+))?', lb)
        if m:
            name = m.group(1)
            if m.group(2):
                try:
                    order = int(m.group(2))
                except Exception:
                    order = None
        else:
            m2 = re.match(r'\(layer\s+dielectric(?:\s+(\d+))?', lb)
            if m2:
                name = "dielectric"
                if m2.group(1):
                    try:
                        order = int(m2.group(1))
                    except Exception:
                        order = None

        if order is None:
            # If no explicit numeric order, keep file order
            order = idx

        mt = re.search(r'\(type\s+"([^"]+)"\)', lb)
        typ = (mt.group(1).strip().lower() if mt else "")

        # Thickness: prefer sum of sublayers if present
        th_mm = None
        sublayers = _sexpr_find_all_blocks(lb, "sublayer")
        if sublayers:
            vals = []
            for sb in sublayers:
                vals += [float(x) for x in re.findall(r"\(thickness\s+([0-9.]+)\)", sb)]
            th_mm = sum(vals) if vals else None
        else:
            mm = re.search(r"\(thickness\s+([0-9.]+)\)", lb)
            if mm:
                try:
                    th_mm = float(mm.group(1))
                except Exception:
                    th_mm = None

        # Categorize
        if typ == "copper" and name:
            try:
                lid = int(board.GetLayerID(name))
            except Exception:
                lid = None
            copper.append({
                "order": order,
                "name": name,
                "layer_id": lid,
                "thickness_mm": th_mm
            })
        elif ("core" in typ) or ("prepreg" in typ) or (typ == "dielectric") or (name == "dielectric"):
            dielectrics.append({
                "order": order,
                "name": name or "dielectric",
                "type": typ or "dielectric",
                "thickness_mm": th_mm
            })

    # Sort copper by explicit stackup order (NOT by layer id!)
    copper.sort(key=lambda d: d["order"])

    copper_ids = [d["layer_id"] for d in copper if isinstance(d.get("layer_id"), int)]

    # Dielectric gaps between adjacent copper layers
    dielectric_gaps_mm = []
    for a, b in zip(copper, copper[1:]):
        oa, ob = a["order"], b["order"]
        if oa > ob:
            oa, ob = ob, oa
        gap = 0.0
        found = False
        for d in dielectrics:
            if oa < d["order"] < ob and d.get("thickness_mm") is not None:
                gap += float(d["thickness_mm"])
                found = True
        dielectric_gaps_mm.append(gap if found else 0.0)

    return {
        "board_thickness_mm": board_thickness_mm,
        "copper": copper,
        "dielectrics": dielectrics,
        "copper_ids": copper_ids,
        "dielectric_gaps_mm": dielectric_gaps_mm,
        "file_layer_count": len(layer_blocks),
    }


def format_stackup_report_um(stack):
    """
    Generate a human-readable stackup report with thicknesses in micrometers.

    Parameters
    ----------
    stack : dict or None
        Stackup dictionary as returned by parse_stackup_from_board_file.
        May contain an 'error' key if parsing failed.

    Returns
    -------
    str
        Multi-line formatted string showing:
        - Board thickness
        - Copper layers with thicknesses
        - Dielectric gaps between copper layers

    Examples
    --------
    >>> stack = parse_stackup_from_board_file(board)
    >>> print(format_stackup_report_um(stack))
    Board thickness (general): 1.600 mm
    Copper layers (top->bottom):
      F.Cu      35.0 um    id=0
      B.Cu      35.0 um    id=31
    Dielectric gaps between copper layers:
      F.Cu -> B.Cu: 1530.0 um
    """
    if not stack or stack.get("error"):
        return f"Stackup: {stack.get('error', 'unavailable')}"
    lines = []
    bt = stack.get("board_thickness_mm", None)
    if bt is not None:
        lines.append(f"Board thickness (general): {bt:.3f} mm")
    copper = stack.get("copper", [])
    if copper:
        lines.append("Copper layers (top->bottom):")
        for c in copper:
            th = c.get("thickness_mm", None)
            th_um = (th * 1000.0) if isinstance(th, (int, float)) else None
            th_s = f"{th_um:.1f} um" if th_um is not None else "(n/a)"
            lid = c.get("layer_id", None)
            lid_s = str(lid) if isinstance(lid, int) else "?"
            lines.append(f"  {c['name']:<8s}  {th_s:<10s}  id={lid_s}")
    gaps = stack.get("dielectric_gaps_mm", [])
    if copper and gaps and len(gaps) == max(0, len(copper)-1):
        lines.append("Dielectric gaps between copper layers:")
        for i, g in enumerate(gaps):
            a = copper[i]["name"]
            b = copper[i+1]["name"]
            lines.append(f"  {a} -> {b}: {g*1000.0:.1f} um")
    return "\n".join(lines)
