"""
Stackup configurations for testing.

This module provides pre-defined stackup configurations that match
the expected output of parse_stackup_from_board_file().
"""


STACKUP_2_LAYER = {
    "board_thickness_mm": 1.6,
    "copper": [
        {"order": 0, "name": "F.Cu", "layer_id": 0, "thickness_mm": 0.035},
        {"order": 2, "name": "B.Cu", "layer_id": 31, "thickness_mm": 0.035},
    ],
    "dielectrics": [
        {"order": 1, "name": "dielectric", "type": "prepreg", "thickness_mm": 1.53},
    ],
    "copper_ids": [0, 31],
    "dielectric_gaps_mm": [1.53],
    "file_layer_count": 3,
}


STACKUP_4_LAYER = {
    "board_thickness_mm": 1.6,
    "copper": [
        {"order": 0, "name": "F.Cu", "layer_id": 0, "thickness_mm": 0.035},
        {"order": 2, "name": "In1.Cu", "layer_id": 1, "thickness_mm": 0.035},
        {"order": 4, "name": "In2.Cu", "layer_id": 2, "thickness_mm": 0.035},
        {"order": 6, "name": "B.Cu", "layer_id": 31, "thickness_mm": 0.035},
    ],
    "dielectrics": [
        {"order": 1, "name": "dielectric", "type": "prepreg", "thickness_mm": 0.2},
        {"order": 3, "name": "dielectric", "type": "core", "thickness_mm": 1.0},
        {"order": 5, "name": "dielectric", "type": "prepreg", "thickness_mm": 0.2},
    ],
    "copper_ids": [0, 1, 2, 31],
    "dielectric_gaps_mm": [0.2, 1.0, 0.2],
    "file_layer_count": 7,
}


STACKUP_6_LAYER = {
    "board_thickness_mm": 1.6,
    "copper": [
        {"order": 0, "name": "F.Cu", "layer_id": 0, "thickness_mm": 0.035},
        {"order": 2, "name": "In1.Cu", "layer_id": 1, "thickness_mm": 0.0175},
        {"order": 4, "name": "In2.Cu", "layer_id": 2, "thickness_mm": 0.0175},
        {"order": 6, "name": "In3.Cu", "layer_id": 3, "thickness_mm": 0.0175},
        {"order": 8, "name": "In4.Cu", "layer_id": 4, "thickness_mm": 0.0175},
        {"order": 10, "name": "B.Cu", "layer_id": 31, "thickness_mm": 0.035},
    ],
    "dielectrics": [
        {"order": 1, "name": "dielectric", "type": "prepreg", "thickness_mm": 0.1},
        {"order": 3, "name": "dielectric", "type": "core", "thickness_mm": 0.36},
        {"order": 5, "name": "dielectric", "type": "prepreg", "thickness_mm": 0.6},
        {"order": 7, "name": "dielectric", "type": "core", "thickness_mm": 0.36},
        {"order": 9, "name": "dielectric", "type": "prepreg", "thickness_mm": 0.1},
    ],
    "copper_ids": [0, 1, 2, 3, 4, 31],
    "dielectric_gaps_mm": [0.1, 0.36, 0.6, 0.36, 0.1],
    "file_layer_count": 11,
}


def make_simple_stackup(layer_count: int, board_thick_mm: float = 1.6):
    """
    Create a simple uniform stackup configuration.

    Parameters
    ----------
    layer_count : int
        Number of copper layers.
    board_thick_mm : float
        Total board thickness in mm.

    Returns
    -------
    dict
        Stackup configuration dictionary.
    """
    cu_thick = 0.035
    total_cu = layer_count * cu_thick
    total_dielectric = board_thick_mm - total_cu
    gap_count = layer_count - 1
    gap_per = total_dielectric / gap_count if gap_count > 0 else 0

    layer_names = ["F.Cu"]
    layer_ids = [0]
    for i in range(layer_count - 2):
        layer_names.append(f"In{i+1}.Cu")
        layer_ids.append(i + 1)
    layer_names.append("B.Cu")
    layer_ids.append(31)

    copper = []
    for idx, (name, lid) in enumerate(zip(layer_names, layer_ids)):
        copper.append({
            "order": idx * 2,
            "name": name,
            "layer_id": lid,
            "thickness_mm": cu_thick
        })

    dielectrics = []
    for i in range(gap_count):
        dielectrics.append({
            "order": i * 2 + 1,
            "name": "dielectric",
            "type": "core" if i == gap_count // 2 else "prepreg",
            "thickness_mm": gap_per
        })

    return {
        "board_thickness_mm": board_thick_mm,
        "copper": copper,
        "dielectrics": dielectrics,
        "copper_ids": layer_ids,
        "dielectric_gaps_mm": [gap_per] * gap_count,
        "file_layer_count": layer_count + gap_count,
    }
