"""
Sample KiCad board content generators for testing.

This module provides functions to generate .kicad_pcb file content
for testing the stackup parser.
"""


SIMPLE_2_LAYER_STACKUP = """(kicad_pcb (version 20240101) (generator "test")
  (general
    (thickness 1.6)
  )
  (stackup
    (layer "F.Cu" 0 (type "copper") (thickness 0.035))
    (layer dielectric 1 (type "prepreg") (thickness 1.53))
    (layer "B.Cu" 2 (type "copper") (thickness 0.035))
  )
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
  )
)
"""


SIMPLE_4_LAYER_STACKUP = """(kicad_pcb (version 20240101) (generator "test")
  (general
    (thickness 1.6)
  )
  (stackup
    (layer "F.Cu" 0 (type "copper") (thickness 0.035))
    (layer dielectric 1 (type "prepreg") (thickness 0.2))
    (layer "In1.Cu" 2 (type "copper") (thickness 0.035))
    (layer dielectric 3 (type "core") (thickness 1.0))
    (layer "In2.Cu" 4 (type "copper") (thickness 0.035))
    (layer dielectric 5 (type "prepreg") (thickness 0.2))
    (layer "B.Cu" 6 (type "copper") (thickness 0.035))
  )
  (layers
    (0 "F.Cu" signal)
    (1 "In1.Cu" signal)
    (2 "In2.Cu" signal)
    (31 "B.Cu" signal)
  )
)
"""


SIMPLE_6_LAYER_STACKUP = """(kicad_pcb (version 20240101) (generator "test")
  (general
    (thickness 1.6)
  )
  (stackup
    (layer "F.Cu" 0 (type "copper") (thickness 0.035))
    (layer dielectric 1 (type "prepreg") (thickness 0.1))
    (layer "In1.Cu" 2 (type "copper") (thickness 0.0175))
    (layer dielectric 3 (type "core") (thickness 0.36))
    (layer "In2.Cu" 4 (type "copper") (thickness 0.0175))
    (layer dielectric 5 (type "prepreg") (thickness 0.6))
    (layer "In3.Cu" 6 (type "copper") (thickness 0.0175))
    (layer dielectric 7 (type "core") (thickness 0.36))
    (layer "In4.Cu" 8 (type "copper") (thickness 0.0175))
    (layer dielectric 9 (type "prepreg") (thickness 0.1))
    (layer "B.Cu" 10 (type "copper") (thickness 0.035))
  )
  (layers
    (0 "F.Cu" signal)
    (1 "In1.Cu" signal)
    (2 "In2.Cu" signal)
    (3 "In3.Cu" signal)
    (4 "In4.Cu" signal)
    (31 "B.Cu" signal)
  )
)
"""


STACKUP_WITH_SUBLAYERS = """(kicad_pcb (version 20240101) (generator "test")
  (general
    (thickness 1.6)
  )
  (stackup
    (layer "F.Cu" 0 (type "copper") (thickness 0.035))
    (layer dielectric 1 (type "prepreg")
      (sublayer (thickness 0.1))
      (sublayer (thickness 0.1))
    )
    (layer "B.Cu" 2 (type "copper") (thickness 0.035))
  )
)
"""


NO_STACKUP_BOARD = """(kicad_pcb (version 20240101) (generator "test")
  (general
    (thickness 1.6)
  )
  (layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
  )
)
"""


def generate_kicad_pcb_content(
    layer_count: int = 2,
    board_thickness_mm: float = 1.6,
    copper_thickness_mm: float = 0.035,
    custom_gaps: list = None
) -> str:
    """
    Generate .kicad_pcb file content with specified parameters.

    Parameters
    ----------
    layer_count : int
        Number of copper layers (2, 4, or 6).
    board_thickness_mm : float
        Total board thickness in mm.
    copper_thickness_mm : float
        Copper thickness per layer in mm.
    custom_gaps : list, optional
        Custom dielectric gap thicknesses.

    Returns
    -------
    str
        Complete .kicad_pcb file content.
    """
    if layer_count == 2:
        gap = board_thickness_mm - 2 * copper_thickness_mm
        stackup = f'''(stackup
    (layer "F.Cu" 0 (type "copper") (thickness {copper_thickness_mm}))
    (layer dielectric 1 (type "core") (thickness {gap:.4f}))
    (layer "B.Cu" 2 (type "copper") (thickness {copper_thickness_mm}))
  )'''
        layers = '''(layers
    (0 "F.Cu" signal)
    (31 "B.Cu" signal)
  )'''

    elif layer_count == 4:
        if custom_gaps:
            g1, g2, g3 = custom_gaps[:3]
        else:
            remaining = board_thickness_mm - 4 * copper_thickness_mm
            g1 = remaining * 0.15
            g2 = remaining * 0.7
            g3 = remaining * 0.15
        stackup = f'''(stackup
    (layer "F.Cu" 0 (type "copper") (thickness {copper_thickness_mm}))
    (layer dielectric 1 (type "prepreg") (thickness {g1:.4f}))
    (layer "In1.Cu" 2 (type "copper") (thickness {copper_thickness_mm}))
    (layer dielectric 3 (type "core") (thickness {g2:.4f}))
    (layer "In2.Cu" 4 (type "copper") (thickness {copper_thickness_mm}))
    (layer dielectric 5 (type "prepreg") (thickness {g3:.4f}))
    (layer "B.Cu" 6 (type "copper") (thickness {copper_thickness_mm}))
  )'''
        layers = '''(layers
    (0 "F.Cu" signal)
    (1 "In1.Cu" signal)
    (2 "In2.Cu" signal)
    (31 "B.Cu" signal)
  )'''

    elif layer_count == 6:
        if custom_gaps:
            g1, g2, g3, g4, g5 = custom_gaps[:5]
        else:
            remaining = board_thickness_mm - 6 * copper_thickness_mm
            g = remaining / 5
            g1 = g2 = g3 = g4 = g5 = g
        stackup = f'''(stackup
    (layer "F.Cu" 0 (type "copper") (thickness {copper_thickness_mm}))
    (layer dielectric 1 (type "prepreg") (thickness {g1:.4f}))
    (layer "In1.Cu" 2 (type "copper") (thickness {copper_thickness_mm}))
    (layer dielectric 3 (type "core") (thickness {g2:.4f}))
    (layer "In2.Cu" 4 (type "copper") (thickness {copper_thickness_mm}))
    (layer dielectric 5 (type "prepreg") (thickness {g3:.4f}))
    (layer "In3.Cu" 6 (type "copper") (thickness {copper_thickness_mm}))
    (layer dielectric 7 (type "core") (thickness {g4:.4f}))
    (layer "In4.Cu" 8 (type "copper") (thickness {copper_thickness_mm}))
    (layer dielectric 9 (type "prepreg") (thickness {g5:.4f}))
    (layer "B.Cu" 10 (type "copper") (thickness {copper_thickness_mm}))
  )'''
        layers = '''(layers
    (0 "F.Cu" signal)
    (1 "In1.Cu" signal)
    (2 "In2.Cu" signal)
    (3 "In3.Cu" signal)
    (4 "In4.Cu" signal)
    (31 "B.Cu" signal)
  )'''

    else:
        raise ValueError(f"Unsupported layer count: {layer_count}")

    return f'''(kicad_pcb (version 20240101) (generator "test")
  (general
    (thickness {board_thickness_mm})
  )
  {stackup}
  {layers}
)
'''
