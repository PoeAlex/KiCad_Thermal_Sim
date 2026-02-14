"""
Test board configurations for current path analysis.

Provides MockBoard setups with tracks, pads, and zones wired to
specific nets for electrical conductivity testing.
"""

from tests.mocks.pcbnew_mock import (
    MockBoard, MockFootprint, MockPad, MockTrack, MockVia, MockZone,
    VECTOR2I, EDA_RECT, F_Cu, B_Cu, PAD_ATTRIB_SMD, PAD_ATTRIB_PTH,
    FromMM,
)


def simple_copper_bar_board(length_mm=10.0, width_mm=2.0, res_mm=0.5):
    """
    Create a board with a straight copper track connecting two SMD pads.

    The track runs horizontally on F.Cu from pad A to pad B.
    All elements share net_code=1.

    Parameters
    ----------
    length_mm : float
        Track length in mm.
    width_mm : float
        Track width in mm.
    res_mm : float
        Expected grid resolution (used to size the board).

    Returns
    -------
    dict
        Keys: board, pad_a, pad_b, net_code, copper_ids, length_mm, width_mm
    """
    net_code = 1
    half_w = FromMM(width_mm / 2)
    pad_size = FromMM(width_mm)

    # Pad A at left end
    pad_a_pos = VECTOR2I(FromMM(0.0), FromMM(width_mm / 2))
    pad_a = MockPad(
        position=pad_a_pos, layer=F_Cu, attribute=PAD_ATTRIB_SMD,
        bbox=EDA_RECT(pad_a_pos.x - pad_size // 2, pad_a_pos.y - pad_size // 2,
                      pad_size, pad_size),
        selected=True, net_code=net_code, net_name="VCC", number="1",
    )

    # Pad B at right end
    pad_b_pos = VECTOR2I(FromMM(length_mm), FromMM(width_mm / 2))
    pad_b = MockPad(
        position=pad_b_pos, layer=F_Cu, attribute=PAD_ATTRIB_SMD,
        bbox=EDA_RECT(pad_b_pos.x - pad_size // 2, pad_b_pos.y - pad_size // 2,
                      pad_size, pad_size),
        selected=True, net_code=net_code, net_name="VCC", number="2",
    )

    # Track connecting pads
    track = MockTrack(
        layer=F_Cu,
        start=VECTOR2I(FromMM(0.0), FromMM(width_mm / 2)),
        end=VECTOR2I(FromMM(length_mm), FromMM(width_mm / 2)),
        width=FromMM(width_mm),
        net_code=net_code,
    )

    fp_a = MockFootprint(reference="J1", pads=[pad_a])
    fp_b = MockFootprint(reference="J2", pads=[pad_b])

    board = MockBoard(
        filename="test_bar.kicad_pcb",
        footprints=[fp_a, fp_b],
        tracks=[track],
        zones=[],
        layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
    )

    return {
        "board": board,
        "pad_a": pad_a,
        "pad_b": pad_b,
        "net_code": net_code,
        "copper_ids": [F_Cu, B_Cu],
        "length_mm": length_mm,
        "width_mm": width_mm,
    }


def two_layer_via_board():
    """
    Create a board where current flows: F.Cu track -> via -> B.Cu track.

    Layout (top view, 10mm x 4mm):
    - F.Cu track from (0,2) to (5,2), width 2mm, net 1
    - Via at (5,2), connecting F.Cu and B.Cu, net 1
    - B.Cu track from (5,2) to (10,2), width 2mm, net 1
    - Pad A at (0,2) on F.Cu, Pad B at (10,2) on B.Cu

    Returns
    -------
    dict
        Keys: board, pad_a, pad_b, net_code, copper_ids
    """
    net_code = 1
    pad_size = FromMM(2.0)

    pad_a_pos = VECTOR2I(FromMM(0.0), FromMM(2.0))
    pad_a = MockPad(
        position=pad_a_pos, layer=F_Cu, attribute=PAD_ATTRIB_SMD,
        bbox=EDA_RECT(pad_a_pos.x - pad_size // 2, pad_a_pos.y - pad_size // 2,
                      pad_size, pad_size),
        selected=True, net_code=net_code, net_name="VCC", number="1",
    )

    pad_b_pos = VECTOR2I(FromMM(10.0), FromMM(2.0))
    pad_b = MockPad(
        position=pad_b_pos, layer=B_Cu, attribute=PAD_ATTRIB_SMD,
        bbox=EDA_RECT(pad_b_pos.x - pad_size // 2, pad_b_pos.y - pad_size // 2,
                      pad_size, pad_size),
        selected=True, net_code=net_code, net_name="VCC", number="2",
    )

    track_top = MockTrack(
        layer=F_Cu,
        start=VECTOR2I(FromMM(0.0), FromMM(2.0)),
        end=VECTOR2I(FromMM(5.0), FromMM(2.0)),
        width=FromMM(2.0),
        net_code=net_code,
    )

    track_bot = MockTrack(
        layer=B_Cu,
        start=VECTOR2I(FromMM(5.0), FromMM(2.0)),
        end=VECTOR2I(FromMM(10.0), FromMM(2.0)),
        width=FromMM(2.0),
        net_code=net_code,
    )

    via_pos = VECTOR2I(FromMM(5.0), FromMM(2.0))
    via = MockVia(
        bbox=EDA_RECT(FromMM(4.5), FromMM(1.5), FromMM(1.0), FromMM(1.0)),
        layers=[F_Cu, B_Cu],
        net_code=net_code,
        position=via_pos,
    )

    fp_a = MockFootprint(reference="J1", pads=[pad_a])
    fp_b = MockFootprint(reference="J2", pads=[pad_b])

    board = MockBoard(
        filename="test_via.kicad_pcb",
        footprints=[fp_a, fp_b],
        tracks=[track_top, track_bot, via],
        zones=[],
        layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
    )

    return {
        "board": board,
        "pad_a": pad_a,
        "pad_b": pad_b,
        "net_code": net_code,
        "copper_ids": [F_Cu, B_Cu],
    }
