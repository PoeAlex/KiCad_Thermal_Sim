"""Tests for electrical current-path Joule heating."""

import numpy as np

from ThermalSim.electrical_solver import (
    CurrentTerminal,
    ElectricalConfig,
    _build_net_edges,
    build_electrical_connectivity_report,
    build_electrical_supernet_map,
    prepare_electrical_geometry,
    solve_electrical_heating,
)
from tests.mocks.pcbnew_mock import (
    B_Cu,
    EDA_RECT,
    F_Cu,
    In1_Cu,
    MockBoard,
    MockFootprint,
    MockPad,
    MockTrack,
    MockVia,
    VECTOR2I,
)


def _config(layers=None, rows=4, cols=12, res=1.0, copper_thickness_m=35e-6):
    return ElectricalConfig(
        copper_ids=layers or [F_Cu],
        rows=rows,
        cols=cols,
        x_min=0.0,
        y_min=0.0,
        res=res,
        t_cu=np.array([copper_thickness_m] * len(layers or [F_Cu]), dtype=np.float64),
    )


def _pad(x_mm, y_mm, number, net_code=1, net_name="PWR", layer=F_Cu):
    x = int(x_mm * 1e6)
    y = int(y_mm * 1e6)
    return MockPad(
        position=VECTOR2I(x, y),
        layer=layer,
        bbox=EDA_RECT(x, y, 100000, 100000),
        net_code=net_code,
        net_name=net_name,
        number=number,
    )


def test_unbalanced_current_blocks_solve():
    """A net with non-zero sum(I) must fail validation."""
    pad_a = _pad(0.25, 1.25, "1")
    pad_b = _pad(9.25, 1.25, "2")
    board = MockBoard(footprints=[MockFootprint(pads=[pad_a, pad_b])])

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(pad_a, "J1-1", "PWR", 1, 9.0),
            CurrentTerminal(pad_b, "J2-1", "PWR", 1, -6.0),
        ],
        _config(),
    )

    assert not result.valid
    assert any("not current-balanced" in err for err in result.errors)


def test_disconnected_pads_block_solve():
    """Balanced pads on separate copper islands must fail validation."""
    pad_a = _pad(0.25, 1.25, "1")
    pad_b = _pad(9.25, 1.25, "2")
    board = MockBoard(footprints=[MockFootprint(pads=[pad_a, pad_b])])

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(pad_a, "J1-1", "PWR", 1, 1.0),
            CurrentTerminal(pad_b, "J2-1", "PWR", 1, -1.0),
        ],
        _config(),
    )

    assert not result.valid
    assert any("not electrically connected" in err for err in result.errors)


def test_track_loss_matches_i_squared_r_order():
    """A simple one-cell-wide copper strip should produce I^2R loss."""
    pad_a = _pad(0.25, 1.25, "1")
    pad_b = _pad(9.25, 1.25, "2")
    track = MockTrack(
        layer=F_Cu,
        bbox=EDA_RECT(250000, 1250000, 9100000, 100000),
        start=VECTOR2I(500000, 1500000),
        end=VECTOR2I(9500000, 1500000),
        width=1000000,
        net_code=1,
        net_name="PWR",
    )
    board = MockBoard(
        footprints=[MockFootprint(pads=[pad_a, pad_b])],
        tracks=[track],
    )

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(pad_a, "J1-1", "PWR", 1, 1.0),
            CurrentTerminal(pad_b, "J2-1", "PWR", 1, -1.0),
        ],
        _config(),
    )

    expected_r = 9.0 * (1.724e-8 / 35e-6)
    assert result.valid, result.errors
    assert result.total_loss_w > 0.0
    np.testing.assert_allclose(result.total_loss_w, expected_r, rtol=0.25)


def test_diagonal_track_is_one_electrical_path():
    """Diagonal raster cells must preserve a continuous copper trace."""
    source = _pad(0.25, 0.25, "1")
    sink = _pad(9.25, 9.25, "2")
    track = MockTrack(
        layer=F_Cu,
        bbox=EDA_RECT(250000, 250000, 9000000, 9000000),
        start=VECTOR2I(250000, 250000),
        end=VECTOR2I(9250000, 9250000),
        width=100000,
        net_code=1,
        net_name="PWR",
    )
    board = MockBoard(
        footprints=[MockFootprint(pads=[source, sink])], tracks=[track]
    )

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(source, "J1-1", "PWR", 1, 1.0),
            CurrentTerminal(sink, "J2-2", "PWR", 1, -1.0),
        ],
        _config(rows=12, cols=12),
    )

    assert result.valid, result.errors


def test_diagonal_edges_require_primitive_connectivity_permission():
    """Diagonal occupancy alone creates no edge; an explicit bit does."""
    mask = np.zeros((1, 2, 2), dtype=bool)
    mask[0, 0, 0] = mask[0, 1, 1] = True
    node_ids = np.full(mask.shape, -1, dtype=np.int64)
    node_ids[0, 0, 0], node_ids[0, 1, 1] = 0, 1
    diag_right = np.zeros((1, 1, 1), dtype=bool)
    diag_left = np.zeros_like(diag_right)
    right = np.zeros((1, 2, 1), dtype=bool)
    down = np.zeros((1, 1, 2), dtype=bool)
    empty_vias = np.zeros((2, 2), dtype=bool)

    disconnected = _build_net_edges(
        mask, empty_vias, node_ids, _config(rows=2, cols=2), right, down, diag_right, diag_left
    )
    assert disconnected[0].size == 0
    assert disconnected[5] == 1

    diag_right[0, 0, 0] = True
    connected = _build_net_edges(
        mask, empty_vias, node_ids, _config(rows=2, cols=2), right, down,
        diag_right, diag_left
    )
    assert connected[0].size == 1
    assert connected[4] == 1


def test_cardinal_edges_require_primitive_connectivity_permission():
    """Cardinal raster adjacency alone must not make an electrical edge."""
    mask = np.zeros((1, 2, 2), dtype=bool)
    mask[0, 0, :] = True
    node_ids = np.array([[[0, 1], [-1, -1]]], dtype=np.int64)
    right = np.zeros((1, 2, 1), dtype=bool)
    down = np.zeros((1, 1, 2), dtype=bool)
    diagonal = np.zeros((1, 1, 1), dtype=bool)
    vias = np.zeros((2, 2), dtype=bool)
    args = (mask, vias, node_ids, _config(rows=2, cols=2), right, down, diagonal, diagonal)

    assert _build_net_edges(*args)[0].size == 0
    right[0, 0, 0] = True
    edges = _build_net_edges(*args)
    assert edges[0].tolist() == [0]


def test_separate_same_net_tracks_sharing_a_cell_remain_disconnected():
    """Primitive ownership preserves clearance even when both traces hit one cell."""
    source = _pad(0.25, 0.25, "1")
    sink = _pad(2.25, 0.55, "2")
    tracks = [
        MockTrack(F_Cu, EDA_RECT(200000, 200000, 2100000, 100000),
                  VECTOR2I(250000, 250000), VECTOR2I(2350000, 250000), 100000, 1, "PWR"),
        MockTrack(F_Cu, EDA_RECT(200000, 500000, 2100000, 100000),
                  VECTOR2I(250000, 550000), VECTOR2I(2350000, 550000), 100000, 1, "PWR"),
    ]
    board = MockBoard(footprints=[MockFootprint(pads=[source, sink])], tracks=tracks)
    result = solve_electrical_heating(
        board,
        [CurrentTerminal(source, "J1-1", "PWR", 1, 1.0),
         CurrentTerminal(sink, "J2-1", "PWR", 1, -1.0)],
        _config(rows=2, cols=3, res=1.0),
    )

    assert not result.valid
    assert any("not electrically connected" in error for error in result.errors)


def test_connectivity_report_separates_disconnected_same_net_terminals():
    """The report exposes actual graph components, not just same-net coloring."""
    source, sink = _pad(0.25, 0.25, "1"), _pad(2.25, 0.55, "2")
    tracks = [
        MockTrack(F_Cu, EDA_RECT(200000, 200000, 2100000, 100000),
                  VECTOR2I(250000, 250000), VECTOR2I(2350000, 250000), 100000, 1, "PWR"),
        MockTrack(F_Cu, EDA_RECT(200000, 500000, 2100000, 100000),
                  VECTOR2I(250000, 550000), VECTOR2I(2350000, 550000), 100000, 1, "PWR"),
    ]
    board = MockBoard(footprints=[MockFootprint(pads=[source, sink])], tracks=tracks)
    terminals = [CurrentTerminal(source, "J1-1", "PWR", 1, 1.0),
                 CurrentTerminal(sink, "J2-1", "PWR", 1, -1.0)]
    config = _config(rows=2, cols=3, res=1.0)
    prepared = prepare_electrical_geometry(board, terminals, config)

    report = build_electrical_connectivity_report(prepared, terminals, config)

    net = report["nets"][0]
    assert net["component_count"] == 2
    assert not net["terminals_connected"]
    assert net["terminals"][0]["component_ids"] != net["terminals"][1]["component_ids"]


def test_arc_raster_follows_midpoint_instead_of_start_end_chord():
    """The electrical raster follows the KiCad arc rather than its chord."""
    source = _pad(1.0, 2.0, "1")
    sink = _pad(3.0, 2.0, "2")
    arc = MockTrack(
        F_Cu, EDA_RECT(950000, 1950000, 2100000, 1100000),
        VECTOR2I(1000000, 2000000), VECTOR2I(3000000, 2000000),
        100000, 1, "PWR", mid=VECTOR2I(2000000, 3000000),
    )
    board = MockBoard(footprints=[MockFootprint(pads=[source, sink])], tracks=[arc])
    config = _config(rows=40, cols=40, res=0.1)
    prepared = prepare_electrical_geometry(
        board,
        [CurrentTerminal(source, "J1-1", "PWR", 1, 1.0),
         CurrentTerminal(sink, "J2-1", "PWR", 1, -1.0)],
        config,
    )
    track_raster = next(p for p in prepared.rasters["C:1"].primitives if p.kind == "Track")
    rows, _ = np.nonzero(track_raster.copper_mask[0])

    assert rows.size
    assert track_raster.row0 + np.max(rows) >= 27


def test_separate_diagonal_same_net_tracks_are_not_connected_by_raster_corner():
    """Same-net occupancy in diagonal cells is not a physical connection."""
    source = _pad(0.5, 0.5, "1")
    sink = _pad(1.5, 1.5, "2")
    tracks = [
        MockTrack(F_Cu, EDA_RECT(300000, 450000, 400000, 100000),
                  VECTOR2I(300000, 500000), VECTOR2I(700000, 500000), 100000, 1, "PWR"),
        MockTrack(F_Cu, EDA_RECT(1300000, 1450000, 400000, 100000),
                  VECTOR2I(1300000, 1500000), VECTOR2I(1700000, 1500000), 100000, 1, "PWR"),
    ]
    board = MockBoard(footprints=[MockFootprint(pads=[source, sink])], tracks=tracks)

    result = solve_electrical_heating(
        board,
        [CurrentTerminal(source, "J1-1", "PWR", 1, 1.0),
         CurrentTerminal(sink, "J2-2", "PWR", 1, -1.0)],
        _config(rows=3, cols=3),
    )

    assert not result.valid
    assert any("not electrically connected" in error for error in result.errors)


def test_ideal_20mm_strip_reports_kicad_calculator_order():
    """A 20 mm x 2.0792 mm top trace at 5 A should report KiCad-order diagnostics."""
    width_mm = 2.0792
    length_mm = 20.0
    current_a = 5.0
    pad_size_mm = 0.05
    y_mid_mm = 1.5
    pad_a = MockPad(
        position=VECTOR2I(int(0.025e6), int(y_mid_mm * 1e6)),
        layer=F_Cu,
        bbox=EDA_RECT(0, int((y_mid_mm - pad_size_mm / 2) * 1e6), int(pad_size_mm * 1e6), int(pad_size_mm * 1e6)),
        net_code=1,
        net_name="PWR",
        number="1",
    )
    pad_b = MockPad(
        position=VECTOR2I(int((length_mm - 0.025) * 1e6), int(y_mid_mm * 1e6)),
        layer=F_Cu,
        bbox=EDA_RECT(
            int((length_mm - pad_size_mm) * 1e6),
            int((y_mid_mm - pad_size_mm / 2) * 1e6),
            int(pad_size_mm * 1e6),
            int(pad_size_mm * 1e6),
        ),
        net_code=1,
        net_name="PWR",
        number="2",
    )
    track = MockTrack(
        layer=F_Cu,
        bbox=EDA_RECT(0, int((y_mid_mm - width_mm / 2) * 1e6), int(length_mm * 1e6), int(width_mm * 1e6)),
        start=VECTOR2I(0, int(y_mid_mm * 1e6)),
        end=VECTOR2I(int(length_mm * 1e6), int(y_mid_mm * 1e6)),
        width=int(width_mm * 1e6),
        net_code=1,
        net_name="PWR",
    )
    board = MockBoard(
        footprints=[MockFootprint(pads=[pad_a, pad_b])],
        tracks=[track],
        layer_names={F_Cu: "F.Cu"},
    )
    config = ElectricalConfig(
        copper_ids=[F_Cu],
        rows=70,
        cols=410,
        x_min=0.0,
        y_min=0.0,
        res=0.05,
        t_cu=np.array([35e-6], dtype=np.float64),
        layer_names={F_Cu: "F.Cu"},
    )

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(pad_a, "J1-1", "PWR", 1, current_a),
            CurrentTerminal(pad_b, "J2-1", "PWR", 1, -current_a),
        ],
        config,
    )

    expected_r = 1.724e-8 * (length_mm * 1e-3) / ((width_mm * 1e-3) * 35e-6)
    expected_p = current_a * current_a * expected_r
    summary = result.net_summaries[0]
    assert result.valid, result.errors
    np.testing.assert_allclose(result.total_loss_w, expected_p, rtol=0.20)
    np.testing.assert_allclose(summary.effective_resistance_ohm, expected_r, rtol=0.20)
    np.testing.assert_allclose(summary.equivalent_voltage_drop_v, expected_p / current_a, rtol=0.20)
    assert summary.terminal_diagnostics[0].cell_count > 0
    assert summary.primitive_diagnostics


def test_multi_terminal_summary_uses_effective_values_without_pad_resistance():
    """Multi-terminal nets should avoid pretending there is one pad-to-pad resistance."""
    a1 = _pad(0.25, 1.25, "1")
    a2 = _pad(0.25, 2.25, "2")
    b1 = _pad(9.25, 1.25, "3")
    b2 = _pad(9.25, 2.25, "4")
    track = MockTrack(
        layer=F_Cu,
        bbox=EDA_RECT(250000, 1250000, 9100000, 1100000),
        start=VECTOR2I(500000, 1750000),
        end=VECTOR2I(9500000, 1750000),
        width=2000000,
        net_code=1,
        net_name="PWR",
    )
    board = MockBoard(
        footprints=[MockFootprint(pads=[a1, a2, b1, b2])],
        tracks=[track],
    )

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(a1, "A1", "PWR", 1, 1.0),
            CurrentTerminal(a2, "A2", "PWR", 1, 1.0),
            CurrentTerminal(b1, "B1", "PWR", 1, -1.0),
            CurrentTerminal(b2, "B2", "PWR", 1, -1.0),
        ],
        _config(rows=5, cols=12),
    )

    summary = result.net_summaries[0]
    assert result.valid, result.errors
    assert summary.effective_resistance_ohm is not None
    assert summary.equivalent_voltage_drop_v is not None
    assert summary.pad_resistance_ohm is None


def test_via_connects_current_between_layers():
    """A via should create a valid vertical path between copper layers."""
    top_pad = _pad(2.25, 1.25, "1", layer=F_Cu)
    bottom_pad = _pad(2.25, 1.25, "2", layer=B_Cu)
    via = MockVia(
        bbox=EDA_RECT(2250000, 1250000, 100000, 100000),
        layers=[F_Cu, B_Cu],
        net_code=1,
        net_name="PWR",
    )
    board = MockBoard(
        footprints=[MockFootprint(pads=[top_pad, bottom_pad])],
        tracks=[via],
        layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
    )

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(top_pad, "J1-1", "PWR", 1, 1.0),
            CurrentTerminal(bottom_pad, "J2-1", "PWR", 1, -1.0),
        ],
        _config(layers=[F_Cu, B_Cu]),
    )

    assert result.valid, result.errors
    assert result.total_loss_w > 0.0


def test_inner_layer_smd_pad_uses_its_copper_layer_set():
    """SMD pads on an inner layer must join copper on that layer."""
    pad_a = _pad(0.25, 1.25, "1", layer=F_Cu)
    pad_b = _pad(9.25, 1.25, "2", layer=F_Cu)
    pad_a._layer_set = type(pad_a.GetLayerSet())([In1_Cu])
    pad_b._layer_set = type(pad_b.GetLayerSet())([In1_Cu])
    track = MockTrack(
        layer=In1_Cu,
        bbox=EDA_RECT(250000, 1250000, 9100000, 100000),
        start=VECTOR2I(500000, 1500000),
        end=VECTOR2I(9500000, 1500000),
        width=1000000,
        net_code=1,
        net_name="PWR",
    )
    board = MockBoard(
        footprints=[MockFootprint(pads=[pad_a, pad_b])], tracks=[track]
    )

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(pad_a, "J1-1", "PWR", 1, 1.0),
            CurrentTerminal(pad_b, "J2-1", "PWR", 1, -1.0),
        ],
        _config(layers=[F_Cu, In1_Cu]),
    )

    assert result.valid, result.errors


def test_independent_nets_are_solved_separately():
    """Active nets should not share a global copper matrix."""
    a1 = _pad(0.25, 1.25, "1", net_code=1, net_name="A")
    a2 = _pad(3.25, 1.25, "2", net_code=1, net_name="A")
    b1 = _pad(6.25, 1.25, "1", net_code=2, net_name="B")
    b2 = _pad(9.25, 1.25, "2", net_code=2, net_name="B")
    track_a = MockTrack(
        layer=F_Cu,
        bbox=EDA_RECT(250000, 1250000, 3100000, 100000),
        start=VECTOR2I(500000, 1500000),
        end=VECTOR2I(3500000, 1500000),
        width=1000000,
        net_code=1,
        net_name="A",
    )
    track_b = MockTrack(
        layer=F_Cu,
        bbox=EDA_RECT(6250000, 1250000, 3100000, 100000),
        start=VECTOR2I(6500000, 1500000),
        end=VECTOR2I(9500000, 1500000),
        width=1000000,
        net_code=2,
        net_name="B",
    )
    board = MockBoard(
        footprints=[MockFootprint(pads=[a1, a2, b1, b2])],
        tracks=[track_a, track_b],
    )

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(a1, "A1", "A", 1, 1.0),
            CurrentTerminal(a2, "A2", "A", 1, -1.0),
            CurrentTerminal(b1, "B1", "B", 2, 2.0),
            CurrentTerminal(b2, "B2", "B", 2, -2.0),
        ],
        _config(),
    )

    assert result.valid, result.errors
    assert {summary.net_name for summary in result.net_summaries} == {"A", "B"}


def _net_tie_path_board():
    """Create an A--B--C path with two declared, overlapping-pad net ties."""
    source = _pad(0.25, 1.25, "1", net_code=1, net_name="A")
    sink = _pad(9.25, 1.25, "2", net_code=3, net_name="C")
    tie_a = MockPad(
        position=VECTOR2I(3500000, 1500000),
        bbox=EDA_RECT(3000000, 1000000, 1000000, 1000000),
        net_code=1, net_name="A", number="1",
    )
    tie_b_left = MockPad(
        position=VECTOR2I(3500000, 1500000),
        bbox=EDA_RECT(3000000, 1000000, 1000000, 1000000),
        net_code=2, net_name="B", number="2",
    )
    tie_b_right = MockPad(
        position=VECTOR2I(6500000, 1500000),
        bbox=EDA_RECT(6000000, 1000000, 1000000, 1000000),
        net_code=2, net_name="B", number="1",
    )
    tie_c = MockPad(
        position=VECTOR2I(6500000, 1500000),
        bbox=EDA_RECT(6000000, 1000000, 1000000, 1000000),
        net_code=3, net_name="C", number="2",
    )
    tracks = [
        MockTrack(F_Cu, EDA_RECT(250000, 1250000, 3250000, 500000), VECTOR2I(500000, 1500000), VECTOR2I(3500000, 1500000), 1000000, 1, "A"),
        MockTrack(F_Cu, EDA_RECT(3500000, 1250000, 3000000, 500000), VECTOR2I(3500000, 1500000), VECTOR2I(6500000, 1500000), 1000000, 2, "B"),
        MockTrack(F_Cu, EDA_RECT(6500000, 1250000, 3000000, 500000), VECTOR2I(6500000, 1500000), VECTOR2I(9500000, 1500000), 1000000, 3, "C"),
    ]
    return MockBoard(
        footprints=[
            MockFootprint("J1", [source]),
            MockFootprint("NT1", [tie_a, tie_b_left], [[tie_a, tie_b_left]]),
            MockFootprint("NT2", [tie_b_right, tie_c], [[tie_b_right, tie_c]]),
            MockFootprint("J2", [sink]),
        ],
        tracks=tracks,
    ), source, sink


def test_net_tie_supernet_map_is_transitive():
    """Chained declared net ties must resolve to one supernet."""
    board, _, _ = _net_tie_path_board()

    supernets = build_electrical_supernet_map(board)

    assert supernets["C:1"] is supernets["C:2"]
    assert supernets["C:2"] is supernets["C:3"]
    assert supernets["C:1"].member_names == ("A", "B", "C")


def test_net_tie_connected_terminals_solve_as_one_supernet():
    """End terminals across a net-tie chain must solve one physical path."""
    board, source, sink = _net_tie_path_board()

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(source, "J1-1", "A", 1, 1.0),
            CurrentTerminal(sink, "J2-2", "C", 3, -1.0),
        ],
        _config(),
    )

    assert result.valid, result.errors
    assert len(result.net_summaries) == 1
    assert result.net_summaries[0].net_name == "A + B + C"
    summary = result.net_summaries[0]
    assert summary.raw_net_names == ["A", "B", "C"]
    assert summary.net_tie_edge_count == 2
    assert summary.component_count_before_ties > summary.component_count_after_ties


def test_separated_net_tie_pads_bridge_the_current_path():
    """A declared net tie must connect its pads even when raster cells do not touch."""
    source = _pad(0.25, 1.25, "1", net_code=1, net_name="A")
    sink = _pad(9.25, 1.25, "2", net_code=2, net_name="C")
    tie_a = MockPad(
        position=VECTOR2I(3050000, 1300000),
        bbox=EDA_RECT(3000000, 1250000, 100000, 100000),
        net_code=1, net_name="A", number="1",
    )
    tie_c = MockPad(
        position=VECTOR2I(6050000, 1300000),
        bbox=EDA_RECT(6000000, 1250000, 100000, 100000),
        net_code=2, net_name="C", number="2",
    )
    tracks = [
        MockTrack(F_Cu, EDA_RECT(250000, 1000000, 2800000, 500000), VECTOR2I(250000, 1250000), VECTOR2I(3050000, 1250000), 500000, 1, "A"),
        MockTrack(F_Cu, EDA_RECT(6000000, 1000000, 3500000, 500000), VECTOR2I(6050000, 1250000), VECTOR2I(9250000, 1250000), 500000, 2, "C"),
    ]
    board = MockBoard(
        footprints=[
            MockFootprint("J1", [source]),
            MockFootprint("NT1", [tie_a, tie_c], [[tie_a, tie_c]]),
            MockFootprint("J2", [sink]),
        ],
        tracks=tracks,
    )

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(source, "J1-1", "A", 1, 1.0),
            CurrentTerminal(sink, "J2-2", "C", 2, -1.0),
        ],
        _config(rows=16, cols=48, res=0.25),
    )

    assert result.valid, result.errors
    assert result.net_summaries[0].net_name == "A + C"
    assert result.net_summaries[0].net_tie_edge_count == 1


def test_net_tie_raw_net_geometry_keeps_separate_nodes_at_other_overlaps():
    """Overlapping raw-net masks stay distinct except for the declared tie edge."""
    source = _pad(0.25, 1.25, "1", net_code=1, net_name="A")
    sink = _pad(9.25, 1.25, "2", net_code=2, net_name="B")
    tie_a = MockPad(
        position=VECTOR2I(5000000, 1250000),
        bbox=EDA_RECT(4750000, 1000000, 500000, 500000),
        net_code=1, net_name="A", number="3",
    )
    tie_b = MockPad(
        position=VECTOR2I(5000000, 1250000),
        bbox=EDA_RECT(4750000, 1000000, 500000, 500000),
        net_code=2, net_name="B", number="4",
    )
    tracks = [
        MockTrack(F_Cu, EDA_RECT(0, 1000000, 9500000, 500000),
                  VECTOR2I(250000, 1250000), VECTOR2I(9750000, 1250000), 500000, 1, "A"),
        MockTrack(F_Cu, EDA_RECT(0, 1000000, 9500000, 500000),
                  VECTOR2I(250000, 1250000), VECTOR2I(9750000, 1250000), 500000, 2, "B"),
    ]
    board = MockBoard(
        footprints=[MockFootprint("J1", [source]),
                    MockFootprint("NT1", [tie_a, tie_b], [[tie_a, tie_b]]),
                    MockFootprint("J2", [sink])],
        tracks=tracks,
    )

    result = solve_electrical_heating(
        board,
        [CurrentTerminal(source, "J1-1", "A", 1, 1.0),
         CurrentTerminal(sink, "J2-2", "B", 2, -1.0)],
        _config(rows=8, cols=40, res=0.25),
    )

    assert result.valid, result.errors
    summary = result.net_summaries[0]
    assert summary.copper_cell_count > 2 * 30
    assert summary.net_tie_edge_count == 1
    assert summary.component_count_before_ties > summary.component_count_after_ties


def test_unmappable_declared_net_tie_pad_blocks_solve():
    """A declared tie with a pad outside the electrical grid is reported."""
    source = _pad(0.25, 1.25, "1", net_code=1, net_name="A")
    sink = _pad(9.25, 1.25, "2", net_code=2, net_name="B")
    tie_a = _pad(3.25, 1.25, "1", net_code=1, net_name="A")
    tie_b = MockPad(
        position=VECTOR2I(20000000, 20000000),
        bbox=EDA_RECT(20000000, 20000000, 100000, 100000),
        net_code=2, net_name="B", number="2",
    )
    tracks = [
        MockTrack(F_Cu, EDA_RECT(250000, 1000000, 3000000, 500000),
                  VECTOR2I(250000, 1250000), VECTOR2I(3250000, 1250000), 500000, 1, "A"),
        MockTrack(F_Cu, EDA_RECT(3250000, 1000000, 6000000, 500000),
                  VECTOR2I(3250000, 1250000), VECTOR2I(9250000, 1250000), 500000, 2, "B"),
    ]
    board = MockBoard(
        footprints=[MockFootprint("J1", [source]),
                    MockFootprint("NT1", [tie_a, tie_b], [[tie_a, tie_b]]),
                    MockFootprint("J2", [sink])],
        tracks=tracks,
    )

    result = solve_electrical_heating(
        board,
        [CurrentTerminal(source, "J1-1", "A", 1, 1.0),
         CurrentTerminal(sink, "J2-2", "B", 2, -1.0)],
        _config(rows=4, cols=12),
    )

    assert not result.valid
    assert any("do not both map to copper" in error for error in result.errors)


def test_net_tie_supernet_requires_balanced_total_current():
    """Balance is checked across the supernet, not each member net."""
    board, source, sink = _net_tie_path_board()

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(source, "J1-1", "A", 1, 1.0),
            CurrentTerminal(sink, "J2-2", "C", 3, -0.5),
        ],
        _config(),
    )

    assert not result.valid
    assert any("A + B + C is not current-balanced" in error for error in result.errors)


def test_unrelated_overlapping_nets_still_fail_collision_validation():
    """Only declared supernet members may share rasterized copper cells."""
    a1 = _pad(0.25, 1.25, "1", net_code=1, net_name="A")
    a2 = _pad(9.25, 1.25, "2", net_code=1, net_name="A")
    b1 = _pad(0.25, 2.25, "1", net_code=2, net_name="B")
    b2 = _pad(9.25, 2.25, "2", net_code=2, net_name="B")
    tracks = [
        MockTrack(
            F_Cu, EDA_RECT(250000, 1000000, 9250000, 1000000),
            VECTOR2I(500000, 1500000), VECTOR2I(9500000, 1500000),
            2000000, 1, "A",
        ),
        MockTrack(
            F_Cu, EDA_RECT(250000, 1500000, 9250000, 1000000),
            VECTOR2I(500000, 2500000), VECTOR2I(9500000, 2500000),
            2000000, 2, "B",
        ),
    ]
    board = MockBoard(
        footprints=[MockFootprint("J1", [a1, a2, b1, b2])], tracks=tracks
    )

    result = solve_electrical_heating(
        board,
        [
            CurrentTerminal(a1, "A1", "A", 1, 1.0),
            CurrentTerminal(a2, "A2", "A", 1, -1.0),
            CurrentTerminal(b1, "B1", "B", 2, 1.0),
            CurrentTerminal(b2, "B2", "B", 2, -1.0),
        ],
        _config(),
    )

    assert not result.valid
    assert any("multiple active nets overlap" in error for error in result.errors)
