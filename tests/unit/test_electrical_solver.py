"""Tests for electrical current-path Joule heating."""

from types import SimpleNamespace
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import ThermalSim.electrical_solver as electrical_solver
from ThermalSim.electrical_solver import (
    CurrentTerminal,
    ElectricalConfig,
    _build_relevant_net_masks,
    _solve_electrical_system,
    solve_electrical_heating,
)
from tests.mocks.pcbnew_mock import (
    B_Cu,
    EDA_RECT,
    F_Cu,
    MockBoard,
    MockFootprint,
    MockPad,
    MockTrack,
    MockVia,
    MockZone,
    PAD_ATTRIB_PTH,
    VECTOR2I,
)


def _config(layers=None, rows=4, cols=12, res=1.0):
    return ElectricalConfig(
        copper_ids=layers or [F_Cu],
        rows=rows,
        cols=cols,
        x_min=0.0,
        y_min=0.0,
        res=res,
        t_cu=np.array([35e-6] * len(layers or [F_Cu]), dtype=np.float64),
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


def _build_grid_laplacian(rows=6, cols=8):
    """Build a small grounded 2D resistor-network SPD Laplacian and RHS."""
    n_total = rows * cols
    edges_i = []
    edges_j = []
    edges_g = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if c + 1 < cols:
                edges_i.append(idx)
                edges_j.append(idx + 1)
                edges_g.append(12.5)
            if r + 1 < rows:
                edges_i.append(idx)
                edges_j.append(idx + cols)
                edges_g.append(12.5)
    ei = np.asarray(edges_i, dtype=np.int64)
    ej = np.asarray(edges_j, dtype=np.int64)
    eg = np.asarray(edges_g, dtype=np.float64)
    lap = sp.coo_matrix(
        (
            np.concatenate([eg, eg, -eg, -eg]),
            (
                np.concatenate([ei, ej, ei, ej]),
                np.concatenate([ei, ej, ej, ei]),
            ),
        ),
        shape=(n_total, n_total),
    ).tocsr()
    solve_nodes = np.arange(1, n_total, dtype=np.int64)
    sub_lap = lap[solve_nodes][:, solve_nodes].tocsr()
    sub_rhs = np.zeros(solve_nodes.size, dtype=np.float64)
    sub_rhs[0] = 2.0
    sub_rhs[-1] = -2.0
    return sub_lap, sub_rhs


def test_local_bbox_mapped_cell_counts_exact_for_all_primitives(monkeypatch):
    """BBox-sliced counting matches exact newly-occupied cells for pads, vias, tracks, zones."""
    config = ElectricalConfig(
        copper_ids=[F_Cu, B_Cu],
        rows=40,
        cols=50,
        x_min=0.0,
        y_min=0.0,
        res=1.0,
        t_cu=np.array([35e-6, 35e-6], dtype=np.float64),
        layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
    )

    # 1) Two partially overlapping SMD pads on F.Cu:
    #    pad1 covers cols 2..4 (3 cols) x rows 2..4 (3 rows) = 9 cells
    #    pad2 covers cols 4..6 (3 cols) x rows 2..4 (3 rows) = 9 cells, 3 already occupied -> 6 new
    smd_pad_1 = MockPad(
        position=VECTOR2I(3000000, 3000000),
        layer=F_Cu,
        bbox=EDA_RECT(2000000, 2000000, 2100000, 2100000),
        net_code=1,
        net_name="PWR",
        number="1",
    )
    smd_pad_2 = MockPad(
        position=VECTOR2I(5000000, 3000000),
        layer=F_Cu,
        bbox=EDA_RECT(4000000, 2000000, 2100000, 2100000),
        net_code=1,
        net_name="PWR",
        number="2",
    )
    # 2) PTH pad at cols 10..11 x rows 2..3 (2x2 = 4 cells):
    #    adds 4 cells to via_mask + 4 cells to F.Cu + 4 cells to B.Cu = 12 newly occupied cells
    pth_pad = MockPad(
        position=VECTOR2I(10500000, 2500000),
        layer=F_Cu,
        attribute=PAD_ATTRIB_PTH,
        bbox=EDA_RECT(10000000, 2000000, 1100000, 1100000),
        net_code=1,
        net_name="PWR",
        number="3",
    )
    # 3) Off-board SMD pad (negative/out-of-range bbox) must contribute 0 cells safely
    offboard_pad = MockPad(
        position=VECTOR2I(-5000000, -5000000),
        layer=F_Cu,
        bbox=EDA_RECT(-6000000, -6000000, 1000000, 1000000),
        net_code=1,
        net_name="PWR",
        number="4",
    )
    # 4) Via overlapping 1 column of pth_pad: cols 11..12 x rows 2..3 (2x2 = 4 cells per mask)
    #    col 11 rows 2..3 (2 cells) already filled in via_mask, F.Cu, B.Cu -> 2 new per mask = 6 total
    via = MockVia(
        bbox=EDA_RECT(11000000, 2000000, 1100000, 1100000),
        layers=[F_Cu, B_Cu],
        net_code=1,
        net_name="PWR",
    )
    # 5) Track on F.Cu bridging smd_pad_2 (col 6) to pth_pad (col 10) along row 3 (y=3.5 mm):
    #    cols 6..10 at row 3; cols 6 and 10 are already occupied -> cols 7, 8, 9 are newly occupied (3 cells)
    track = MockTrack(
        layer=F_Cu,
        bbox=EDA_RECT(6000000, 3100000, 4100000, 800000),
        start=VECTOR2I(6500000, 3500000),
        end=VECTOR2I(10500000, 3500000),
        width=1000000,
        net_code=1,
        net_name="PWR",
    )
    # 6) Filled zone on F.Cu and B.Cu: bbox x in [20.0, 24.1], y in [10.0, 13.1].
    #    Cell centers inside bbox: x in {20.5, 21.5, 22.5, 23.5} (4 cols) x y in {10.5, 11.5, 12.5} (3 rows) = 12 cells/layer.
    #    Second zone on B.Cu only: bbox x in [23.0, 26.1], y in [10.0, 13.1] -> x in {23.5, 24.5, 25.5},
    #    where x=23.5 is already filled -> 2 new cols x 3 rows = 6 new cells.
    #    Plus an off-board zone contributing 0 cells.
    zone_1 = MockZone(
        layers=[F_Cu, B_Cu],
        bbox=EDA_RECT(20000000, 10000000, 4100000, 3100000),
        filled=True,
        net_code=1,
        net_name="PWR",
    )
    zone_2 = MockZone(
        layers=[B_Cu],
        bbox=EDA_RECT(23000000, 10000000, 3100000, 3100000),
        filled=True,
        net_code=1,
        net_name="PWR",
    )
    offboard_zone = MockZone(
        layers=[F_Cu],
        bbox=EDA_RECT(-8000000, -8000000, 1000000, 1000000),
        filled=True,
        net_code=1,
        net_name="PWR",
    )

    board = MockBoard(
        footprints=[MockFootprint(pads=[smd_pad_1, smd_pad_2, pth_pad, offboard_pad])],
        tracks=[via, track],
        zones=[zone_1, zone_2, offboard_zone],
        layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
    )

    # Spy on np.count_nonzero to ensure full-grid (40x50 = 2000) scans are never performed
    seen_sizes = []
    real_count_nonzero = np.count_nonzero

    def _spy_count_nonzero(arr, *args, **kwargs):
        seen_sizes.append(np.asarray(arr).size)
        return real_count_nonzero(arr, *args, **kwargs)

    monkeypatch.setattr(electrical_solver.np, "count_nonzero", _spy_count_nonzero)

    net_masks, via_masks, primitive_summaries, collision_count = _build_relevant_net_masks(
        board, config, {"C:1"}
    )

    assert collision_count == 0
    assert seen_sizes, "Expected local count_nonzero calls"
    assert max(seen_sizes) < config.rows * config.cols

    by_type_layer = {
        (item.primitive_type, item.layer): item
        for item in primitive_summaries["C:1"]
    }
    assert by_type_layer[("Pad", "F.Cu")].mapped_cell_count == 9 + 6
    assert by_type_layer[("Pad", "All copper")].mapped_cell_count == 12
    assert by_type_layer[("Via/PTH", "All copper")].mapped_cell_count == 12
    assert by_type_layer[("Via/PTH", "F.Cu -> B.Cu")].mapped_cell_count == 6
    assert by_type_layer[("Track", "F.Cu")].mapped_cell_count == 3
    assert by_type_layer[("Zone", "F.Cu")].mapped_cell_count == 12
    assert by_type_layer[("Zone", "B.Cu")].mapped_cell_count == 12 + 6

    # Total occupied cells in net_masks + via_masks must equal sum of unique additions
    expected_total_cells = (
        9 + 6  # SMD pads on F.Cu
        + 12   # PTH pad (via_mask + F.Cu + B.Cu)
        + 6    # Via (via_mask + F.Cu + B.Cu)
        + 3    # Track on F.Cu
        + 12   # zone_1 on F.Cu
        + 12 + 6  # zone_1 + zone_2 on B.Cu
    )
    actual_total_cells = int(real_count_nonzero(net_masks["C:1"])) + int(
        real_count_nonzero(via_masks["C:1"])
    )
    assert actual_total_cells == expected_total_cells


def test_solve_electrical_system_iterative_pcg_and_fallback(monkeypatch):
    """Large systems use PCG and fall back to CSC spsolve on failure."""
    sub_lap, sub_rhs = _build_grid_laplacian(rows=8, cols=10)
    expected = spla.spsolve(sub_lap.tocsc(), sub_rhs)

    # 1) Small system (< threshold) uses CSC spsolve directly without calling cg or warnings
    cg_calls = []
    spsolve_formats = []
    real_cg = spla.cg
    real_spsolve = spla.spsolve

    def spy_cg(*args, **kwargs):
        cg_calls.append(True)
        return real_cg(*args, **kwargs)

    def spy_spsolve(matrix, rhs, *args, **kwargs):
        spsolve_formats.append(matrix.format)
        return real_spsolve(matrix, rhs, *args, **kwargs)

    monkeypatch.setattr(electrical_solver.spla, "cg", spy_cg)
    monkeypatch.setattr(electrical_solver.spla, "spsolve", spy_spsolve)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sol_small = _solve_electrical_system(sub_lap, sub_rhs, iterative_threshold=1000)
    np.testing.assert_allclose(sol_small, expected, rtol=1e-12, atol=1e-12)
    assert cg_calls == []
    assert spsolve_formats == ["csc"]
    assert not any(issubclass(w.category, sp.SparseEfficiencyWarning) for w in caught)

    # 2) Large system (>= threshold) solves via PCG without calling spsolve
    cg_calls.clear()
    spsolve_formats.clear()
    sol_pcg = _solve_electrical_system(sub_lap, sub_rhs, iterative_threshold=10)
    np.testing.assert_allclose(sol_pcg, expected, rtol=1e-7, atol=1e-10)
    assert cg_calls == [True]
    assert spsolve_formats == []

    # 3) PCG non-convergence safely falls back to CSC spsolve
    spsolve_formats.clear()
    monkeypatch.setattr(
        electrical_solver.spla,
        "cg",
        lambda a, b, **kwargs: (np.zeros_like(b), 4000),
    )
    sol_nonconv = _solve_electrical_system(sub_lap, sub_rhs, iterative_threshold=10)
    np.testing.assert_allclose(sol_nonconv, expected, rtol=1e-12, atol=1e-12)
    assert spsolve_formats == ["csc"]

    # 4) PCG reporting info=0 with bad residual safely falls back to CSC spsolve
    spsolve_formats.clear()
    monkeypatch.setattr(
        electrical_solver.spla,
        "cg",
        lambda a, b, **kwargs: (expected * 0.5, 0),
    )
    sol_bad_res = _solve_electrical_system(sub_lap, sub_rhs, iterative_threshold=10)
    np.testing.assert_allclose(sol_bad_res, expected, rtol=1e-12, atol=1e-12)
    assert spsolve_formats == ["csc"]


def test_end_to_end_joule_loss_matches_between_direct_and_pcg(monkeypatch):
    """End-to-end potentials and Joule loss match between direct spsolve and iterative PCG."""
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
    terminals = [
        CurrentTerminal(pad_a, "J1-1", "PWR", 1, 2.5),
        CurrentTerminal(pad_b, "J2-1", "PWR", 1, -2.5),
    ]
    config = _config()

    monkeypatch.setattr(electrical_solver, "_ELECTRICAL_ITERATIVE_NODE_THRESHOLD", 100_000)
    res_direct = solve_electrical_heating(board, terminals, config)

    monkeypatch.setattr(electrical_solver, "_ELECTRICAL_ITERATIVE_NODE_THRESHOLD", 2)
    res_pcg = solve_electrical_heating(board, terminals, config)

    assert res_direct.valid and res_pcg.valid
    np.testing.assert_allclose(res_pcg.q_joule, res_direct.q_joule, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(
        res_pcg.total_loss_w, res_direct.total_loss_w, rtol=1e-7, atol=1e-12
    )
    np.testing.assert_allclose(
        res_pcg.net_summaries[0].pad_voltage_drop_v,
        res_direct.net_summaries[0].pad_voltage_drop_v,
        rtol=1e-7,
        atol=1e-12,
    )
