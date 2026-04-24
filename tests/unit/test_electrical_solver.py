"""Tests for electrical current-path Joule heating."""

import numpy as np

from ThermalSim.electrical_solver import (
    CurrentTerminal,
    ElectricalConfig,
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
