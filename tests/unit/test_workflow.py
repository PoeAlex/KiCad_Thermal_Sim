"""Tests for shared workflow, grid, cancellation, and cache behavior."""

from ThermalSim.electrical_solver import CurrentTerminal
from ThermalSim.thermal_plugin import (
    _coarsen_grid_resolution,
    _estimate_simulation_area,
    _estimate_solver_grid,
)
from ThermalSim.thermal_solver import SolverConfig, _build_phase_plan
from ThermalSim.workflow import (
    BoardSnapshot,
    CancellationToken,
    GeometryCache,
    GridEstimate,
    PreflightResult,
    SimulationArtifacts,
    ThermalFactorizationCache,
    ThermalOperatorCache,
    geometry_cache_key,
)
from tests.mocks.pcbnew_mock import (
    B_Cu,
    EDA_RECT,
    F_Cu,
    MockBoard,
    MockFootprint,
    MockPad,
    MockTrack,
    MockZone,
    VECTOR2I,
)


def test_area_limit_uses_original_maximum_after_moving_minimum():
    bbox = EDA_RECT(0, 0, 100_000_000, 50_000_000)
    pad = MockPad(position=VECTOR2I(90_000_000, 25_000_000), layer=F_Cu)
    grid = _estimate_solver_grid(
        bbox,
        1.0,
        {"limit_area": True, "pad_dist_mm": 20.0},
        2,
        [pad],
    )

    assert grid.x_min_mm == 70.0
    assert grid.width_mm == 30.0
    assert grid.x_min_mm + grid.width_mm == 100.0


def test_grid_estimate_reports_total_nodes_and_complexity():
    grid = GridEstimate(0.1, 0.2, 0, 0, 10, 10, 500, 500, 4, True, False, 200000, 100000)
    assert grid.base_cells == 250000
    assert grid.nodes == 1000000
    assert grid.complexity == "High"


def test_current_aware_area_includes_complete_active_net_geometry():
    """A routed current net must remain inside the cropped domain."""
    bbox = EDA_RECT(0, 0, 100_000_000, 50_000_000)
    source = MockPad(position=VECTOR2I(10_000_000, 10_000_000))
    terminal_a = MockPad(
        position=VECTOR2I(20_000_000, 5_000_000), net_code=7, net_name="LOAD"
    )
    terminal_b = MockPad(
        position=VECTOR2I(30_000_000, 5_000_000), net_code=7, net_name="LOAD"
    )
    routed_track = MockTrack(
        bbox=EDA_RECT(20_000_000, 4_500_000, 60_000_000, 1_000_000),
        net_code=7,
        net_name="LOAD",
    )
    board = MockBoard(
        footprints=[MockFootprint(pads=[terminal_a, terminal_b])],
        tracks=[routed_track],
    )
    terminals = [
        CurrentTerminal(terminal_a, "J1-1", "LOAD", 7, 2.0),
        CurrentTerminal(terminal_b, "J2-1", "LOAD", 7, -2.0),
    ]

    area = _estimate_simulation_area(
        board,
        bbox,
        {"area_mode": "active", "area_margin_mm": 5.0, "res": 1.0},
        power_pads=[source],
        terminals=terminals,
    )

    assert area.mode == "active"
    assert area.x_min_mm <= 3.5
    assert area.x_min_mm + area.width_mm >= 86.0
    assert area.active_net_names == ("LOAD",)
    assert area.area_fraction < 1.0


def test_current_zone_covering_board_reports_little_area_saving():
    """A full-board active zone must not be clipped to terminal pads."""
    bbox = EDA_RECT(0, 0, 100_000_000, 50_000_000)
    pad_a = MockPad(position=VECTOR2I(10_000_000, 10_000_000), net_code=1, net_name="GND")
    pad_b = MockPad(position=VECTOR2I(20_000_000, 10_000_000), net_code=1, net_name="GND")
    board = MockBoard(
        footprints=[MockFootprint(pads=[pad_a, pad_b])],
        zones=[MockZone(bbox=bbox, net_code=1, net_name="GND")],
    )
    terminals = [
        CurrentTerminal(pad_a, "J1-1", "GND", 1, 1.0),
        CurrentTerminal(pad_b, "J2-1", "GND", 1, -1.0),
    ]

    area = _estimate_simulation_area(
        board, bbox,
        {"area_mode": "active", "area_margin_mm": 10.0, "res": 0.5},
        terminals=terminals,
    )

    assert area.area_fraction == 1.0
    assert any("almost the full board" in item for item in area.warnings)


def test_balanced_detail_budget_is_layer_aware():
    """Four layers should map the Balanced node budget to legacy defaults."""
    _, _, expert, max_cells, target_cells = _coarsen_grid_resolution(
        300.0,
        75.0,
        0.1,
        {"grid_detail_level": "balanced"},
        layer_count=4,
    )

    assert expert is False
    assert max_cells == 200000
    assert target_cells == 100000


def test_geometry_cache_key_ignores_non_geometry_settings():
    snapshot = BoardSnapshot("board", "abc", (0, 0, 10, 10), (0, 2), 1, 1, 1)
    grid = GridEstimate(0.5, 0.5, 0, 0, 10, 10, 24, 24, 2, False, False, 200000, 100000)
    base = {"ignore_traces": False, "time": 1.0, "amb": 25.0}
    changed = {"ignore_traces": False, "time": 100.0, "amb": 80.0}
    assert geometry_cache_key(snapshot, grid, base, ["U1:1"]) == geometry_cache_key(
        snapshot, grid, changed, ["U1:1"]
    )


def test_geometry_cache_invalidates_geometry_changes():
    cache = GeometryCache()
    cache.put("first", object())
    assert cache.get("other") is None
    cache.clear()
    assert cache.get("first") is None


def test_thermal_operator_cache_reuses_only_matching_key():
    cache = ThermalOperatorCache()
    operator = object()
    cache.put("thermal-inputs", operator)
    assert cache.get("thermal-inputs") is operator
    assert cache.get("changed-boundary-condition") is None
    cache.clear()
    assert cache.get("thermal-inputs") is None


def test_factorization_cache_releases_evicted_value():
    class CachedValue:
        released = False

        def release(self):
            self.released = True

    cache = ThermalFactorizationCache()
    first = CachedValue()
    cache.put("first", first)
    cache.put("second", CachedValue())
    assert first.released is True
    cache.clear()


def test_cancellation_token_is_thread_safe_signal():
    token = CancellationToken()
    assert token.cancelled is False
    token.cancel()
    assert token.cancelled is True


def test_preflight_status_prioritizes_errors():
    result = PreflightResult(warnings=["warning"])
    assert result.status == "Warning"
    result.errors.append("error")
    assert result.status == "Blocked"
    assert result.ready is False


def test_simulation_artifacts_carry_compact_result_summary():
    artifacts = SimulationArtifacts(
        report_path="report.html",
        run_dir="results",
        status="success",
        elapsed_s=8.2,
        max_temp_c=56.3,
    )
    assert artifacts.elapsed_s == 8.2
    assert artifacts.max_temp_c == 56.3


def test_large_auto_pardiso_plan_uses_single_uniform_phase():
    config = SolverConfig(1.0, 25.0, 1.0 / 120.0, 120, use_pardiso=True, time_stepping="auto")
    mode, plan = _build_phase_plan(config, 600000, True)
    assert mode == "uniform"
    assert len(plan) == 1
    assert plan[0][2] == 95
    assert plan[0][0]["derivative_start"] is True


def test_small_auto_plan_keeps_legacy_phases():
    config = SolverConfig(1.0, 25.0, 1.0 / 120.0, 120, use_pardiso=True, time_stepping="auto")
    mode, plan = _build_phase_plan(config, 10000, True)
    assert mode == "multi_phase"
    assert [item[0]["name"] for item in plan] == ["A", "B", "C"]
