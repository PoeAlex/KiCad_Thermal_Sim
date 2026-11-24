"""Tests for shared workflow, grid, cancellation, and cache behavior."""

import numpy as np

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
    get_process_memory_mb,
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
    """Four layers should map the Balanced node budget (3.0M / 2.55M) across layers."""
    _, _, expert, max_cells, target_cells = _coarsen_grid_resolution(
        300.0,
        75.0,
        0.1,
        {"grid_detail_level": "balanced"},
        layer_count=4,
    )

    assert expert is False
    assert max_cells == 750_000
    assert target_cells == 637_500


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


def test_geometry_cache_can_persist_between_instances(tmp_path):
    cache_dir = tmp_path / "geometry-cache"
    first = GeometryCache(
        persistent=True,
        max_bytes=1024 * 1024,
        cache_dir=str(cache_dir),
    )
    value = {"mask": np.arange(12).reshape(3, 4)}
    first.put("abc123", value)

    second = GeometryCache(
        persistent=True,
        max_bytes=1024 * 1024,
        cache_dir=str(cache_dir),
    )
    loaded = second.get("abc123")

    np.testing.assert_array_equal(loaded["mask"], value["mask"])


def test_geometry_cache_prunes_old_files(tmp_path):
    cache = GeometryCache(
        persistent=True,
        max_bytes=600,
        cache_dir=str(tmp_path / "geometry-cache"),
    )
    cache.put("first", {"data": b"x" * 400})
    cache.put("second", {"data": b"y" * 400})

    files = list((tmp_path / "geometry-cache").glob("*.json"))
    assert len(files) == 1
    assert files[0].name == "second.json"


def test_geometry_cache_persists_geometry_state_as_npz(tmp_path):
    from ThermalSim.geometry_mapper import GeometryState

    cache_dir = tmp_path / "geometry-cache"
    first = GeometryCache(
        persistent=True,
        max_bytes=10 * 1024 * 1024,
        cache_dir=str(cache_dir),
    )
    state = GeometryState(
        copper_mask=np.array([[[True, False], [False, True]], [[False, True], [True, False]]], dtype=bool),
        via_map=np.array([[1.0, 0.0], [0.0, 2.5]], dtype=np.float64),
        heatsink_mask=np.array([[True, True], [False, False]], dtype=bool),
        area_mask=np.array([[True, True], [True, False]], dtype=bool),
        x_min=1.25,
        y_min=-3.5,
        res=0.25,
        rows=2,
        cols=2,
        x_centers_iu=np.array([1_375_000, 1_625_000], dtype=np.int64),
        y_centers_iu=np.array([-3_375_000, -3_125_000], dtype=np.int64),
    )
    first.put("geom_with_area", state)

    state_no_area = GeometryState(
        copper_mask=state.copper_mask,
        via_map=state.via_map,
        heatsink_mask=state.heatsink_mask,
        area_mask=None,
        x_min=0.0,
        y_min=0.0,
        res=0.5,
        rows=2,
        cols=2,
        x_centers_iu=state.x_centers_iu,
        y_centers_iu=state.y_centers_iu,
    )
    first.put("geom_no_area", state_no_area)

    assert (cache_dir / "geom_with_area.npz").is_file()
    assert (cache_dir / "geom_no_area.npz").is_file()
    assert list(cache_dir.glob("*.pickle")) == []

    second = GeometryCache(
        persistent=True,
        max_bytes=10 * 1024 * 1024,
        cache_dir=str(cache_dir),
    )
    loaded = second.get("geom_with_area")
    assert isinstance(loaded, GeometryState)
    np.testing.assert_array_equal(loaded.copper_mask, state.copper_mask)
    np.testing.assert_array_equal(loaded.via_map, state.via_map)
    np.testing.assert_array_equal(loaded.heatsink_mask, state.heatsink_mask)
    np.testing.assert_array_equal(loaded.area_mask, state.area_mask)
    np.testing.assert_array_equal(loaded.x_centers_iu, state.x_centers_iu)
    np.testing.assert_array_equal(loaded.y_centers_iu, state.y_centers_iu)
    assert loaded.x_min == state.x_min
    assert loaded.y_min == state.y_min
    assert loaded.res == state.res
    assert loaded.rows == state.rows
    assert loaded.cols == state.cols

    loaded_no_area = second.get("geom_no_area")
    assert isinstance(loaded_no_area, GeometryState)
    assert loaded_no_area.area_mask is None


def test_geometry_cache_never_loads_legacy_pickle_and_prunes_it(tmp_path):
    cache_dir = tmp_path / "geometry-cache"
    cache_dir.mkdir()
    legacy_pickle = cache_dir / "legacy_key.pickle"
    legacy_pickle.write_bytes(b"x" * 500)

    cache = GeometryCache(
        persistent=True,
        max_bytes=600,
        cache_dir=str(cache_dir),
    )
    assert cache.get("legacy_key") is None

    cache.put("fresh_key", {"data": "y" * 400})
    assert not legacy_pickle.exists()
    assert (cache_dir / "fresh_key.json").is_file()


def test_geometry_cache_removes_corrupt_files(tmp_path):
    cache_dir = tmp_path / "geometry-cache"
    cache_dir.mkdir()
    bad_npz = cache_dir / "bad_npz.npz"
    bad_npz.write_bytes(b"not a valid zip archive")
    bad_json = cache_dir / "bad_json.json"
    bad_json.write_text("{corrupt", encoding="utf-8")

    cache = GeometryCache(persistent=True, cache_dir=str(cache_dir))
    assert cache.get("bad_npz") is None
    assert not bad_npz.exists()
    assert cache.get("bad_json") is None
    assert not bad_json.exists()


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


def test_process_memory_probe_is_safe():
    value = get_process_memory_mb()
    peak = get_process_memory_mb(peak=True)
    assert value is None or value > 0.0
    assert peak is None or peak > 0.0


def test_process_memory_linux_proc_status(monkeypatch):
    import io
    import ThermalSim.workflow as workflow_mod

    status_content = "Name:\tpython\nVmHWM:\t  20480 kB\nVmRSS:\t  10240 kB\n"
    monkeypatch.setattr(workflow_mod.os, "name", "posix")
    monkeypatch.setattr("builtins.open", lambda path, *a, **kw: io.StringIO(status_content))

    assert workflow_mod.get_process_memory_mb(peak=False) == 10.0
    assert workflow_mod.get_process_memory_mb(peak=True) == 20.0


def test_process_memory_rusage_fallback_darwin_and_posix(monkeypatch):
    import types
    import ThermalSim.workflow as workflow_mod

    monkeypatch.setattr(workflow_mod.os, "name", "posix")

    def fail_open(path, *args, **kwargs):
        raise OSError("no /proc")

    monkeypatch.setattr("builtins.open", fail_open)

    fake_resource = types.SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _who: types.SimpleNamespace(ru_maxrss=64 * 1024 * 1024),
    )
    monkeypatch.setitem(__import__("sys").modules, "resource", fake_resource)
    monkeypatch.setattr(workflow_mod.sys, "platform", "darwin")
    assert workflow_mod.get_process_memory_mb(peak=True) == 64.0

    fake_resource_linux = types.SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _who: types.SimpleNamespace(ru_maxrss=32 * 1024),
    )
    monkeypatch.setitem(__import__("sys").modules, "resource", fake_resource_linux)
    monkeypatch.setattr(workflow_mod.sys, "platform", "linux")
    assert workflow_mod.get_process_memory_mb(peak=True) == 32.0


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


def test_large_auto_plan_uses_single_uniform_phase():
    config = SolverConfig(1.0, 25.0, 1.0 / 120.0, 120, time_stepping="auto")
    mode, plan = _build_phase_plan(config, 600000)
    assert mode == "uniform"
    assert len(plan) == 1
    assert plan[0][2] == 95
    assert plan[0][0]["derivative_start"] is True


def test_small_auto_plan_keeps_legacy_phases():
    config = SolverConfig(1.0, 25.0, 1.0 / 120.0, 120, time_stepping="auto")
    mode, plan = _build_phase_plan(config, 10000)
    assert mode == "multi_phase"
    assert [item[0]["name"] for item in plan] == ["A", "B", "C"]
