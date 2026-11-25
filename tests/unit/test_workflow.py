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


def test_geometry_cache_key_tracks_and_normalizes_via_settings():
    snapshot = BoardSnapshot("board", "abc", (0, 0, 10, 10), (0, 2), 1, 1, 1)
    grid = GridEstimate(0.5, 0.5, 0, 0, 10, 10, 24, 24, 2, False, False, 200000, 100000)
    default_key = geometry_cache_key(snapshot, grid, {}, ["U1:1"])
    plated_alias_key = geometry_cache_key(
        snapshot, grid, {"via_plating_um": 25.0, "via_fill": "unfilled"}, ["U1:1"]
    )
    assert default_key == plated_alias_key

    copper_key = geometry_cache_key(
        snapshot, grid, {"via_plating_um": 25.0, "via_fill": "copper"}, ["U1:1"]
    )
    copper_alias_key = geometry_cache_key(
        snapshot, grid, {"via_plating_um": 25.0, "via_fill": "copper-filled"}, ["U1:1"]
    )
    thick_key = geometry_cache_key(
        snapshot, grid, {"via_plating_um": 35.0, "via_fill": "plated"}, ["U1:1"]
    )
    assert copper_key == copper_alias_key
    assert copper_key != default_key
    assert thick_key != default_key


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
    assert loaded_no_area.via_map_3d is None

    # Optional via_map_3d round-trip when present, absent in archive when None
    with np.load(cache_dir / "geom_no_area.npz", allow_pickle=False) as archive:
        assert "via_map_3d" not in archive

    via_3d = np.ones((3, 2, 2), dtype=np.float32)
    via_3d[0, 0, 1] = 85.0
    state_with_3d = GeometryState(
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
        via_map_3d=via_3d,
    )
    first.put("geom_with_3d", state_with_3d)
    loaded_3d = second.get("geom_with_3d")
    assert isinstance(loaded_3d, GeometryState)
    assert loaded_3d.via_map_3d is not None
    np.testing.assert_array_equal(loaded_3d.via_map_3d, via_3d)


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


def test_simulation_settings_default_budget_and_cache_key_area_mode():
    from ThermalSim.workflow import SimulationSettings

    cfg = SimulationSettings.from_mapping({})
    assert cfg.grid_node_budget == 3_000_000

    snapshot = BoardSnapshot("board", "abc", (0, 0, 10, 10), (0, 2), 1, 1, 1)
    grid = GridEstimate(0.5, 0.5, 0, 0, 10, 10, 24, 24, 2, False, False, 200000, 100000)
    k_full = geometry_cache_key(snapshot, grid, {"area_mode": "full"}, ["U1:1"])
    k_active = geometry_cache_key(snapshot, grid, {"area_mode": "active"}, ["U1:1"])
    k_legacy = geometry_cache_key(snapshot, grid, {}, ["U1:1"])
    assert k_full != k_active
    assert k_full != k_legacy


def test_pth_pad_splits_heat_evenly_across_spanned_layers():
    import pytest
    from ThermalSim.simulation_core import build_sparse_pad_contributions
    from tests.mocks.pcbnew_mock import In1_Cu, In2_Cu, PAD_ATTRIB_PTH, PAD_ATTRIB_SMD

    copper_ids = [F_Cu, In1_Cu, In2_Cu, B_Cu]
    pth_pad = MockPad(
        position=VECTOR2I(2_500_000, 2_500_000),
        layer=F_Cu,
        attribute=PAD_ATTRIB_PTH,
        bbox=EDA_RECT(2_000_000, 2_000_000, 1_000_000, 1_000_000),
    )
    smd_pad = MockPad(
        position=VECTOR2I(5_500_000, 2_500_000),
        layer=B_Cu,
        attribute=PAD_ATTRIB_SMD,
        bbox=EDA_RECT(5_000_000, 2_000_000, 1_000_000, 1_000_000),
    )
    board = MockBoard(
        footprints=[MockFootprint(pads=[pth_pad, smd_pad])],
        layer_names={F_Cu: "F.Cu", In1_Cu: "In1.Cu", In2_Cu: "In2.Cu", B_Cu: "B.Cu"},
    )
    rows, cols, res = 8, 8, 1.0
    contribs = build_sparse_pad_contributions(
        board, copper_ids, [pth_pad, smd_pad], rows, cols, 0.0, 0.0, res
    )

    pth_c, smd_c = contribs
    assert np.sum(pth_c.weights) == pytest.approx(1.0)
    layer_size = rows * cols
    for layer_idx in range(4):
        in_layer = (pth_c.indices >= layer_idx * layer_size) & (
            pth_c.indices < (layer_idx + 1) * layer_size
        )
        assert np.any(in_layer)
        assert np.sum(pth_c.weights[in_layer]) == pytest.approx(0.25)

    # SMD pad remains strictly on B.Cu (layer_idx == 3)
    assert np.all(smd_c.indices >= 3 * layer_size)
    assert np.sum(smd_c.weights) == pytest.approx(1.0)


def test_build_power_vector_non_aliasing_breakpoints_and_step_average():
    import pytest
    from ThermalSim.simulation_core import SparsePadContribution, build_power_vector

    c_const = SparsePadContribution(indices=np.array([0], dtype=np.int64), weights=np.array([1.0]))
    c_pwl_a = SparsePadContribution(indices=np.array([1], dtype=np.int64), weights=np.array([1.0]))
    c_pwl_b = SparsePadContribution(indices=np.array([2], dtype=np.int64), weights=np.array([1.0]))

    # Narrow 20 W pulse from 50.00 to 50.15 s (total energy = 0.15 * 20 = 3.0 J)
    t_a = np.array([0.0, 50.0, 50.001, 50.149, 50.15, 100.0])
    p_a = np.array([0.0, 0.0, 20.0, 20.0, 0.0, 0.0])
    t_b = np.array([0.0, 10.0, 20.0])
    p_b = np.array([0.0, 4.0, 4.0])

    q0, q_func = build_power_vector(
        [("const", 1.5), ("pwl", (t_a, p_a)), ("pwl", (t_b, p_b))],
        [c_const, c_pwl_a, c_pwl_b],
        total_nodes=4,
    )
    assert q0[0] == pytest.approx(1.5)
    assert q_func is not None

    # Non-aliasing: stored evaluations must remain independent
    q_t5 = q_func(5.0)
    q_t50 = q_func(50.05)
    assert q_t5[1] == pytest.approx(0.0)
    assert q_t5[2] == pytest.approx(2.0)
    assert q_t50[1] == pytest.approx(20.0)
    assert q_t50[2] == pytest.approx(4.0)

    # Breakpoints sorted unique finite tuple
    assert q_func.breakpoints == (0.0, 10.0, 20.0, 50.0, 50.001, 50.149, 50.15, 100.0)

    # Analytical step_average preserves exact pulse energy across coarse step [50.0, 50.3333]
    dt_step = 50.3333 - 50.0
    expected_energy_a = float(np.trapezoid(p_a, t_a)) if hasattr(np, "trapezoid") else float(np.trapz(p_a, t_a))
    q_avg = q_func.step_average(50.0, 50.3333)
    assert q_avg[0] == pytest.approx(1.5)
    assert q_avg[1] * dt_step == pytest.approx(expected_energy_a, rel=1e-12)
    assert q_avg[2] == pytest.approx(4.0)

    # Degenerate t1 <= t0 matches Q_func(t1)
    np.testing.assert_allclose(q_func.step_average(5.0, 5.0), q_func(5.0))
    np.testing.assert_allclose(q_func.step_average(10.0, 5.0), q_func(5.0))


def test_effective_fr4_thickness_allows_thick_boards_up_to_50mm():
    import pytest
    from ThermalSim.simulation_core import effective_fr4_control_volume_thicknesses

    # 20 mm 2-layer board -> gap = 20 mm -> CV thickness = 10 mm each (not clipped at 5 mm)
    th = effective_fr4_control_volume_thicknesses([20e-3], 20e-3, 2)
    assert th[0] == pytest.approx(10e-3)
    assert th[1] == pytest.approx(10e-3)
