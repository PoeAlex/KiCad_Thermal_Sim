"""
Unit tests for thermal_plugin helper functions.

These tests focus on initialization-time helper logic that was refactored
for performance, while keeping numerical behavior unchanged.
"""

import json

import numpy as np
import pytest

from ThermalSim.geometry_mapper import get_pad_pixels
from ThermalSim.thermal_plugin import (
    _build_power_vector,
    _build_sparse_pad_contributions,
    _coarsen_grid_resolution,
    _effective_fr4_control_volume_thicknesses,
    _find_pcb_editor_parent,
    _resolve_grid_policy,
)
from tests.mocks.pcbnew_mock import (
    MockBoard,
    MockDrawing,
    MockFootprint,
    MockPad,
    MockTrack,
    MockZone,
    PAD_ATTRIB_PTH,
    VECTOR2I,
    EDA_RECT,
    Eco1_User,
    F_Cu,
    B_Cu,
)


class TestStartupSafety:
    """Regression tests for KiCad startup and zone-map safety."""

    def test_unfilled_zones_are_detected_without_running_zone_filler(self):
        from ThermalSim.thermal_plugin import ThermalPlugin

        board = MockBoard(zones=[
            MockZone(layers=[F_Cu], filled=False),
            MockZone(layers=[F_Cu], filled=True),
        ])
        plugin = ThermalPlugin()
        plugin.defaults()

        assert plugin._count_unfilled_copper_zones(board) == 1
        assert plugin._require_filled_zones(board) is False

    def test_cache_snapshot_never_reads_filled_polygon_map(self):
        from ThermalSim.thermal_plugin import ThermalPlugin

        class UnsafePolygonZone(MockZone):
            def GetFilledPolysList(self, _layer):
                raise AssertionError("unsafe filled-polygon map access")

        board = MockBoard(
            zones=[UnsafePolygonZone(layers=[F_Cu], filled=True)],
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
        )
        plugin = ThermalPlugin()
        plugin.defaults()

        snapshot = plugin._capture_board_snapshot(
            board,
            [F_Cu, B_Cu],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
        )

        assert snapshot.zone_count == 1

    def test_selected_descriptors_reuse_existing_selection_scan(self):
        from ThermalSim.thermal_plugin import ThermalPlugin

        pad = MockPad(
            position=VECTOR2I(1_000_000, 2_000_000),
            layer=F_Cu,
            number="7",
        )
        board = MockBoard(layer_names={F_Cu: "F.Cu"})
        plugin = ThermalPlugin()
        plugin.defaults()

        descriptors = plugin._descriptors_from_selected_pads(
            board, [("U42-7", pad)]
        )

        assert descriptors[0]["name"].startswith("U42-7")

    def test_dialog_parent_prefers_pcb_editor_over_project_manager(self, monkeypatch):
        """Plugin dialogs should remain owned by the PCB Editor frame."""
        import ThermalSim.thermal_plugin as thermal_plugin_module

        wx_api = thermal_plugin_module.wx

        class Window:
            def __init__(self, title, parent=None):
                self.title = title
                self.parent = parent

            def GetTitle(self):
                return self.title

            def GetParent(self):
                return self.parent

        manager = Window("KiCad Project Manager")
        editor = Window("power_board.kicad_pcb - PCB Editor")
        toolbar_child = Window("Plugin toolbar", parent=editor)
        board = MockBoard(filename=r"C:\boards\power_board.kicad_pcb")

        monkeypatch.setattr(wx_api, "GetActiveWindow", lambda: toolbar_child, raising=False)
        monkeypatch.setattr(
            wx_api, "GetTopLevelWindows", lambda: [manager, editor], raising=False
        )

        assert _find_pcb_editor_parent(board) is editor

    def test_dialog_parent_does_not_choose_active_project_manager(self, monkeypatch):
        """An identifiable editor wins even if the manager is currently active."""
        import ThermalSim.thermal_plugin as thermal_plugin_module

        wx_api = thermal_plugin_module.wx

        class Window:
            def __init__(self, title):
                self.title = title

            def GetTitle(self):
                return self.title

            def GetParent(self):
                return None

        manager = Window("KiCad Project Manager")
        editor = Window("layout - PCB Editor")
        monkeypatch.setattr(wx_api, "GetActiveWindow", lambda: manager, raising=False)
        monkeypatch.setattr(
            wx_api, "GetTopLevelWindows", lambda: [manager, editor], raising=False
        )

        assert _find_pcb_editor_parent() is editor


class TestFr4ControlVolumes:
    """Regression tests for conserved dielectric control-volume thickness."""

    def test_two_layer_gap_is_split_between_outer_planes(self):
        """A two-layer board must contain one gap, not one gap per plane."""
        thicknesses = _effective_fr4_control_volume_thicknesses(
            [1.53e-3], 1.6e-3, 2
        )

        np.testing.assert_allclose(thicknesses, [0.765e-3, 0.765e-3])
        np.testing.assert_allclose(np.sum(thicknesses), 1.53e-3)

    def test_multilayer_control_volumes_conserve_all_gaps(self):
        """Outer half-gaps and inner half-pairs must conserve FR4 volume."""
        gaps = np.asarray([0.2e-3, 1.0e-3, 0.2e-3])
        thicknesses = _effective_fr4_control_volume_thicknesses(
            gaps, 1.6e-3, 4
        )

        np.testing.assert_allclose(
            thicknesses,
            [0.1e-3, 0.6e-3, 0.6e-3, 0.1e-3],
        )
        np.testing.assert_allclose(np.sum(thicknesses), np.sum(gaps))

    def test_single_layer_uses_full_board_thickness(self):
        """A single copper plane owns the full fallback board thickness."""
        thicknesses = _effective_fr4_control_volume_thicknesses(
            [], 1.6e-3, 1
        )

        np.testing.assert_allclose(thicknesses, [1.6e-3])


def _legacy_power_vector(board, copper_ids, pads_list, pad_sources, rows, cols, x_min, y_min, res):
    """Reference implementation matching the legacy dense-Q setup."""
    rc = rows * cols
    layer_count = len(copper_ids)
    total_nodes = rc * layer_count

    q_units = []
    for pad in pads_list:
        q_pad = np.zeros(total_nodes, dtype=np.float64)
        pad_lid = pad.GetLayer()
        if pad_lid in copper_ids:
            target_idx = copper_ids.index(pad_lid)
        else:
            lname = board.GetLayerName(pad_lid).upper()
            target_idx = layer_count - 1 if ("B." in lname or "BOT" in lname) else 0

        pixels = get_pad_pixels(pad, rows, cols, x_min, y_min, res)
        if pixels:
            pix = np.array(pixels, dtype=np.int64)
            r, c = pix[:, 0], pix[:, 1]
            valid = (r < rows) & (c < cols) & (r >= 0) & (c >= 0)
            r, c = r[valid], c[valid]
            if r.size > 0:
                idxs = target_idx * rc + r * cols + c
                np.add.at(q_pad, idxs, 1.0 / float(r.size))
        q_units.append(q_pad)

    q = np.zeros(total_nodes, dtype=np.float64)
    has_pwl = any(source_type == 'pwl' for source_type, _ in pad_sources)
    for idx, (source_type, source_value) in enumerate(pad_sources):
        if idx >= len(q_units):
            break
        if source_type == 'const':
            q += float(source_value) * q_units[idx]
        else:
            q += float(np.interp(0.0, source_value[0], source_value[1])) * q_units[idx]

    if not has_pwl:
        return q, None

    def q_func(t, _sources=pad_sources, _units=q_units, _n=total_nodes):
        q_t = np.zeros(_n, dtype=np.float64)
        for idx, (source_type, source_value) in enumerate(_sources):
            if idx >= len(_units):
                break
            if source_type == 'const':
                q_t += float(source_value) * _units[idx]
            else:
                q_t += float(np.interp(t, source_value[0], source_value[1])) * _units[idx]
        return q_t

    return q, q_func


class TestPowerVectorHelpers:
    """Tests for sparse pad power helper functions."""

    def _board_and_pads(self):
        board = MockBoard(layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"})
        top_pad = MockPad(
            position=VECTOR2I(5000000, 5000000),
            layer=F_Cu,
            bbox=EDA_RECT(4500000, 4500000, 1000000, 1000000),
        )
        bottom_pad = MockPad(
            position=VECTOR2I(7000000, 5000000),
            layer=B_Cu,
            bbox=EDA_RECT(6500000, 4500000, 1000000, 1000000),
        )
        return board, [top_pad, bottom_pad]

    def test_sparse_power_matches_legacy_constant_case(self):
        """Constant pad powers should match the legacy dense-Q construction."""
        board, pads_list = self._board_and_pads()
        copper_ids = [F_Cu, B_Cu]
        rows = cols = 20
        x_min = y_min = 0.0
        res = 0.5
        total_nodes = rows * cols * len(copper_ids)
        pad_sources = [('const', 1.25), ('const', 0.75)]

        contributions = _build_sparse_pad_contributions(
            board=board,
            copper_ids=copper_ids,
            pads_list=pads_list,
            rows=rows,
            cols=cols,
            x_min=x_min,
            y_min=y_min,
            res=res,
        )
        q_new, q_func_new = _build_power_vector(pad_sources, contributions, total_nodes)
        q_ref, q_func_ref = _legacy_power_vector(
            board, copper_ids, pads_list, pad_sources, rows, cols, x_min, y_min, res
        )

        assert q_func_new is None
        assert q_func_ref is None
        np.testing.assert_allclose(q_new, q_ref, atol=1e-12)

    def test_sparse_power_matches_legacy_pwl_case(self):
        """PWL pad powers should match the legacy dense-Q evaluation at runtime."""
        board, pads_list = self._board_and_pads()
        copper_ids = [F_Cu, B_Cu]
        rows = cols = 20
        x_min = y_min = 0.0
        res = 0.5
        total_nodes = rows * cols * len(copper_ids)
        pad_sources = [
            ('const', 1.0),
            ('pwl', (np.array([0.0, 2.0, 4.0]), np.array([0.0, 2.0, 1.0]))),
        ]

        contributions = _build_sparse_pad_contributions(
            board=board,
            copper_ids=copper_ids,
            pads_list=pads_list,
            rows=rows,
            cols=cols,
            x_min=x_min,
            y_min=y_min,
            res=res,
        )
        q_new, q_func_new = _build_power_vector(pad_sources, contributions, total_nodes)
        q_ref, q_func_ref = _legacy_power_vector(
            board, copper_ids, pads_list, pad_sources, rows, cols, x_min, y_min, res
        )

        assert q_func_new is not None
        assert q_func_ref is not None
        np.testing.assert_allclose(q_new, q_ref, atol=1e-12)
        np.testing.assert_allclose(q_func_new(1.0), q_func_ref(1.0), atol=1e-12)
        np.testing.assert_allclose(q_func_new(3.0), q_func_ref(3.0), atol=1e-12)

    def test_power_and_current_pad_resolution_are_separate(self):
        """Manual power pads and current terminals should resolve independently."""
        from ThermalSim.thermal_plugin import ThermalPlugin

        power_pad = MockPad(
            position=VECTOR2I(5000000, 5000000),
            layer=F_Cu,
            net_code=1,
            net_name="VIN",
            number="1",
        )
        current_pad = MockPad(
            position=VECTOR2I(7000000, 5000000),
            layer=F_Cu,
            net_code=1,
            net_name="VIN",
            number="2",
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("J1", [power_pad, current_pad])],
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        power_descriptor = plugin._pad_descriptor(board, "J1", power_pad)
        current_descriptor = plugin._pad_descriptor(board, "J1", current_pad)

        settings = {
            "power_pads": [{**power_descriptor, "power": "2.0"}],
            "current_enabled": True,
            "current_groups": [{
                "name": "Load",
                "mode": "per_pad",
                "pads": [{**current_descriptor, "current_a": 5.0}],
            }],
        }

        assert plugin._resolve_power_pad_objects(board, settings) == [power_pad]
        assert plugin._resolve_current_pad_objects(board, settings) == [current_pad]

    def test_preflight_scans_board_footprints_only_once(self, tmp_path):
        """Preflight should reuse a single pad lookup across power and current resolution."""
        from ThermalSim.thermal_plugin import ThermalPlugin

        power_pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            net_code=1,
            net_name="VIN",
            number="1",
        )
        sink_pad = MockPad(
            position=VECTOR2I(7_000_000, 5_000_000),
            layer=F_Cu,
            net_code=1,
            net_name="VIN",
            number="2",
        )

        class CountingBoard(MockBoard):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.footprint_scans = 0

            def Footprints(self):
                self.footprint_scans += 1
                return super().Footprints()

            def GetFootprints(self):
                self.footprint_scans += 1
                return super().GetFootprints()

        board = CountingBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("J1", [power_pad, sink_pad])],
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        d1 = plugin._pad_descriptor(board, "J1", power_pad)
        d2 = plugin._pad_descriptor(board, "J1", sink_pad)
        settings = {
            "res": 0.5,
            "output_dir": str(tmp_path),
            "power_pads": [{**d1, "power": "1.5"}],
            "current_enabled": True,
            "current_groups": [{
                "name": "Rail",
                "mode": "per_pad",
                "pads": [
                    {**d1, "current_a": 2.0},
                    {**d2, "current_a": -2.0},
                ],
            }],
        }
        board.footprint_scans = 0
        result = plugin._preflight(
            board,
            [F_Cu, B_Cu],
            EDA_RECT(0, 0, 20_000_000, 20_000_000),
            settings,
        )

        assert result.ready is True
        assert board.footprint_scans == 1


class TestBoardSnapshotCompleteness:
    """Tests for _capture_board_snapshot Eco1 drawings and KiCad 10 via safety."""

    def test_snapshot_includes_eco1_drawings_and_invalidates_on_move(self):
        from ThermalSim.thermal_plugin import ThermalPlugin

        plugin = ThermalPlugin()
        plugin.defaults()
        bbox = EDA_RECT(0, 0, 20_000_000, 20_000_000)

        board_a = MockBoard(
            drawings=[MockDrawing(layer=Eco1_User, bbox=EDA_RECT(1_000_000, 1_000_000, 5_000_000, 5_000_000))]
        )
        board_b = MockBoard(
            drawings=[MockDrawing(layer=Eco1_User, bbox=EDA_RECT(4_000_000, 2_000_000, 5_000_000, 5_000_000))]
        )
        board_other_layer = MockBoard(
            drawings=[
                MockDrawing(layer=Eco1_User, bbox=EDA_RECT(1_000_000, 1_000_000, 5_000_000, 5_000_000)),
                MockDrawing(layer=F_Cu, bbox=EDA_RECT(8_000_000, 8_000_000, 2_000_000, 2_000_000)),
            ]
        )

        snap_a = plugin._capture_board_snapshot(board_a, [F_Cu, B_Cu], bbox)
        snap_b = plugin._capture_board_snapshot(board_b, [F_Cu, B_Cu], bbox)
        snap_other = plugin._capture_board_snapshot(board_other_layer, [F_Cu, B_Cu], bbox)

        assert snap_a.fingerprint != snap_b.fingerprint
        assert snap_a.fingerprint == snap_other.fingerprint

    def test_snapshot_via_uses_layer_arg_and_avoids_zero_arg_get_width(self):
        from ThermalSim.thermal_plugin import ThermalPlugin

        class KiCad10PCB_VIA(MockTrack):
            def __init__(self):
                super().__init__(layer=F_Cu, bbox=EDA_RECT(2_000_000, 2_000_000, 600_000, 600_000))
                self.called_layers = []

            def GetWidth(self, layer=None):
                if layer is None:
                    raise AssertionError("PCB_VIA.GetWidth() called without layer argument")
                self.called_layers.append(layer)
                return 600_000

        via = KiCad10PCB_VIA()
        board = MockBoard(tracks=[via])
        plugin = ThermalPlugin()
        plugin.defaults()

        snap = plugin._capture_board_snapshot(
            board, [F_Cu, B_Cu], EDA_RECT(0, 0, 10_000_000, 10_000_000)
        )

        assert snap.track_count == 1
        assert via.called_layers == [F_Cu]

        # Also safe when copper_ids is empty (never calls zero-arg GetWidth)
        via_empty = KiCad10PCB_VIA()
        snap_empty = plugin._capture_board_snapshot(
            MockBoard(tracks=[via_empty]), [], EDA_RECT(0, 0, 10_000_000, 10_000_000)
        )
        assert snap_empty.track_count == 1
        assert via_empty.called_layers == []


class TestSettingsPersistence:
    """Tests for JSON settings file persistence helpers."""

    def test_save_and_load_settings_from_custom_path(self, tmp_path):
        """Settings should round-trip through a caller-selected JSON file."""
        from ThermalSim.thermal_plugin import ThermalPlugin

        plugin = ThermalPlugin()
        plugin.defaults()
        settings_path = tmp_path / "my_thermal_settings.json"
        settings = {
            "power_str": "2.5",
            "time": 60.0,
            "amb": 25.0,
            "res": 0.2,
            "power_pads": [{
                "pad_key": "U1:1:1:100:200",
                "name": "U1-1 [VIN]",
                "net_name": "VIN",
                "net_code": 1,
                "layer": "F.Cu",
                "power": "2.5",
            }],
            "current_enabled": True,
            "current_groups": [{
                "name": "Load",
                "color": "#d62728",
                "mode": "per_pad",
                "total_current_a": 0.0,
                "pads": [{
                    "pad_key": "J1:1:1:300:400",
                    "name": "J1-1 [VIN]",
                    "net_name": "VIN",
                    "net_code": 1,
                    "layer": "F.Cu",
                    "current_a": 5.0,
                }],
            }],
        }

        assert plugin._save_settings(settings, str(settings_path)) is True
        assert plugin._load_settings(str(settings_path)) == {**settings, "schema_version": 2}

    def test_load_settings_rejects_non_dict_json(self, tmp_path):
        """Only object-shaped JSON files are valid settings files."""
        from ThermalSim.thermal_plugin import ThermalPlugin

        plugin = ThermalPlugin()
        plugin.defaults()
        settings_path = tmp_path / "not_settings.json"
        settings_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

        assert plugin._load_settings(str(settings_path)) == {}

    def test_load_settings_rejects_invalid_json(self, tmp_path):
        """Malformed JSON should not raise during settings load."""
        from ThermalSim.thermal_plugin import ThermalPlugin

        plugin = ThermalPlugin()
        plugin.defaults()
        settings_path = tmp_path / "broken_settings.json"
        settings_path.write_text("{broken json", encoding="utf-8")

        assert plugin._load_settings(str(settings_path)) == {}


class TestGridCoarsening:
    """Tests for automatic grid coarsening limits."""

    def test_custom_node_budget_supports_one_hundred_million_nodes(self):
        """Custom mode should preserve the documented 100 million-node limit."""
        detail, expert, max_cells, target_cells, max_nodes = _resolve_grid_policy(
            {
                "grid_detail_level": "custom",
                "grid_node_budget": 100_000_000,
            },
            layer_count=4,
        )

        assert detail == "custom"
        assert expert is True
        assert max_nodes == 100_000_000
        assert max_cells == 25_000_000
        assert target_cells == 21_250_000

    def test_custom_node_budget_is_clamped_at_one_hundred_million(self):
        """Imported settings must not bypass the 100 million-node safety cap."""
        _, _, max_cells, target_cells, max_nodes = _resolve_grid_policy(
            {
                "grid_detail_level": "custom",
                "grid_node_budget": 250_000_000,
            },
            layer_count=2,
        )

        assert max_nodes == 100_000_000
        assert max_cells == 50_000_000
        assert target_cells == 42_500_000

    def test_named_presets_resolve_to_updated_budgets_and_85pct_targets(self):
        """Fast/Balanced/Detailed/Very detailed presets use 1M/3M/8M/16M with 85% target."""
        expected = {
            "fast": (1_000_000, 850_000),
            "balanced": (3_000_000, 2_550_000),
            "detailed": (8_000_000, 6_800_000),
            "very_detailed": (16_000_000, 13_600_000),
        }
        for name, (exp_max, exp_target) in expected.items():
            detail, expert, max_cells, target_cells, max_nodes = _resolve_grid_policy(
                {"grid_detail_level": name}, layer_count=4
            )
            assert detail == name
            assert expert is False
            assert max_nodes == exp_max
            assert max_cells == exp_max // 4
            assert target_cells == exp_target // 4

    def test_auto_fast_engine_threshold_and_mesh_mode_resolution(self, monkeypatch):
        """Auto engine threshold follows HAS_NATIVE_MG and mesh mode is always uniform."""
        import ThermalSim.thermal_plugin as plugin_mod

        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_MG", True)
        assert plugin_mod._auto_fast_engine_threshold() == 50_000
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "auto"}, True, 1_000_000) == "uniform"
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "auto"}, True, 25_000_000) == "uniform"
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "adaptive"}, True, 100_000) == "uniform"
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "uniform"}, True, 10_000_000) == "uniform"

        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_MG", False)
        assert plugin_mod._auto_fast_engine_threshold() == 250_000
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "auto"}, True, 1_000_000) == "uniform"

    def test_estimate_solver_cost_reflects_native_mg_and_ignores_legacy_backend(self, monkeypatch):
        """Native compact/MG cost models use calibrated v4 RAM/runtime bounds and ignore legacy backend."""
        import ThermalSim.thermal_plugin as plugin_mod

        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_MG", True)
        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_COMPACT", True)

        # v4 Compact uniform calibrated against 6M (~0.5-0.75 GB) & 25.6M (~2.0 GB) measurements
        assert plugin_mod._estimate_solver_cost(
            3_000_000, {"compute_engine": "auto", "mesh_mode": "auto"}
        ) == (240, 375, "Moderate")
        assert plugin_mod._estimate_solver_cost(
            6_000_000, {"compute_engine": "auto", "mesh_mode": "uniform"}
        ) == (480, 750, "Moderate")
        assert plugin_mod._estimate_solver_cost(
            25_600_000, {"compute_engine": "auto", "mesh_mode": "auto"}
        ) == (2048, 3200, "Very slow")

        # v4 runtime thresholds (2M, 8M, 20M)
        assert plugin_mod._estimate_solver_cost(1_000_000, {"compute_engine": "auto"})[2] == "Fast"
        assert plugin_mod._estimate_solver_cost(1_999_999, {"compute_engine": "auto"})[2] == "Fast"
        assert plugin_mod._estimate_solver_cost(2_000_000, {"compute_engine": "auto"})[2] == "Moderate"
        assert plugin_mod._estimate_solver_cost(6_000_000, {"compute_engine": "auto"})[2] == "Moderate"
        assert plugin_mod._estimate_solver_cost(10_000_000, {"compute_engine": "auto"})[2] == "Slow"
        assert plugin_mod._estimate_solver_cost(20_500_000, {"compute_engine": "auto"})[2] == "Very slow"

        # v2-only native MG (HAS_NATIVE_COMPACT=False)
        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_COMPACT", False)
        low, high, runtime = plugin_mod._estimate_solver_cost(
            3_000_000, {"compute_engine": "auto"}
        )
        assert low == 390
        assert high == 660
        assert runtime == "Slow"

        _, _, rt_fast = plugin_mod._estimate_solver_cost(500_000, {"compute_engine": "auto"})
        _, _, rt_mod = plugin_mod._estimate_solver_cost(1_200_000, {"compute_engine": "auto"})
        _, _, rt_vslow = plugin_mod._estimate_solver_cost(9_000_000, {"compute_engine": "auto"})
        assert (rt_fast, rt_mod, rt_vslow) == ("Fast", "Moderate", "Very slow")

        # Below 50k threshold uses legacy direct estimates; legacy backend key ignored
        legacy_val = "par" + "diso"
        c_plain = plugin_mod._estimate_solver_cost(40_000, {"compute_engine": "auto"})
        c_legacy = plugin_mod._estimate_solver_cost(
            40_000, {"compute_engine": "auto", "solver_backend": legacy_val}
        )
        assert c_plain == (14, 48, "Fast")
        assert c_plain == c_legacy

        # NumPy/SciPy matrix-free without native MG
        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_MG", False)
        low_no_mg, high_no_mg, rt_no_mg = plugin_mod._estimate_solver_cost(
            1_000_000, {"compute_engine": "fast_native"}
        )
        assert low_no_mg == 110
        assert high_no_mg == 190
        assert rt_no_mg == "Moderate"

    def test_preflight_ignores_legacy_backend_setting(self, tmp_path):
        """Old JSON settings containing deprecated solver_backend must not fail preflight."""
        from ThermalSim.thermal_plugin import ThermalPlugin

        pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            net_code=1,
            net_name="VIN",
            number="1",
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("U1", [pad])],
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        desc = plugin._pad_descriptor(board, "U1", pad)
        legacy_val = "par" + "diso"
        result = plugin._preflight(
            board,
            [F_Cu, B_Cu],
            EDA_RECT(0, 0, 20_000_000, 20_000_000),
            {
                "res": 0.5,
                "output_dir": str(tmp_path),
                "solver_backend": legacy_val,
                "power_pads": [{**desc, "power": "1.5"}],
            },
        )
        assert result.ready is True
        assert result.errors == []

    def test_generate_preview_always_renders_uniform_without_adaptive_overlay(self, tmp_path, monkeypatch):
        """Preview renders the uniform geometry cleanly even when legacy mesh_mode='adaptive' is passed."""
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_plugin import ThermalPlugin

        pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            bbox=EDA_RECT(4_500_000, 4_500_000, 1_000_000, 1_000_000),
            net_code=1,
            net_name="VIN",
            number="1",
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("U1", [pad])],
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        plugin.board = board
        plugin.copper_ids = [F_Cu, B_Cu]
        plugin.bbox = EDA_RECT(0, 0, 10_000_000, 10_000_000)
        plugin.pads_list = [pad]
        plugin.stack_info = {}

        preview_meshes = []

        def fake_save_preview_image(*args, **kwargs):
            preview_meshes.append(kwargs.get("adaptive_mesh"))
            return str(tmp_path / "preview.png")

        monkeypatch.setattr(plugin_mod, "save_preview_image", fake_save_preview_image)
        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_MG", True)

        for mode in ("auto", "adaptive"):
            plugin.generate_preview(
                {
                    "res": 2.0,
                    "compute_engine": "fast_native",
                    "mesh_mode": mode,
                    "output_dir": str(tmp_path),
                },
                ["F.Cu", "B.Cu"],
            )
        assert preview_meshes == [None, None]

    def test_default_coarsening_matches_legacy_formula(self):
        """Without expert settings, the historic 200k/100k limits apply."""
        res, auto_coarsened, expert, max_cells, target_cells = _coarsen_grid_resolution(
            w_mm=300.0,
            h_mm=75.0,
            requested_res=0.1,
            settings={},
        )

        assert auto_coarsened is True
        assert expert is False
        assert max_cells == 200000
        assert target_cells == 100000
        np.testing.assert_allclose(res, np.sqrt((300.0 * 75.0) / 100000.0))

    def test_expert_limits_change_coarsened_resolution(self):
        """Expert target cells should control the resulting coarsened resolution."""
        res, auto_coarsened, expert, max_cells, target_cells = _coarsen_grid_resolution(
            w_mm=300.0,
            h_mm=75.0,
            requested_res=0.1,
            settings={
                "grid_expert_limits": True,
                "grid_max_cells": 1000000,
                "grid_target_cells": 500000,
            },
        )

        assert auto_coarsened is True
        assert expert is True
        assert max_cells == 1000000
        assert target_cells == 500000
        np.testing.assert_allclose(res, np.sqrt((300.0 * 75.0) / 500000.0))

    def test_expert_limits_can_avoid_coarsening(self):
        """Raising max cells above the estimate should preserve requested resolution."""
        res, auto_coarsened, expert, max_cells, target_cells = _coarsen_grid_resolution(
            w_mm=300.0,
            h_mm=75.0,
            requested_res=0.1,
            settings={
                "grid_expert_limits": True,
                "grid_max_cells": 3000000,
                "grid_target_cells": 1000000,
            },
        )

        assert res == 0.1
        assert auto_coarsened is False
        assert expert is True
        assert max_cells == 3000000
        assert target_cells == 1000000


class TestSteadyStatePluginWiring:
    """Tests for steady-state preflight warning and solver config wiring."""

    def test_preflight_warns_when_steady_state_used_with_pwl(self, tmp_path):
        from ThermalSim.thermal_plugin import ThermalPlugin

        pwl_file = tmp_path / "pulse.pwl"
        pwl_file.write_text("0.0 0.0\n1.0 2.5\n2.0 2.5\n", encoding="utf-8")

        pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            net_code=1,
            net_name="VIN",
            number="1",
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("U1", [pad])],
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        desc = plugin._pad_descriptor(board, "U1", pad)

        settings = {
            "res": 0.5,
            "output_dir": str(tmp_path),
            "steady_state": True,
            "power_pads": [{**desc, "power": str(pwl_file)}],
        }
        result = plugin._preflight(
            board,
            [F_Cu, B_Cu],
            EDA_RECT(0, 0, 20_000_000, 20_000_000),
            settings,
        )

        expected_warning = (
            "Steady-state mode ignores PWL time variation; "
            "disable Steady-state for time-varying PWL profiles."
        )
        assert result.ready is True
        assert expected_warning in result.warnings

        # Constant power should not produce the PWL steady-state warning
        settings_const = {
            "res": 0.5,
            "output_dir": str(tmp_path),
            "steady_state": True,
            "power_pads": [{**desc, "power": "2.0"}],
        }
        result_const = plugin._preflight(
            board,
            [F_Cu, B_Cu],
            EDA_RECT(0, 0, 20_000_000, 20_000_000),
            settings_const,
        )
        assert result_const.ready is True
        assert expected_warning not in result_const.warnings

    def test_run_simulation_wires_steady_state_config_and_progress_message(
        self, tmp_path, monkeypatch
    ):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_plugin import ThermalPlugin
        from ThermalSim.thermal_solver import SolverResult

        pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            bbox=EDA_RECT(4_500_000, 4_500_000, 1_000_000, 1_000_000),
            net_code=1,
            net_name="VIN",
            number="1",
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("U1", [pad])],
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        desc = plugin._pad_descriptor(board, "U1", pad)

        captured_configs = []
        progress_messages = []

        class RecordingConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                captured_configs.append(kwargs)

        class RecordingProgressDialog:
            def __init__(self, title, message, maximum=100, **kwargs):
                progress_messages.append(message)

            def Update(self, value, newmsg=""):
                if newmsg:
                    progress_messages.append(newmsg)
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                pass

        def fake_run_simulation(
            config, K, C, Q, b, hA, layer_count, rows, cols, *args, **kwargs
        ):
            return SolverResult(
                T=np.full((layer_count, rows, cols), 40.0),
                k_norm_info={},
                step_counter=1,
                total_solve_time=0.01,
                total_factor_time=0.01,
                factor_count=1,
            )

        monkeypatch.setattr(plugin_mod, "SolverConfig", RecordingConfig)
        monkeypatch.setattr(plugin_mod, "run_simulation", fake_run_simulation)
        monkeypatch.setattr(
            plugin_mod.wx, "ProgressDialog", RecordingProgressDialog, raising=False
        )
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: None, raising=False)
        monkeypatch.setattr(
            plugin_mod, "show_results_all_layers", lambda *a, **k: str(tmp_path / "hm.png")
        )
        monkeypatch.setattr(
            plugin_mod, "save_preview_image", lambda *a, **k: str(tmp_path / "prev.png")
        )
        monkeypatch.setattr(
            plugin_mod, "build_interactive_heatmap_payload", lambda *a, **k: {}
        )
        import webbrowser
        monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
        monkeypatch.setattr(
            plugin_mod, "write_html_report", lambda *a, **k: str(tmp_path / "rep.html")
        )

        settings = {
            "res": 2.0,
            "time": 20.0,
            "amb": 25.0,
            "thick": 1.6,
            "h_conv": 10.0,
            "pad_th": 1.0,
            "pad_k": 3.0,
            "show_all": True,
            "snapshots": True,
            "snap_count": 3,
            "steady_state": True,
            "compute_engine": "legacy",
            "output_dir": str(tmp_path),
            "power_pads": [{**desc, "power": "2.0"}],
        }
        plugin._run_simulation(
            board,
            [F_Cu, B_Cu],
            ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad],
            settings,
            {},
            ["U1-1"],
        )

        assert len(captured_configs) == 1
        assert captured_configs[0]["steady_state"] is True
        assert captured_configs[0]["snapshots_enabled"] is False
        assert captured_configs[0]["snap_times"] == []
        assert "Solving steady-state thermal model..." in progress_messages

        # With PWL power source (Q_func is not None), steady_state must be False
        pwl_file = tmp_path / "ramp.pwl"
        pwl_file.write_text("0.0 0.0\n10.0 2.0\n", encoding="utf-8")
        captured_configs.clear()
        progress_messages.clear()
        settings_pwl = {
            **settings,
            "power_pads": [{**desc, "power": str(pwl_file)}],
        }
        plugin._run_simulation(
            board,
            [F_Cu, B_Cu],
            ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad],
            settings_pwl,
            {},
            ["U1-1"],
        )

        assert len(captured_configs) == 1
        assert captured_configs[0]["steady_state"] is False
        assert "Solving steady-state thermal model..." not in progress_messages

    def test_run_simulation_fallback_config_preserves_steady_state(
        self, tmp_path, monkeypatch
    ):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_plugin import ThermalPlugin
        from ThermalSim.thermal_solver import SolverResult

        pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            bbox=EDA_RECT(4_500_000, 4_500_000, 1_000_000, 1_000_000),
            net_code=1,
            net_name="VIN",
            number="1",
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("U1", [pad])],
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        desc = plugin._pad_descriptor(board, "U1", pad)

        captured_configs = []

        class RecordingConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                captured_configs.append(kwargs)

        def failing_fast_solver(*args, **kwargs):
            raise RuntimeError("simulated fast solver failure")

        def fake_legacy_solver(
            config, K, C, Q, b, hA, layer_count, rows, cols, *args, **kwargs
        ):
            return SolverResult(
                T=np.full((layer_count, rows, cols), 42.0),
                k_norm_info={},
                step_counter=1,
                total_solve_time=0.01,
                total_factor_time=0.01,
                factor_count=1,
            )

        class DummyProgressDialog:
            def __init__(self, *args, **kwargs):
                pass

            def Update(self, value, newmsg=""):
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                pass

        monkeypatch.setattr(plugin_mod, "SolverConfig", RecordingConfig)
        monkeypatch.setattr(plugin_mod, "run_simulation_matrix_free", failing_fast_solver)
        monkeypatch.setattr(plugin_mod, "run_simulation", fake_legacy_solver)
        monkeypatch.setattr(
            plugin_mod.wx, "ProgressDialog", DummyProgressDialog, raising=False
        )
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: None, raising=False)
        monkeypatch.setattr(
            plugin_mod, "show_results_all_layers", lambda *a, **k: str(tmp_path / "hm.png")
        )
        monkeypatch.setattr(
            plugin_mod, "save_preview_image", lambda *a, **k: str(tmp_path / "prev.png")
        )
        monkeypatch.setattr(
            plugin_mod, "build_interactive_heatmap_payload", lambda *a, **k: {}
        )
        import webbrowser
        monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
        monkeypatch.setattr(
            plugin_mod, "write_html_report", lambda *a, **k: str(tmp_path / "rep.html")
        )

        settings = {
            "res": 2.0,
            "time": 20.0,
            "amb": 25.0,
            "thick": 1.6,
            "h_conv": 10.0,
            "pad_th": 1.0,
            "pad_k": 3.0,
            "show_all": True,
            "snapshots": False,
            "steady_state": True,
            "compute_engine": "fast_native",
            "output_dir": str(tmp_path),
            "power_pads": [{**desc, "power": "2.0"}],
        }
        plugin._run_simulation(
            board,
            [F_Cu, B_Cu],
            ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad],
            settings,
            {},
            ["U1-1"],
        )

        assert len(captured_configs) == 2
        assert captured_configs[0]["steady_state"] is True
        assert captured_configs[1]["steady_state"] is True


class TestDynamicHeatsinkPluginWiring:
    """Tests for dynamic 0D heatsink cache keys, model restriction, and metadata."""

    def test_operator_and_factorization_keys_invalidate_on_heatsink_settings(self):
        from ThermalSim.thermal_plugin import (
            _build_factorization_cache_key,
            _build_operator_cache_key,
        )
        from ThermalSim.thermal_solver import SolverConfig

        cfg = SolverConfig(sim_time=20.0, amb=25.0, dt_base=0.25, steps_target=80)
        base_settings = {
            "h_conv": 10.0,
            "pad_th": 1.0,
            "pad_k": 3.0,
            "pad_cap_areal": 0.0,
            "pad_sink_mode": "fixed",
            "pad_contact_factor": 0.2,
            "pad_sink_temp_use_ambient": True,
            "pad_sink_temp_c": 25.0,
            "heatsink_rth_kw": 5.0,
            "heatsink_material": "aluminium",
            "heatsink_mass_g": 50.0,
            "heatsink_cp_j_kgk": 897.0,
        }

        op_base = _build_operator_cache_key(
            "geom", "legacy", [3.5e-5, 3.5e-5], [1.5e-3], [7.5e-4, 7.5e-4], 5e-4, 25.0, base_settings
        )
        fac_base = _build_factorization_cache_key(op_base, cfg, False, base_settings)

        for key, new_val in (
            ("pad_sink_mode", "heatsink"),
            ("pad_contact_factor", 0.5),
            ("pad_sink_temp_use_ambient", False),
            ("pad_sink_temp_c", 40.0),
        ):
            mod = {**base_settings, key: new_val}
            op_mod = _build_operator_cache_key(
                "geom", "legacy", [3.5e-5, 3.5e-5], [1.5e-3], [7.5e-4, 7.5e-4], 5e-4, 25.0, mod
            )
            assert op_mod != op_base, f"operator_key did not change for {key}"
            assert _build_factorization_cache_key(op_mod, cfg, False, mod) != fac_base

        # Changing 0D heatsink parameters keeps PCB operator_key unchanged but invalidates factorization_cache_key
        for key, new_val in (
            ("heatsink_rth_kw", 2.5),
            ("heatsink_mass_g", 100.0),
            ("heatsink_material", "copper"),
        ):
            mod = {**base_settings, key: new_val}
            op_mod = _build_operator_cache_key(
                "geom", "legacy", [3.5e-5, 3.5e-5], [1.5e-3], [7.5e-4, 7.5e-4], 5e-4, 25.0, mod
            )
            assert op_mod == op_base
            assert _build_factorization_cache_key(op_mod, cfg, False, mod) != fac_base

    def _run_plugin_with_heatsink(
        self, tmp_path, monkeypatch, settings_overrides, with_eco1=True
    ):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_plugin import ThermalPlugin
        from ThermalSim.thermal_solver import SolverResult

        pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            bbox=EDA_RECT(4_500_000, 4_500_000, 1_000_000, 1_000_000),
            net_code=1,
            net_name="VIN",
            number="1",
        )
        drawings = (
            [MockDrawing(layer=Eco1_User, bbox=EDA_RECT(2_000_000, 2_000_000, 6_000_000, 6_000_000))]
            if with_eco1
            else []
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("U1", [pad])],
            drawings=drawings,
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        desc = plugin._pad_descriptor(board, "U1", pad)

        recorded = {
            "build_calls": [],
            "restrict_calls": [],
            "mf_kwargs": [],
            "legacy_kwargs": [],
            "report_k_norm": [],
        }

        def fake_build_heatsink_model(layer_count, rows, cols, dx, dy, H_map, settings, amb):
            recorded["build_calls"].append((layer_count, rows, cols, bool(np.any(H_map))))
            if settings.get("use_heatsink") and settings.get("pad_sink_mode") == "heatsink" and np.any(H_map):
                return {"kind": "full_heatsink_model", "nodes": layer_count * rows * cols}
            return None

        def fake_restrict_heatsink_model(model, mesh, layer_count):
            recorded["restrict_calls"].append(layer_count)
            return {"kind": "restricted_heatsink_model", "parent": model}

        def fake_mf_solver(
            config, solver_op, solver_cap, solver_q, solver_b, solver_hA,
            progress_cb, snap_cb, **kwargs
        ):
            recorded["mf_kwargs"].append(kwargs)
            n_nodes = solver_op.shape[0] if solver_cap is None else solver_cap.shape[0]
            if hasattr(solver_op, "layer_count"):
                T_out = np.full((solver_op.layer_count, solver_op.rows, solver_op.cols), 45.0)
            else:
                T_out = np.full(n_nodes, 45.0)
            return SolverResult(
                T=T_out,
                aborted=False,
                step_counter=1,
                total_solve_time=0.01,
                total_factor_time=0.01,
                factor_count=1,
                snapshot_stats=[],
                snapshot_files=[],
                phase_metrics=[],
                balance_history=[],
                k_norm_info={},
            )

        def fake_legacy_solver(
            config, K, C, Q, b, hA, layer_count, rows, cols, *args, **kwargs
        ):
            recorded["legacy_kwargs"].append(kwargs)
            return SolverResult(
                T=np.full((layer_count, rows, cols), 45.0),
                aborted=False,
                step_counter=1,
                total_solve_time=0.01,
                total_factor_time=0.01,
                factor_count=1,
                snapshot_stats=[],
                snapshot_files=[],
                phase_metrics=[],
                balance_history=[],
                k_norm_info={},
            )

        class DummyProgressDialog:
            def __init__(self, *args, **kwargs):
                pass

            def Update(self, value, newmsg=""):
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                pass

        monkeypatch.setattr(plugin_mod, "build_heatsink_model", fake_build_heatsink_model)
        monkeypatch.setattr(plugin_mod, "run_simulation_matrix_free", fake_mf_solver)
        monkeypatch.setattr(plugin_mod, "run_simulation", fake_legacy_solver)
        monkeypatch.setattr(plugin_mod.wx, "ProgressDialog", DummyProgressDialog, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: None, raising=False)
        monkeypatch.setattr(
            plugin_mod, "show_results_all_layers", lambda *a, **k: str(tmp_path / "hm.png")
        )
        monkeypatch.setattr(
            plugin_mod, "save_preview_image", lambda *a, **k: str(tmp_path / "prev.png")
        )
        monkeypatch.setattr(
            plugin_mod, "build_interactive_heatmap_payload", lambda *a, **k: {}
        )
        import webbrowser
        monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)

        def fake_write_report(*args, **kwargs):
            recorded["report_k_norm"].append(dict(kwargs.get("k_norm_info", {})))
            return str(tmp_path / "rep.html")

        monkeypatch.setattr(plugin_mod, "write_html_report", fake_write_report)

        settings = {
            "res": 2.0,
            "time": 20.0,
            "amb": 25.0,
            "thick": 1.6,
            "h_conv": 10.0,
            "use_heatsink": True,
            "pad_th": 1.0,
            "pad_k": 3.0,
            "pad_contact_factor": 0.6,
            "pad_sink_mode": "heatsink",
            "heatsink_rth_kw": 3.0,
            "heatsink_material": "aluminium",
            "heatsink_mass_g": 100.0,
            "heatsink_cp_j_kgk": 897.0,
            "show_all": True,
            "snapshots": False,
            "steady_state": True,
            "output_dir": str(tmp_path),
            "power_pads": [{**desc, "power": "2.0"}],
            **settings_overrides,
        }
        plugin._run_simulation(
            board,
            [F_Cu, B_Cu],
            ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad],
            settings,
            {},
            ["U1-1"],
        )
        return recorded

    def test_heatsink_model_wiring_uniform_legacy_and_metadata(self, tmp_path, monkeypatch):
        # Uniform matrix-free passes full heatsink_model
        rec_uniform = self._run_plugin_with_heatsink(
            tmp_path, monkeypatch, {"compute_engine": "fast_native", "mesh_mode": "uniform"}
        )
        assert len(rec_uniform["build_calls"]) == 1
        assert rec_uniform["mf_kwargs"][0]["heatsink_model"]["kind"] == "full_heatsink_model"

        # Legacy mesh_mode='adaptive' also routes through uniform matrix-free with full heatsink_model
        rec_legacy_mesh = self._run_plugin_with_heatsink(
            tmp_path, monkeypatch, {"compute_engine": "fast_native", "mesh_mode": "adaptive"}
        )
        assert len(rec_legacy_mesh["build_calls"]) == 1
        assert rec_legacy_mesh["mf_kwargs"][0]["heatsink_model"]["kind"] == "full_heatsink_model"

        # Legacy sparse path passes full heatsink_model
        rec_legacy = self._run_plugin_with_heatsink(
            tmp_path, monkeypatch, {"compute_engine": "legacy"}
        )
        assert rec_legacy["legacy_kwargs"][0]["heatsink_model"]["kind"] == "full_heatsink_model"

        # Fixed mode records heatsink_mode='fixed' and pad_contact_factor, omits heatsink_model kwarg
        rec_fixed = self._run_plugin_with_heatsink(
            tmp_path, monkeypatch, {"compute_engine": "legacy", "pad_sink_mode": "fixed", "pad_contact_factor": 0.4}
        )
        assert "heatsink_model" not in rec_fixed["legacy_kwargs"][0]
        assert rec_fixed["report_k_norm"][0]["heatsink_mode"] == "fixed"
        assert rec_fixed["report_k_norm"][0]["pad_contact_factor"] == 0.4

        # Dynamic heatsink mode without User.Eco1 shapes records inactive reason
        rec_no_eco = self._run_plugin_with_heatsink(
            tmp_path, monkeypatch, {"compute_engine": "legacy", "pad_sink_mode": "heatsink"}, with_eco1=False
        )
        assert "heatsink_model" not in rec_no_eco["legacy_kwargs"][0]
        info = rec_no_eco["report_k_norm"][0]
        assert info["heatsink_enabled"] is False
        assert info["heatsink_mode"] == "heatsink"
        assert info["heatsink_inactive_reason"] == "No User.Eco1 thermal pad shapes found on board"


class TestCompactLowRamPluginWiring:
    """Tests for zero-peak compact operator, direct adaptive, cleanup, and fallback wiring."""

    def _setup_plugin_env(self, tmp_path, monkeypatch, with_eco1=True):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_plugin import ThermalPlugin

        pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            bbox=EDA_RECT(4_500_000, 4_500_000, 1_000_000, 1_000_000),
            net_code=1,
            net_name="VIN",
            number="1",
        )
        drawings = (
            [MockDrawing(layer=Eco1_User, bbox=EDA_RECT(2_000_000, 2_000_000, 6_000_000, 6_000_000))]
            if with_eco1
            else []
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("U1", [pad])],
            drawings=drawings,
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        desc = plugin._pad_descriptor(board, "U1", pad)

        class DummyProgressDialog:
            def __init__(self, *args, **kwargs):
                pass

            def Update(self, value, newmsg=""):
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                pass

        monkeypatch.setattr(plugin_mod.wx, "ProgressDialog", DummyProgressDialog, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: None, raising=False)
        monkeypatch.setattr(
            plugin_mod, "show_results_all_layers", lambda *a, **k: str(tmp_path / "hm.png")
        )
        monkeypatch.setattr(
            plugin_mod, "save_preview_image", lambda *a, **k: str(tmp_path / "prev.png")
        )
        monkeypatch.setattr(
            plugin_mod, "build_interactive_heatmap_payload", lambda *a, **k: {}
        )
        import webbrowser
        monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
        monkeypatch.setattr(
            plugin_mod, "write_html_report", lambda *a, **k: str(tmp_path / "rep.html")
        )

        base_settings = {
            "res": 2.0,
            "time": 20.0,
            "amb": 25.0,
            "thick": 1.6,
            "h_conv": 10.0,
            "use_heatsink": with_eco1,
            "pad_th": 1.0,
            "pad_k": 3.0,
            "pad_contact_factor": 0.6,
            "pad_sink_mode": "heatsink" if with_eco1 else "fixed",
            "heatsink_rth_kw": 3.0,
            "heatsink_mass_g": 100.0,
            "heatsink_cp_j_kgk": 897.0,
            "show_all": True,
            "snapshots": False,
            "steady_state": True,
            "output_dir": str(tmp_path),
            "power_pads": [{**desc, "power": "2.0"}],
        }
        return plugin, board, pad, base_settings

    def test_uniform_fast_native_uses_compact_operator_sparse_power_and_cleans_up(
        self, tmp_path, monkeypatch
    ):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_solver import (
            CompactStructuredThermalOperator,
            SolverResult,
            SparsePowerVector,
        )

        plugin, board, pad, settings = self._setup_plugin_env(tmp_path, monkeypatch, with_eco1=True)
        settings["compute_engine"] = "fast_native"
        settings["mesh_mode"] = "uniform"

        recorded = {"mf_args": [], "scratch_clears": 0, "op_cache_clears": 0}
        orig_clear_caches = CompactStructuredThermalOperator.clear_caches

        def spy_clear_caches(self_op):
            recorded["op_cache_clears"] += 1
            return orig_clear_caches(self_op)

        def spy_clear_scratch():
            recorded["scratch_clears"] += 1

        def forbid_structured(*args, **kwargs):
            raise AssertionError("build_structured_operator must not be called on compact path")

        def spy_mf_solver(config, op, C, Q, b, hA, *args, **kwargs):
            recorded["mf_args"].append((op, C, Q, b, hA, kwargs.get("heatsink_model")))
            return SolverResult(
                T=np.full((op.layer_count, op.rows, op.cols), 40.0),
                k_norm_info={},
                step_counter=1,
                total_solve_time=0.01,
                total_factor_time=0.0,
                factor_count=0,
            )

        monkeypatch.setattr(CompactStructuredThermalOperator, "clear_caches", spy_clear_caches)
        monkeypatch.setattr(plugin_mod, "clear_native_scratch", spy_clear_scratch)
        monkeypatch.setattr(plugin_mod, "run_simulation_matrix_free", spy_mf_solver)
        # Keep _ORIG_BUILD_STRUCTURED_OPERATOR check satisfied while verifying build_structured_operator isn't called
        monkeypatch.setattr(plugin_mod, "build_structured_operator", forbid_structured)
        monkeypatch.setattr(plugin_mod, "_ORIG_BUILD_STRUCTURED_OPERATOR", forbid_structured)

        plugin._run_simulation(
            board,
            [F_Cu, B_Cu],
            ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad],
            settings,
            {},
            ["U1-1"],
        )

        assert len(recorded["mf_args"]) == 1
        op, C, Q, b, hA, hs_model = recorded["mf_args"][0]
        assert isinstance(op, CompactStructuredThermalOperator)
        assert C is None
        assert b is None
        assert hA is None
        assert isinstance(Q, SparsePowerVector)
        assert np.isclose(Q.total_power(), 2.0)
        assert hs_model is not None
        assert hs_model.bottom_layer_only is True
        assert recorded["op_cache_clears"] == 1
        assert recorded["scratch_clears"] == 1

    def test_compact_uniform_falls_back_lazily_to_legacy_when_fast_solver_raises(
        self, tmp_path, monkeypatch
    ):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_solver import SolverResult

        plugin, board, pad, settings = self._setup_plugin_env(tmp_path, monkeypatch, with_eco1=True)
        settings["compute_engine"] = "fast_native"
        settings["mesh_mode"] = "uniform"

        recorded = {"legacy_calls": [], "scratch_clears": 0}

        def failing_mf(*args, **kwargs):
            raise RuntimeError("boom")

        def spy_legacy(config, K, C, Q, b, hA, layer_count, rows, cols, *args, **kwargs):
            recorded["legacy_calls"].append((C, Q, kwargs.get("heatsink_model")))
            return SolverResult(
                T=np.full((layer_count, rows, cols), 42.0),
                k_norm_info={},
                step_counter=1,
                total_solve_time=0.01,
                total_factor_time=0.01,
                factor_count=1,
            )

        monkeypatch.setattr(plugin_mod, "run_simulation_matrix_free", failing_mf)
        monkeypatch.setattr(plugin_mod, "run_simulation", spy_legacy)
        monkeypatch.setattr(
            plugin_mod,
            "clear_native_scratch",
            lambda: recorded.__setitem__("scratch_clears", recorded["scratch_clears"] + 1),
        )

        plugin._run_simulation(
            board,
            [F_Cu, B_Cu],
            ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad],
            settings,
            {},
            ["U1-1"],
        )

        assert len(recorded["legacy_calls"]) == 1
        C_fb, Q_fb, hs_fb = recorded["legacy_calls"][0]
        assert isinstance(C_fb, np.ndarray)
        assert isinstance(Q_fb, np.ndarray)
        assert np.isclose(np.sum(Q_fb), 2.0)
        assert hs_fb is not None
        assert hs_fb.bottom_layer_only is False
        assert recorded["scratch_clears"] == 1

    def test_dialog_close_callback_clears_in_memory_caches_only(self, tmp_path, monkeypatch):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_plugin import ThermalPlugin
        from ThermalSim.workflow import GeometryCache

        disk_dir = tmp_path / "geom_cache"
        plugin = ThermalPlugin()
        plugin.defaults()
        plugin.geometry_cache = GeometryCache(persistent=True, cache_dir=str(disk_dir))

        pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            bbox=EDA_RECT(4_500_000, 4_500_000, 1_000_000, 1_000_000),
            net_code=1,
            net_name="VIN",
            number="1",
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("U1", [pad])],
        )
        # Seed in-memory + disk geometry cache and in-memory operator/factorization/electrical caches
        from ThermalSim.geometry_mapper import build_geometry_state
        state = build_geometry_state(
            board=board,
            copper_ids=[F_Cu, B_Cu],
            rows=4,
            cols=4,
            x_min=0.0,
            y_min=0.0,
            res=1.0,
            settings={"ignore_traces": False, "ignore_polygons": False, "use_heatsink": False},
            via_factor=1300.0,
            pads_list=[pad],
        )
        plugin.geometry_cache.put("geom-key", state)
        npz_files_before = list(disk_dir.glob("*.npz"))
        assert len(npz_files_before) == 1

        plugin.operator_cache.put("op-key", ("op",))
        plugin.electrical_cache.put("el-key", ("el",))
        released = []

        class DummyFac:
            def release(self):
                released.append(True)

        plugin.factorization_cache.put("fac-key", DummyFac())

        captured = {}

        class DummyDialog:
            def __init__(self, *args, **kwargs):
                captured["close_callback"] = kwargs["close_callback"]

            def Show(self):
                pass

        monkeypatch.setattr(plugin_mod.pcbnew, "GetBoard", lambda: board, raising=False)
        monkeypatch.setattr(plugin_mod, "_find_pcb_editor_parent", lambda _b=None: None)
        monkeypatch.setattr(plugin_mod, "SettingsDialog", DummyDialog)
        monkeypatch.setattr(plugin_mod, "parse_stackup_from_board_file", lambda *a, **k: {})
        monkeypatch.setattr(plugin, "_show_startup_progress", lambda *a, **k: None)
        monkeypatch.setattr(plugin, "_close_startup_progress", lambda *a, **k: None)
        monkeypatch.setattr(plugin, "_load_settings", lambda *a, **k: {})
        monkeypatch.setattr(plugin, "_detect_copper_layers", lambda _b: ([F_Cu, B_Cu], ["F.Cu", "B.Cu"]))
        monkeypatch.setattr(board, "GetBoundingBox", lambda: EDA_RECT(0, 0, 10_000_000, 10_000_000), raising=False)

        plugin.Run()
        assert plugin.settings_dialog is not None

        captured["close_callback"]()
        assert plugin.settings_dialog is None
        assert plugin.operator_cache.key is None
        assert plugin.operator_cache.value is None
        assert plugin.electrical_cache.key is None
        assert plugin.electrical_cache.value is None
        assert plugin.factorization_cache.key is None
        assert plugin.factorization_cache.value is None
        assert released == [True]
        assert plugin.geometry_cache.key is None
        assert plugin.geometry_cache.value is None
        # Disk .npz cache file must remain intact
        assert list(disk_dir.glob("*.npz")) == npz_files_before


class TestGuiResponsivenessAndProgress:
    """Tests for end-to-end progress reporting, cancellation, and fast snapshots."""

    def _make_board_and_plugin(self):
        from ThermalSim.thermal_plugin import ThermalPlugin
        from ThermalSim.workflow import GeometryCache

        pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            bbox=EDA_RECT(4_500_000, 4_500_000, 1_000_000, 1_000_000),
            net_code=1,
            net_name="VIN",
            number="1",
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("U1", [pad])],
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        plugin.geometry_cache = GeometryCache(persistent=False)
        plugin.board = board
        plugin.copper_ids = [F_Cu, B_Cu]
        plugin.bbox = EDA_RECT(0, 0, 10_000_000, 10_000_000)
        plugin.pads_list = [pad]
        plugin.stack_info = {}
        desc = plugin._pad_descriptor(board, "U1", pad)
        return plugin, board, pad, desc

    def _make_settings(self, tmp_path, desc, **overrides):
        base = {
            "res": 2.0,
            "time": 20.0,
            "amb": 25.0,
            "thick": 1.6,
            "h_conv": 10.0,
            "pad_th": 1.0,
            "pad_k": 3.0,
            "show_all": True,
            "snapshots": False,
            "steady_state": True,
            "compute_engine": "legacy",
            "output_dir": str(tmp_path),
            "power_pads": [{**desc, "power": "2.0"}],
        }
        base.update(overrides)
        return base

    def test_run_simulation_drives_progress_across_all_stages(self, tmp_path, monkeypatch):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_solver import SolverResult

        plugin, board, pad, desc = self._make_board_and_plugin()
        updates = []
        lifecycle = {"hidden": 0, "destroyed": 0}
        yield_calls = []

        class RecordingProgressDialog:
            def __init__(self, title, message, maximum=100, **kwargs):
                updates.append((0, message))

            def Update(self, value, newmsg=""):
                updates.append((int(value), newmsg))
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                lifecycle["hidden"] += 1

            def Destroy(self):
                lifecycle["destroyed"] += 1

        class DummyApp:
            def Yield(self, only_if_needed=False):
                yield_calls.append(only_if_needed)

        def fake_solver(config, K, C, Q, b, hA, layer_count, rows, cols,
                        progress_cb, snap_cb, **kwargs):
            progress_cb(1, 2)
            progress_cb(2, 2)
            return SolverResult(
                T=np.full((layer_count, rows, cols), 42.0),
                aborted=False,
                step_counter=2,
                total_solve_time=0.01,
                total_factor_time=0.01,
                factor_count=1,
                snapshot_stats=[],
                snapshot_files=[],
                phase_metrics=[],
                balance_history=[],
                k_norm_info={},
            )

        # Also verify build_geometry_state without progress_callback/cancel_check kwargs works
        orig_build_geom = plugin_mod.build_geometry_state

        def no_kw_build_geom(board, copper_ids, rows, cols, x_min, y_min, res,
                              settings, via_factor, pads_list):
            return orig_build_geom(
                board, copper_ids, rows, cols, x_min, y_min, res,
                settings, via_factor, pads_list,
            )

        monkeypatch.setattr(plugin_mod, "build_geometry_state", no_kw_build_geom)
        monkeypatch.setattr(plugin_mod, "run_simulation", fake_solver)
        monkeypatch.setattr(plugin_mod.wx, "ProgressDialog", RecordingProgressDialog, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: DummyApp(), raising=False)
        monkeypatch.setattr(plugin_mod, "show_results_all_layers", lambda *a, **k: str(tmp_path / "hm.png"))
        monkeypatch.setattr(plugin_mod, "save_preview_image", lambda *a, **k: str(tmp_path / "prev.png"))
        monkeypatch.setattr(plugin_mod, "build_interactive_heatmap_payload", lambda *a, **k: {})
        monkeypatch.setattr(plugin_mod, "write_html_report", lambda *a, **k: str(tmp_path / "rep.html"))
        import webbrowser
        monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)

        plugin._run_simulation(
            board, [F_Cu, B_Cu], ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad], self._make_settings(tmp_path, desc), {}, ["U1-1"],
        )

        messages = [msg for _, msg in updates]
        percents = [pct for pct, _ in updates]
        assert any("Mapping exact PCB geometry" in m for m in messages)
        assert any("Building thermal operator" in m for m in messages)
        assert any("Solving steady-state thermal model" in m for m in messages)
        assert any("Rendering layer heatmaps" in m for m in messages)
        assert any("Generating board preview" in m for m in messages)
        assert any("Building interactive heatmap" in m for m in messages)
        assert any("Writing HTML report" in m for m in messages)
        assert percents[-1] == 100
        assert lifecycle["hidden"] >= 1
        assert lifecycle["destroyed"] >= 1
        assert len(yield_calls) > 0

    def test_run_simulation_cancels_cleanly_on_geometry_cancelled_error(
        self, tmp_path, monkeypatch
    ):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_plugin import GeometryCancelledError

        plugin, board, pad, desc = self._make_board_and_plugin()
        destroyed = []
        message_boxes = []
        dialog_states = []

        class DummyDialog:
            def set_run_state(self, status, message=""):
                dialog_states.append((status, message))

        class DummyProgressDialog:
            def __init__(self, *args, **kwargs):
                pass

            def Update(self, value, newmsg=""):
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                destroyed.append(True)

        def cancelling_geom(*args, **kwargs):
            raise GeometryCancelledError("user cancelled geometry")

        plugin.settings_dialog = DummyDialog()
        monkeypatch.setattr(plugin_mod, "build_geometry_state", cancelling_geom)
        monkeypatch.setattr(plugin_mod.wx, "ProgressDialog", DummyProgressDialog, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: None, raising=False)
        monkeypatch.setattr(
            plugin_mod.wx, "MessageBox",
            lambda *a, **k: message_boxes.append(a), raising=False,
        )

        plugin._run_simulation(
            board, [F_Cu, B_Cu], ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad], self._make_settings(tmp_path, desc), {}, ["U1-1"],
        )

        assert message_boxes == []
        assert destroyed == [True]
        assert dialog_states[-1][0] == "cancelled"
        manifests = list(tmp_path.glob("Thermalsim_*/run_manifest.json"))
        assert len(manifests) == 1
        assert json.loads(manifests[0].read_text(encoding="utf-8"))["status"] == "cancelled"

    def test_run_simulation_cancels_cleanly_during_operator_and_solver(
        self, tmp_path, monkeypatch
    ):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_solver import SolverResult

        plugin, board, pad, desc = self._make_board_and_plugin()
        message_boxes = []
        dialog_states = []
        solver_called = []

        class DummyDialog:
            def set_run_state(self, status, message=""):
                dialog_states.append((status, message))

        class CancelOnOperatorDialog:
            def __init__(self, *args, **kwargs):
                self.destroyed = False

            def Update(self, value, newmsg=""):
                if "thermal operator" in newmsg.lower():
                    return (False, False)
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                self.destroyed = True

        plugin.settings_dialog = DummyDialog()
        monkeypatch.setattr(plugin_mod.wx, "ProgressDialog", CancelOnOperatorDialog, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: None, raising=False)
        monkeypatch.setattr(
            plugin_mod.wx, "MessageBox",
            lambda *a, **k: message_boxes.append(a), raising=False,
        )
        monkeypatch.setattr(
            plugin_mod, "run_simulation",
            lambda *a, **k: solver_called.append(True),
        )

        plugin._run_simulation(
            board, [F_Cu, B_Cu], ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad], self._make_settings(tmp_path, desc), {}, ["U1-1"],
        )

        assert solver_called == []
        assert message_boxes == []
        assert dialog_states[-1][0] == "cancelled"

        # Now test cancelling mid-solver
        dialog_states.clear()

        class CancelDuringSolverDialog:
            def __init__(self, *args, **kwargs):
                pass

            def Update(self, value, newmsg=""):
                if "Solving" in newmsg:
                    return (False, False)
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                pass

        def aborting_solver(config, K, C, Q, b, hA, layer_count, rows, cols,
                            progress_cb, snap_cb, cancel_check=None, **kwargs):
            assert cancel_check is not None and cancel_check() is True
            return SolverResult(
                T=np.full((layer_count, rows, cols), 25.0),
                aborted=True,
                step_counter=0,
                total_solve_time=0.0,
                total_factor_time=0.0,
                factor_count=0,
                snapshot_stats=[],
                snapshot_files=[],
                phase_metrics=[],
                balance_history=[],
                k_norm_info={},
            )

        monkeypatch.setattr(plugin_mod.wx, "ProgressDialog", CancelDuringSolverDialog, raising=False)
        monkeypatch.setattr(plugin_mod, "run_simulation", aborting_solver)

        plugin._run_simulation(
            board, [F_Cu, B_Cu], ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad], self._make_settings(tmp_path, desc), {}, ["U1-1"],
        )

        assert message_boxes == []
        assert dialog_states[-1][0] == "cancelled"

    def test_capture_board_snapshot_avoids_exceptions_and_supports_progress_callback(self):
        from ThermalSim.thermal_plugin import GeometryCancelledError, ThermalPlugin

        class PCB_ARC(MockTrack):
            def __init__(self):
                super().__init__(layer=F_Cu, bbox=EDA_RECT(0, 0, 2_000_000, 2_000_000))

            def GetMid(self):
                return VECTOR2I(1_000_000, 1_500_000)

        track = MockTrack(layer=F_Cu, bbox=EDA_RECT(0, 0, 4_000_000, 200_000))
        arc = PCB_ARC()
        pad = MockPad(position=VECTOR2I(2_000_000, 2_000_000), layer=F_Cu, number="1")
        zone = MockZone(layers=[F_Cu], filled=True)
        board = MockBoard(
            tracks=[track, arc],
            zones=[zone],
            footprints=[MockFootprint("U1", [pad])],
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        bbox = EDA_RECT(0, 0, 10_000_000, 10_000_000)

        events = []
        snap1 = plugin._capture_board_snapshot(
            board, [F_Cu, B_Cu], bbox,
            progress_callback=lambda d, t, m: events.append((d, t, m)) or True,
        )
        snap2 = plugin._capture_board_snapshot(board, [F_Cu, B_Cu], bbox)
        assert snap1.fingerprint == snap2.fingerprint
        assert len(events) >= 2
        assert events[0][0] == 0
        assert events[-1][0] == events[-1][1]

        # Cancelling via progress_callback raises GeometryCancelledError
        import pytest
        with pytest.raises(GeometryCancelledError):
            plugin._capture_board_snapshot(
                board, [F_Cu, B_Cu], bbox,
                progress_callback=lambda d, t, m: False,
            )

        # Ensure straight track without GetMid does not trigger exceptions
        assert not hasattr(track, "GetMid")

    def test_generate_preview_progress_and_cancellation(self, tmp_path, monkeypatch):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_plugin import GeometryCancelledError

        plugin, board, pad, desc = self._make_board_and_plugin()
        progress_updates = []
        message_boxes = []
        status_labels = []

        class DummyDialog:
            class _Lbl:
                def SetLabel(self, text):
                    status_labels.append(text)

            lbl_preflight = _Lbl()

            def _set_footer_status(self, title, msg):
                status_labels.append(f"{title}: {msg}")

        class RecordingProgressDialog:
            def __init__(self, title, message, maximum=100, **kwargs):
                self.cancel_on_geom = False
                progress_updates.append((0, message))

            def Update(self, value, newmsg=""):
                progress_updates.append((int(value), newmsg))
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                pass

        plugin.settings_dialog = DummyDialog()
        monkeypatch.setattr(plugin_mod.wx, "ProgressDialog", RecordingProgressDialog, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: None, raising=False)
        monkeypatch.setattr(
            plugin_mod.wx, "MessageBox",
            lambda *a, **k: message_boxes.append(a), raising=False,
        )
        monkeypatch.setattr(
            plugin_mod, "save_preview_image",
            lambda *a, **k: str(tmp_path / "preview.png"),
        )

        out = plugin.generate_preview(
            self._make_settings(tmp_path, desc), ["F.Cu", "B.Cu"]
        )
        assert out == str(tmp_path / "preview.png")
        assert progress_updates[-1][0] == 100
        assert message_boxes == []

        # Cancellation during geometry mapping returns None cleanly with no error box
        plugin.geometry_cache.clear()
        monkeypatch.setattr(
            plugin_mod, "build_geometry_state",
            lambda *a, **k: (_ for _ in ()).throw(GeometryCancelledError("cancelled")),
        )
        out_cancelled = plugin.generate_preview(
            self._make_settings(tmp_path, desc), ["F.Cu", "B.Cu"]
        )
        assert out_cancelled is None
        assert message_boxes == []
        assert "Preview cancelled." in status_labels


class _PluginShapeLayerSet:
    """Minimal KiCad LayerSet supporting Contains and CuStack."""

    def __init__(self, layers):
        self._layers = list(layers)

    def Contains(self, layer_id):
        return int(layer_id) in self._layers

    def CuStack(self):
        return list(self._layers)

    def IntSeq(self):
        return list(self._layers)


class _PluginShapePad(MockPad):
    """MockPad exposing GetLayerSet, GetShape, GetSize, orientation, and parent."""

    def __init__(
        self,
        *,
        layer_set=None,
        shape=None,
        size=None,
        orientation_deg=0.0,
        corner_radius_nm=0,
        parent=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._layer_set = _PluginShapeLayerSet(layer_set) if layer_set is not None else None
        self._shape = shape
        self._size = size
        self._orientation_deg = float(orientation_deg)
        self._corner_radius_nm = int(corner_radius_nm)
        self._parent = parent

    def GetLayerSet(self):
        if self._layer_set is None:
            return _PluginShapeLayerSet([self._layer])
        return self._layer_set

    def GetShape(self):
        return self._shape

    def GetSize(self):
        if self._size is not None:
            return self._size
        return VECTOR2I(self._bbox.GetWidth(), self._bbox.GetHeight())

    def GetOrientationDegrees(self):
        return self._orientation_deg

    def GetRoundRectCornerRadius(self):
        return self._corner_radius_nm

    def GetParent(self):
        return self._parent


class TestPadAlignmentAndBottomLayerFixes:
    """Regression tests for v0.4.5 pad copper/power alignment and B.Cu LayerSet fixes."""

    def test_bcu_smd_pad_with_fcu_getlayer_resolves_bcu_and_injects_onto_bcu_copper(
        self, monkeypatch
    ):
        import ThermalSim.geometry_mapper as geom_mod
        from ThermalSim.thermal_plugin import ThermalPlugin, _pad_target_layer_index

        monkeypatch.setattr(geom_mod.pcbnew, "PAD_SHAPE_RECT", 1, raising=False)
        bcu_pad = _PluginShapePad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,  # KiCad 10 reports 0 (F_Cu) on B.Cu SMD pads
            layer_set=[B_Cu],
            shape=1,
            size=VECTOR2I(2_000_000, 2_000_000),
            bbox=EDA_RECT(4_000_000, 4_000_000, 2_000_000, 2_000_000),
            net_code=1,
            net_name="SW",
            number="1",
        )
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("Q1", [bcu_pad])],
        )
        plugin = ThermalPlugin()
        plugin.defaults()
        plugin.copper_ids = [F_Cu, B_Cu]

        # Descriptors and lookup must report B.Cu instead of F.Cu
        selected_desc = plugin._descriptors_from_selected_pads(board, [("Q1-1", bcu_pad)])
        assert selected_desc[0]["layer"] == "B.Cu"
        _, by_name = plugin._build_pad_lookup(board)
        assert by_name["Q1-1 [SW]"][0]["layer"] == "B.Cu"

        copper_ids = [F_Cu, B_Cu]
        lid_to_idx = {F_Cu: 0, B_Cu: 1}
        assert _pad_target_layer_index(board, copper_ids, bcu_pad, lid_to_idx) == 1

        rows = cols = 20
        res = 0.5
        state = geom_mod.build_geometry_state(
            board=board,
            copper_ids=copper_ids,
            rows=rows,
            cols=cols,
            x_min=0.0,
            y_min=0.0,
            res=res,
            settings={"ignore_traces": True, "ignore_polygons": True, "use_heatsink": False, "area_mode": "full"},
            via_factor=1300.0,
            pads_list=[bcu_pad],
        )
        contribs = _build_sparse_pad_contributions(
            board, copper_ids, [bcu_pad], rows, cols, 0.0, 0.0, res, area_mask=state.area_mask
        )
        assert contribs[0].indices.size > 0
        assert np.all(contribs[0].indices >= rows * cols)
        assert not np.any(state.copper_mask[0])
        assert np.all(state.copper_mask.ravel()[contribs[0].indices])

    def test_pth_pad_prefers_parent_footprint_layer(self):
        from ThermalSim.thermal_plugin import _pad_target_layer_index

        fp_bot = MockFootprint("J1", [])
        fp_bot.GetLayer = lambda: B_Cu
        pth_pad = _PluginShapePad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            layer_set=[F_Cu, B_Cu],
            parent=fp_bot,
            number="1",
        )
        board = MockBoard(layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"})
        copper_ids = [F_Cu, B_Cu]
        assert _pad_target_layer_index(board, copper_ids, pth_pad, {F_Cu: 0, B_Cu: 1}) == 1

    def test_roundrect_circle_rotated_narrow_pads_have_zero_fr4_power_cells(self, monkeypatch):
        import ThermalSim.geometry_mapper as geom_mod

        monkeypatch.setattr(geom_mod.pcbnew, "PAD_SHAPE_RECT", 1, raising=False)
        monkeypatch.setattr(geom_mod.pcbnew, "PAD_SHAPE_CIRCLE", 2, raising=False)
        monkeypatch.setattr(geom_mod.pcbnew, "PAD_SHAPE_ROUNDRECT", 3, raising=False)

        diag_half = int(round(0.5 * (4.0 + 1.0) / np.sqrt(2.0) * 1e6))
        rotated_pad = _PluginShapePad(
            position=VECTOR2I(10_000_000, 10_000_000),
            layer=F_Cu,
            layer_set=[F_Cu],
            shape=1,
            size=VECTOR2I(4_000_000, 1_000_000),
            orientation_deg=45.0,
            bbox=EDA_RECT(10_000_000 - diag_half, 10_000_000 - diag_half, 2 * diag_half, 2 * diag_half),
            number="1",
        )
        circle_pad = _PluginShapePad(
            position=VECTOR2I(20_000_000, 10_000_000),
            layer=F_Cu,
            layer_set=[F_Cu],
            shape=2,
            size=VECTOR2I(4_000_000, 4_000_000),
            bbox=EDA_RECT(18_000_000, 8_000_000, 4_000_000, 4_000_000),
            number="2",
        )
        roundrect_pad = _PluginShapePad(
            position=VECTOR2I(30_000_000, 10_000_000),
            layer=F_Cu,
            layer_set=[F_Cu],
            shape=3,
            size=VECTOR2I(4_000_000, 4_000_000),
            corner_radius_nm=1_000_000,
            bbox=EDA_RECT(28_000_000, 8_000_000, 4_000_000, 4_000_000),
            number="3",
        )
        narrow_pad = _PluginShapePad(
            position=VECTOR2I(10_250_000, 20_250_000),
            layer=F_Cu,
            layer_set=[F_Cu],
            shape=1,
            size=VECTOR2I(120_000, 120_000),
            bbox=EDA_RECT(10_190_000, 20_190_000, 120_000, 120_000),
            number="4",
        )
        pads = [rotated_pad, circle_pad, roundrect_pad, narrow_pad]
        board = MockBoard(
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
            footprints=[MockFootprint("U1", pads)],
        )
        copper_ids = [F_Cu, B_Cu]
        rows, cols, res = 60, 80, 0.5
        state = geom_mod.build_geometry_state(
            board=board,
            copper_ids=copper_ids,
            rows=rows,
            cols=cols,
            x_min=0.0,
            y_min=0.0,
            res=res,
            settings={"ignore_traces": True, "ignore_polygons": True, "use_heatsink": False, "area_mode": "full"},
            via_factor=1300.0,
            pads_list=pads,
        )
        contribs = _build_sparse_pad_contributions(
            board, copper_ids, pads, rows, cols, 0.0, 0.0, res, area_mask=state.area_mask
        )
        flat_copper = state.copper_mask.ravel()
        for c in contribs:
            assert c.indices.size > 0
            assert np.all(flat_copper[c.indices])

    def test_geometry_key_invalidates_stale_v044_and_bcu_layerset_changes(self):
        from ThermalSim.thermal_plugin import ThermalPlugin
        from ThermalSim.workflow import GridEstimate, geometry_cache_key

        pad_fcu = _PluginShapePad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            layer_set=[F_Cu],
            number="1",
        )
        pad_bcu = _PluginShapePad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            layer_set=[B_Cu],
            number="1",
        )
        bbox = EDA_RECT(0, 0, 10_000_000, 10_000_000)
        board_f = MockBoard(footprints=[MockFootprint("U1", [pad_fcu])])
        board_b = MockBoard(footprints=[MockFootprint("U1", [pad_bcu])])
        grid = GridEstimate(
            rows=20,
            cols=20,
            layer_count=2,
            requested_res_mm=0.5,
            actual_res_mm=0.5,
            x_min_mm=0.0,
            y_min_mm=0.0,
            width_mm=10.0,
            height_mm=10.0,
            auto_coarsened=False,
            expert_limits=False,
            max_cells=200_000,
            target_cells=100_000,
            detail_level="balanced",
            node_budget=1_200_000,
            memory_mb_low=1.0,
            memory_mb_high=2.0,
            runtime_class="fast",
        )
        settings = {"ignore_traces": False, "ignore_polygons": False}
        plugin = ThermalPlugin()
        plugin.defaults()

        key_f = plugin._geometry_key(board_f, [F_Cu, B_Cu], bbox, grid, settings, [pad_fcu])
        key_b = plugin._geometry_key(board_b, [F_Cu, B_Cu], bbox, grid, settings, [pad_bcu])
        # Legacy v0.4.4 key without ("pad_raster_v", 2)
        snap_f = plugin._capture_board_snapshot(board_f, [F_Cu, B_Cu], bbox)
        legacy_key = geometry_cache_key(
            snap_f, grid, settings, [(pad_fcu.GetNumber(), pad_fcu.GetLayer(), 5_000_000, 5_000_000)]
        )

        assert key_f != legacy_key
        assert key_f != key_b


class TestProgressEtaAndPcgReporting:
    """Tests for English ETA formatting, PD_REMAINING_TIME removal, and 3-arg PCG progress."""

    def test_progress_dialog_omits_pd_remaining_time_and_keeps_elapsed_time(self, monkeypatch):
        import ThermalSim.thermal_plugin as plugin_mod

        captured_styles = []

        class RecordingStyleProgressDialog:
            def __init__(self, title, message, maximum=100, **kwargs):
                captured_styles.append(kwargs.get("style", 0))

            def Update(self, value, newmsg=""):
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                pass

        monkeypatch.setattr(plugin_mod.wx, "PD_CAN_ABORT", 0x01, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "PD_APP_MODAL", 0x02, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "PD_REMAINING_TIME", 0x04, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "PD_ELAPSED_TIME", 0x08, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "PD_AUTO_HIDE", 0x10, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "ProgressDialog", RecordingStyleProgressDialog, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: None, raising=False)

        with plugin_mod._UiProgressReporter(title="ThermalSim", initial_message="Preparing..."):
            pass

        assert len(captured_styles) == 1
        style = captured_styles[0]
        assert (style & plugin_mod.wx.PD_REMAINING_TIME) == 0
        assert (style & plugin_mod.wx.PD_ELAPSED_TIME) == plugin_mod.wx.PD_ELAPSED_TIME
        assert (style & plugin_mod.wx.PD_CAN_ABORT) == plugin_mod.wx.PD_CAN_ABORT
        assert (style & plugin_mod.wx.PD_APP_MODAL) == plugin_mod.wx.PD_APP_MODAL
        assert (style & plugin_mod.wx.PD_AUTO_HIDE) == plugin_mod.wx.PD_AUTO_HIDE

    def test_format_eta_seconds_english_ranges(self):
        from ThermalSim.thermal_plugin import _format_eta_seconds

        assert _format_eta_seconds(0.2) == "~1s remaining"
        assert _format_eta_seconds(12.4) == "~12s remaining"
        assert _format_eta_seconds(59.6) == "~1m 00s remaining"
        assert _format_eta_seconds(135.0) == "~2m 15s remaining"
        assert _format_eta_seconds(3900.0) == "~1h 05m remaining"

    def test_run_in_background_with_progress_state_stays_at_base_when_zero(self, monkeypatch):
        import threading
        import time
        import ThermalSim.thermal_plugin as plugin_mod

        updates = []

        class RecordingProgressDialog:
            def __init__(self, title, message, maximum=100, **kwargs):
                updates.append((0, message))

            def Update(self, value, newmsg=""):
                updates.append((int(value), newmsg))
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                pass

        monkeypatch.setattr(plugin_mod.wx, "ProgressDialog", RecordingProgressDialog, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: None, raising=False)

        state = {"current": 0, "total": 100, "message": ""}
        release_worker = threading.Event()

        def worker():
            time.sleep(0.06)
            state["current"] = 50
            state["message"] = "Solving steady-state (PCG iter 10, residual 1e-4 -> 1e-7)..."
            release_worker.wait(timeout=1.0)

        with plugin_mod._UiProgressReporter(
            title="ThermalSim",
            initial_message="Preparing...",
            throttle_s=0.01,
            poll_interval_s=0.01,
        ) as reporter:
            t = threading.Timer(0.12, release_worker.set)
            t.start()
            try:
                reporter.run_in_background(
                    worker,
                    54,
                    88,
                    "Solving steady-state thermal model...",
                    progress_state=state,
                    message_fn=lambda s: s.get("message") or "Solving steady-state thermal model...",
                )
            finally:
                t.cancel()
                release_worker.set()

        # Before PCG updates arrive, percentage must stay cleanly at base (54), never creep upward
        pre_pcg_pcts = [
            pct for pct, msg in updates[1:]
            if "PCG iter" not in msg and pct < 88
        ]
        assert len(pre_pcg_pcts) >= 2
        assert all(pct == 54 for pct in pre_pcg_pcts)
        assert any("PCG iter 10" in msg for _, msg in updates)
        assert updates[-1][0] == 88

    def test_run_simulation_propagates_steady_state_pcg_and_transient_eta(self, tmp_path, monkeypatch):
        import threading
        import time
        import webbrowser
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_solver import SolverResult

        suite = TestGuiResponsivenessAndProgress()
        plugin, board, pad, desc = suite._make_board_and_plugin()
        updates = []
        pcg_seen = threading.Event()
        eta_seen = threading.Event()

        class RecordingProgressDialog:
            def __init__(self, title, message, maximum=100, **kwargs):
                updates.append((0, message))

            def Update(self, value, newmsg=""):
                updates.append((int(value), newmsg))
                if "PCG iter 24" in newmsg:
                    pcg_seen.set()
                if "step 4/10 (" in newmsg and "remaining)" in newmsg:
                    eta_seen.set()
                return (True, False)

            def WasCancelled(self):
                return False

            def Hide(self):
                pass

            def Destroy(self):
                pass

        def steady_pcg_solver(config, K, C, Q, b, hA, layer_count, rows, cols,
                              progress_cb, snap_cb, **kwargs):
            progress_cb(
                45,
                100,
                "Solving steady-state (PCG iter 24, residual 4.2e-5 -> 1e-7)...",
            )
            pcg_seen.wait(timeout=1.0)
            progress_cb(
                100,
                100,
                "Solving steady-state (PCG iter 48, residual 8.1e-8 -> 1e-7)...",
            )
            return SolverResult(
                T=np.full((layer_count, rows, cols), 42.0),
                aborted=False,
                step_counter=48,
                total_solve_time=0.08,
                total_factor_time=0.01,
                factor_count=1,
                snapshot_stats=[],
                snapshot_files=[],
                phase_metrics=[],
                balance_history=[],
                k_norm_info={},
            )

        monkeypatch.setattr(plugin_mod, "run_simulation", steady_pcg_solver)
        monkeypatch.setattr(plugin_mod.wx, "ProgressDialog", RecordingProgressDialog, raising=False)
        monkeypatch.setattr(plugin_mod.wx, "GetApp", lambda: None, raising=False)
        monkeypatch.setattr(plugin_mod, "show_results_all_layers", lambda *a, **k: str(tmp_path / "hm.png"))
        monkeypatch.setattr(plugin_mod, "save_preview_image", lambda *a, **k: str(tmp_path / "prev.png"))
        monkeypatch.setattr(plugin_mod, "build_interactive_heatmap_payload", lambda *a, **k: {})
        monkeypatch.setattr(plugin_mod, "write_html_report", lambda *a, **k: str(tmp_path / "rep.html"))
        monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)

        steady_settings = suite._make_settings(tmp_path, desc)
        steady_settings["steady_state"] = True
        plugin._run_simulation(
            board, [F_Cu, B_Cu], ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad], steady_settings, {}, ["U1-1"],
        )

        assert pcg_seen.is_set()
        assert any(
            "Solving steady-state (PCG iter 24, residual 4.2e-5 -> 1e-7)..." in msg
            for _, msg in updates
        )

        # Transient run appends English ETA once cur >= 3, cur < tot, and elapsed >= 1.5s
        updates.clear()
        orig_monotonic = time.monotonic
        fake_clock = {"offset": 0.0}
        monkeypatch.setattr(plugin_mod.time, "monotonic", lambda: orig_monotonic() + fake_clock["offset"])

        def transient_solver(config, K, C, Q, b, hA, layer_count, rows, cols,
                             progress_cb, snap_cb, **kwargs):
            progress_cb(1, 10)
            fake_clock["offset"] = 2.0
            progress_cb(4, 10)
            eta_seen.wait(timeout=1.0)
            progress_cb(10, 10)
            return SolverResult(
                T=np.full((layer_count, rows, cols), 42.0),
                aborted=False,
                step_counter=10,
                total_solve_time=0.08,
                total_factor_time=0.01,
                factor_count=1,
                snapshot_stats=[],
                snapshot_files=[],
                phase_metrics=[],
                balance_history=[],
                k_norm_info={},
            )

        monkeypatch.setattr(plugin_mod, "run_simulation", transient_solver)
        transient_settings = suite._make_settings(tmp_path, desc)
        transient_settings["steady_state"] = False
        plugin._run_simulation(
            board, [F_Cu, B_Cu], ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad], transient_settings, {}, ["U1-1"],
        )

        assert eta_seen.is_set()
        assert any(
            "Solving thermal model - step 4/10 (~3s remaining)" in msg
            for _, msg in updates
        )


class TestSimulationCoreDeduplication:
    """Verify simulation_core single-source-of-truth exports and auto time_stepping wiring."""

    def test_simulation_core_exports_and_dead_helper_removed(self):
        import ThermalSim.headless_api as headless_mod
        import ThermalSim.simulation_core as core_mod
        import ThermalSim.thermal_plugin as plugin_mod

        for name in (
            "GRID_DETAIL_PRESETS",
            "DEFAULT_GRID_DETAIL",
            "DEFAULT_GRID_NODE_BUDGET",
            "auto_fast_engine_threshold",
            "resolve_mesh_mode",
            "resolve_grid_policy",
            "coarsen_grid_resolution",
            "estimate_solver_cost",
            "bbox_bounds_mm",
            "calculate_suggested_resolution",
            "estimate_simulation_area",
            "estimate_solver_grid",
            "derive_stackup_thicknesses",
            "effective_fr4_control_volume_thicknesses",
            "build_dense_capacity",
            "build_compact_heatsink_model",
            "SparsePadContribution",
            "pad_target_layer_index",
            "build_sparse_pad_contributions",
            "build_power_vector",
            "build_sparse_power_vector",
        ):
            assert hasattr(core_mod, name), f"Missing {name} in simulation_core"

        assert plugin_mod.GRID_DETAIL_PRESETS is core_mod.GRID_DETAIL_PRESETS
        assert headless_mod.GRID_DETAIL_PRESETS is core_mod.GRID_DETAIL_PRESETS
        assert plugin_mod.SparsePadContribution is core_mod.SparsePadContribution
        assert headless_mod._SparsePadContribution is core_mod.SparsePadContribution
        assert not hasattr(plugin_mod, "_bbox_to_power_indices")
        assert not hasattr(headless_mod, "_bbox_to_power_indices")

    def test_fast_engine_preserves_auto_time_stepping(self, tmp_path, monkeypatch):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_solver import SolverResult

        suite = TestCompactLowRamPluginWiring()
        plugin, board, pad, settings = suite._setup_plugin_env(tmp_path, monkeypatch, with_eco1=False)
        from ThermalSim.workflow import GeometryCache
        plugin.geometry_cache = GeometryCache(cache_dir=str(tmp_path / "geom_cache"))
        settings["compute_engine"] = "fast_native"
        settings["mesh_mode"] = "uniform"
        settings["steady_state"] = False

        recorded = []

        def spy_mf(config, op, C, Q, b, hA, *args, **kwargs):
            recorded.append(config.time_stepping)
            return SolverResult(
                T=np.full((op.layer_count, op.rows, op.cols), 35.0),
                k_norm_info={},
                step_counter=1,
                total_solve_time=0.01,
                total_factor_time=0.0,
                factor_count=0,
            )

        monkeypatch.setattr(plugin_mod, "run_simulation_matrix_free", spy_mf)

        plugin._run_simulation(
            board,
            [F_Cu, B_Cu],
            ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad],
            settings,
            {},
            ["U1-1"],
        )
        assert recorded == ["auto"]

        settings["time_stepping"] = "uniform"
        plugin._run_simulation(
            board,
            [F_Cu, B_Cu],
            ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad],
            settings,
            {},
            ["U1-1"],
        )
        assert recorded == ["auto", "uniform"]


class TestV048PluginWiringAndSnapshot:
    """Tests for v0.4.8 snapshot invalidation, operator cache key, and 3D via routing."""

    def test_snapshot_invalidates_on_via_drill_layers_and_pad_drill(self):
        from ThermalSim.thermal_plugin import ThermalPlugin

        class _DrillVia(MockTrack):
            def __init__(self, drill_nm, layers):
                super().__init__(layer=F_Cu, bbox=EDA_RECT(2_000_000, 2_000_000, 600_000, 600_000))
                self._drill_nm = drill_nm
                self._layers = list(layers)

            def GetWidth(self, layer=None):
                return 600_000

            def GetDrillValue(self):
                return self._drill_nm

            def GetLayerSet(self):
                return _PluginShapeLayerSet(self._layers)

        class _DrillPad(MockPad):
            def __init__(self, drill_nm):
                super().__init__(
                    position=VECTOR2I(5_000_000, 5_000_000),
                    layer=F_Cu,
                    attribute=PAD_ATTRIB_PTH,
                )
                self._drill_nm = drill_nm

            def GetDrillSize(self):
                return VECTOR2I(self._drill_nm, self._drill_nm)

        plugin = ThermalPlugin()
        plugin.defaults()
        bbox = EDA_RECT(0, 0, 20_000_000, 20_000_000)

        snap_1 = plugin._capture_board_snapshot(
            MockBoard(tracks=[_DrillVia(300_000, [F_Cu, B_Cu])]), [F_Cu, B_Cu], bbox
        )
        snap_2 = plugin._capture_board_snapshot(
            MockBoard(tracks=[_DrillVia(450_000, [F_Cu, B_Cu])]), [F_Cu, B_Cu], bbox
        )
        snap_3 = plugin._capture_board_snapshot(
            MockBoard(tracks=[_DrillVia(300_000, [F_Cu])]), [F_Cu, B_Cu], bbox
        )
        assert snap_1.fingerprint != snap_2.fingerprint
        assert snap_1.fingerprint != snap_3.fingerprint

        pad_board_a = MockBoard(footprints=[MockFootprint(reference="J1", pads=[_DrillPad(800_000)])])
        pad_board_b = MockBoard(footprints=[MockFootprint(reference="J1", pads=[_DrillPad(1_200_000)])])
        snap_pa = plugin._capture_board_snapshot(pad_board_a, [F_Cu, B_Cu], bbox)
        snap_pb = plugin._capture_board_snapshot(pad_board_b, [F_Cu, B_Cu], bbox)
        assert snap_pa.fingerprint != snap_pb.fingerprint

    def test_operator_cache_key_includes_h_conv_bottom(self):
        from ThermalSim.thermal_plugin import _build_operator_cache_key

        k1 = _build_operator_cache_key(
            "geom", "legacy", [35e-6, 35e-6], [1.5e-3], [0.75e-3, 0.75e-3], 0.5e-3, 25.0,
            {"h_conv": 10.0, "h_conv_bottom": 10.0},
        )
        k2 = _build_operator_cache_key(
            "geom", "legacy", [35e-6, 35e-6], [1.5e-3], [0.75e-3, 0.75e-3], 0.5e-3, 25.0,
            {"h_conv": 10.0, "h_conv_bottom": 2.0},
        )
        assert k1 != k2

    def test_run_safe_forwards_stackup_thickness_to_dialog(self, monkeypatch):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_plugin import ThermalPlugin

        plugin = ThermalPlugin()
        plugin.defaults()
        captured_kwargs = {}

        class DummyDialog:
            def __init__(self, *args, **kwargs):
                captured_kwargs.update(kwargs)

            def Show(self):
                pass

        board = MockBoard(footprints=[MockFootprint(reference="U1", pads=[])])
        monkeypatch.setattr(plugin_mod.pcbnew, "GetBoard", lambda: board, raising=False)
        monkeypatch.setattr(plugin_mod, "_find_pcb_editor_parent", lambda _b=None: None)
        monkeypatch.setattr(plugin_mod, "SettingsDialog", DummyDialog)
        monkeypatch.setattr(
            plugin_mod,
            "parse_stackup_from_board_file",
            lambda *a, **k: {"board_thickness_mm": 0.8},
        )
        monkeypatch.setattr(plugin, "_show_startup_progress", lambda *a, **k: None)
        monkeypatch.setattr(plugin, "_close_startup_progress", lambda *a, **k: None)
        monkeypatch.setattr(plugin, "_load_settings", lambda *a, **k: {})
        monkeypatch.setattr(plugin, "_detect_copper_layers", lambda _b: ([F_Cu, B_Cu], ["F.Cu", "B.Cu"]))
        monkeypatch.setattr(board, "GetBoundingBox", lambda: EDA_RECT(0, 0, 10_000_000, 10_000_000), raising=False)

        plugin.Run()
        assert captured_kwargs["stackup_thickness_mm"] == pytest.approx(0.8)

    def test_run_simulation_via_map_3d_uses_structured_operator_and_records_k_norm(
        self, tmp_path, monkeypatch
    ):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.geometry_mapper import GeometryState
        from ThermalSim.thermal_solver import SolverResult

        suite = TestCompactLowRamPluginWiring()
        plugin, board, pad, settings = suite._setup_plugin_env(tmp_path, monkeypatch, with_eco1=False)
        from ThermalSim.workflow import GeometryCache
        plugin.geometry_cache = GeometryCache(cache_dir=str(tmp_path / "geom_cache"))
        settings["compute_engine"] = "fast_native"
        settings["h_conv"] = 14.0
        settings["h_conv_bottom"] = 1.5
        settings["via_plating_um"] = 30.0
        settings["via_fill"] = "copper"

        def make_3d_state(*args, **kwargs):
            rows = int(kwargs.get("rows", 24))
            cols = int(kwargs.get("cols", 24))
            copper_mask = np.zeros((2, rows, cols), dtype=bool)
            copper_mask[0, :, :] = True
            via_map_2d = np.ones((rows, cols), dtype=np.float32)
            via_map_3d = np.ones((1, rows, cols), dtype=np.float32)
            via_map_3d[0, 3, 3] = 25.0
            state = GeometryState(
                copper_mask=copper_mask,
                via_map=via_map_2d,
                heatsink_mask=np.zeros((rows, cols), dtype=bool),
                area_mask=None,
                x_min=float(kwargs.get("x_min", 0.0)),
                y_min=float(kwargs.get("y_min", 0.0)),
                res=float(kwargs.get("res", 0.5)),
                rows=rows,
                cols=cols,
                x_centers_iu=np.zeros(cols, dtype=np.int64),
                y_centers_iu=np.zeros(rows, dtype=np.int64),
                via_map_3d=via_map_3d,
            )
            captured_state["via_map_3d"] = via_map_3d
            return state

        captured_state = {}
        monkeypatch.setattr(plugin_mod, "build_geometry_state", make_3d_state)

        calls = {"structured": 0, "compact": 0}

        def spy_structured(*args, **kwargs):
            calls["structured"] += 1
            assert args[10] is captured_state["via_map_3d"]
            n_nodes = int(args[0]) * int(args[1]) * int(args[2])
            return ("K_STRUCT", np.zeros(n_nodes), np.zeros(n_nodes), {})

        def spy_compact(*args, **kwargs):
            calls["compact"] += 1
            raise AssertionError("Compact operator must not be used when via_map_3d is present")

        monkeypatch.setattr(plugin_mod, "build_structured_operator", spy_structured)
        monkeypatch.setattr(plugin_mod, "_ORIG_BUILD_STRUCTURED_OPERATOR", spy_structured)
        monkeypatch.setattr(plugin_mod, "build_compact_structured_operator", spy_compact)
        monkeypatch.setattr(
            plugin_mod,
            "run_simulation_matrix_free",
            lambda config, op, C, Q, b, hA, *a, **k: SolverResult(
                T=np.full((2, int(np.sqrt(C.size // 2)), int(np.sqrt(C.size // 2))), 30.0),
                k_norm_info={},
                step_counter=1,
                total_solve_time=0.01,
                total_factor_time=0.0,
                factor_count=0,
            ),
        )
        captured_report = {}
        monkeypatch.setattr(
            plugin_mod,
            "write_html_report",
            lambda *a, **k: captured_report.update(k) or str(tmp_path / "rep.html"),
        )

        plugin._run_simulation(
            board, [F_Cu, B_Cu], ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad], settings, {}, ["U1-1"],
        )
        assert calls["structured"] == 1
        assert calls["compact"] == 0
        k_norm = captured_report["k_norm_info"]
        assert k_norm["h_top"] == pytest.approx(14.0)
        assert k_norm["h_air_bottom"] == pytest.approx(1.5)
        assert k_norm["via_plating_um"] == pytest.approx(30.0)
        assert k_norm["via_fill"] == "copper"

    def test_preview_forwards_via_settings_and_handles_3d_via_map(self, tmp_path):
        from ThermalSim.geometry_mapper import GeometryState
        from ThermalSim.visualization import save_preview_image

        rows, cols = 16, 16
        state = GeometryState(
            copper_mask=np.ones((4, rows, cols), dtype=bool),
            via_map=np.ones((rows, cols), dtype=np.float32),
            heatsink_mask=np.zeros((rows, cols), dtype=bool),
            area_mask=None,
            x_min=0.0,
            y_min=0.0,
            res=1.0,
            rows=rows,
            cols=cols,
            x_centers_iu=np.zeros(cols, dtype=np.int64),
            y_centers_iu=np.zeros(rows, dtype=np.int64),
            via_map_3d=np.ones((3, rows, cols), dtype=np.float32),
        )
        state.via_map_3d[0, 2, 2] = 15.0
        board = MockBoard(footprints=[MockFootprint(reference="U1", pads=[])])
        bbox = EDA_RECT(0, 0, 12_000_000, 12_000_000)
        out = save_preview_image(
            board,
            [F_Cu, 1, 2, B_Cu],
            bbox,
            [],
            {"res": 1.0, "use_heatsink": False},
            ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"],
            {},
            get_pad_pixels,
            None,
            lambda *a, **k: {"copper_thickness_mm_used": [0.035] * 4},
            open_file=False,
            out_dir=str(tmp_path),
            geometry_state=state,
        )
        assert out is not None and out.endswith("thermal_preview.png")

        captured_extra = {}

        def fake_create_maps(
            board, copper_ids, rows, cols, x_min, y_min, res,
            settings, k_fr4_rel, k_cu_layers, via_factor, pads_list,
            via_plating_um=25.0, via_fill="plated", k_cu=390.0, k_fr4=0.3,
        ):
            captured_extra["via_plating_um"] = via_plating_um
            captured_extra["via_fill"] = via_fill
            return (
                np.ones((2, rows, cols), dtype=np.float64),
                np.ones((rows, cols), dtype=np.float32),
                np.zeros((rows, cols), dtype=np.float64),
            )

        out2 = save_preview_image(
            board,
            [F_Cu, B_Cu],
            bbox,
            [],
            {"res": 1.0, "use_heatsink": False, "via_plating_um": 40.0, "via_fill": "copper"},
            ["F.Cu", "B.Cu"],
            {},
            get_pad_pixels,
            fake_create_maps,
            lambda *a, **k: {"copper_thickness_mm_used": [0.035, 0.035]},
            open_file=False,
            out_dir=str(tmp_path),
        )
        assert out2 is not None
        assert captured_extra == {"via_plating_um": 40.0, "via_fill": "copper"}


class TestV049PluginElectricalGapAndCleanup:
    """Tests for v0.4.9 ElectricalConfig gap_m wiring, cache key invalidation, and cleanups."""

    def test_run_simulation_passes_gap_m_and_invalidates_electrical_cache(self, tmp_path, monkeypatch):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.electrical_solver import ElectricalResult
        from ThermalSim.thermal_solver import SolverResult
        from ThermalSim.workflow import GeometryCache

        suite = TestCompactLowRamPluginWiring()
        plugin, board, pad, settings = suite._setup_plugin_env(tmp_path, monkeypatch, with_eco1=False)
        plugin.geometry_cache = GeometryCache(cache_dir=str(tmp_path / "geom_cache"))

        # Add a second pad on the same net so current path is balanced
        pad2 = MockPad(
            position=VECTOR2I(7_000_000, 5_000_000),
            layer=F_Cu,
            bbox=EDA_RECT(6_500_000, 4_500_000, 1_000_000, 1_000_000),
            net_code=1,
            net_name="VIN",
            number="2",
        )
        board.Footprints()[0].Pads().append(pad2)
        desc1 = plugin._pad_descriptor(board, "U1", pad)
        desc2 = plugin._pad_descriptor(board, "U1", pad2)

        settings["compute_engine"] = "legacy"
        settings["steady_state"] = True
        settings["current_enabled"] = True
        settings["current_groups"] = [
            {
                "name": "VIN",
                "mode": "per_pad",
                "pads": [
                    {**desc1, "current_a": 2.0},
                    {**desc2, "current_a": -2.0},
                ],
            }
        ]

        captured_configs = []
        captured_report = {}

        def spy_solve_electrical(b, terminals, config, **kwargs):
            captured_configs.append(config)
            n = len(config.copper_ids) * config.rows * config.cols
            return ElectricalResult(
                q_joule=np.zeros(n, dtype=np.float64),
                net_summaries=[],
                warnings=[],
                errors=[],
            )

        monkeypatch.setattr(plugin_mod, "solve_electrical_heating", spy_solve_electrical)
        monkeypatch.setattr(
            plugin_mod,
            "run_simulation",
            lambda config, K, C, Q, b, hA, lc, r, c, *a, **k: SolverResult(
                T=np.full((lc, r, c), 25.0),
                aborted=False,
                step_counter=1,
                total_solve_time=0.01,
                total_factor_time=0.0,
                factor_count=1,
                snapshot_stats=[],
                snapshot_files=[],
                phase_metrics=[],
                balance_history=[],
                k_norm_info={},
            ),
        )
        monkeypatch.setattr(
            plugin_mod,
            "write_html_report",
            lambda *a, **k: captured_report.update(k) or str(tmp_path / "rep.html"),
        )

        # First run with thick=0.8 mm
        settings_a = dict(settings, thick=0.8)
        plugin._run_simulation(
            board, [F_Cu, B_Cu], ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad, pad2], settings_a, {}, ["U1-1", "U1-2"],
        )
        assert len(captured_configs) == 1
        assert captured_configs[0].gap_m is not None
        assert captured_configs[0].gap_m[0] == pytest.approx(0.8e-3, rel=1e-4)
        first_key = plugin.electrical_cache.key
        assert "init_adaptive_mesh_s" not in captured_report["k_norm_info"]

        # Second run with thick=1.6 mm must miss electrical_cache and pass updated gap_m
        settings_b = dict(settings, thick=1.6)
        plugin._run_simulation(
            board, [F_Cu, B_Cu], ["F.Cu", "B.Cu"],
            EDA_RECT(0, 0, 10_000_000, 10_000_000),
            [pad, pad2], settings_b, {}, ["U1-1", "U1-2"],
        )
        assert len(captured_configs) == 2
        assert captured_configs[1].gap_m[0] == pytest.approx(1.6e-3, rel=1e-4)
        assert plugin.electrical_cache.key != first_key
        assert captured_report["k_norm_info"]["init_electrical_cache_hit"] is False

    def test_preflight_10m_node_warning_has_no_stale_adaptive_text(self, tmp_path, monkeypatch):
        import ThermalSim.thermal_plugin as plugin_mod
        from ThermalSim.thermal_plugin import ThermalPlugin
        from ThermalSim.workflow import GridEstimate

        plugin = ThermalPlugin()
        plugin.defaults()
        pad = MockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            layer=F_Cu,
            bbox=EDA_RECT(4_500_000, 4_500_000, 1_000_000, 1_000_000),
            net_code=1,
            net_name="VIN",
            number="1",
        )
        board = MockBoard(footprints=[MockFootprint("U1", [pad])])
        desc = plugin._pad_descriptor(board, "U1", pad)
        huge_grid = GridEstimate(
            requested_res_mm=0.1,
            actual_res_mm=0.1,
            x_min_mm=0.0,
            y_min_mm=0.0,
            width_mm=200.0,
            height_mm=250.0,
            rows=2000,
            cols=2500,
            layer_count=4,
            auto_coarsened=False,
            expert_limits=False,
            max_cells=20_000_000,
            target_cells=17_000_000,
        )
        monkeypatch.setattr(plugin_mod, "_estimate_solver_grid", lambda *a, **k: huge_grid)

        res = plugin._preflight(
            board,
            [F_Cu, B_Cu],
            EDA_RECT(0, 0, 200_000_000, 250_000_000),
            {
                "res": 0.1,
                "output_dir": str(tmp_path),
                "power_pads": [{**desc, "power": "1.0"}],
            },
        )
        assert any("uniform nodes" in w for w in res.warnings)
        assert all("Adaptive reduction depends" not in w for w in res.warnings)
