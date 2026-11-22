"""
Unit tests for thermal_plugin helper functions.

These tests focus on initialization-time helper logic that was refactored
for performance, while keeping numerical behavior unchanged.
"""

import json

import numpy as np

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
        """Fast/Balanced/Detailed/Very detailed presets use 500k/1.2M/3M/6M with 85% target."""
        expected = {
            "fast": (500_000, 425_000),
            "balanced": (1_200_000, 1_020_000),
            "detailed": (3_000_000, 2_550_000),
            "very_detailed": (6_000_000, 5_100_000),
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
        """Auto engine threshold and Auto mesh mode follow HAS_NATIVE_MG."""
        import ThermalSim.thermal_plugin as plugin_mod

        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_MG", True)
        assert plugin_mod._auto_fast_engine_threshold() == 50_000
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "auto"}, True, 1_000_000) == "uniform"
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "auto"}, True, 5_000_000) == "uniform"
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "auto"}, True, 5_000_001) == "adaptive"
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "adaptive"}, True, 100_000) == "adaptive"
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "uniform"}, True, 10_000_000) == "uniform"
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "adaptive"}, False, 100_000) == "uniform"

        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_MG", False)
        assert plugin_mod._auto_fast_engine_threshold() == 250_000
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "auto"}, True, 1_000_000) == "adaptive"
        assert plugin_mod._resolve_mesh_mode({"mesh_mode": "uniform"}, True, 1_000_000) == "uniform"

    def test_estimate_solver_cost_reflects_native_mg_and_ignores_legacy_backend(self, monkeypatch):
        """Native MG cost model uses calibrated RAM/runtime thresholds and ignores legacy backend."""
        import ThermalSim.thermal_plugin as plugin_mod

        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_MG", True)
        low, high, runtime = plugin_mod._estimate_solver_cost(
            3_000_000, {"compute_engine": "auto"}
        )
        assert low == 750
        assert high == 1500
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
        assert c_plain == c_legacy

        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_MG", False)
        low_no_mg, high_no_mg, rt_no_mg = plugin_mod._estimate_solver_cost(
            1_000_000, {"compute_engine": "fast_native"}
        )
        assert low_no_mg == 120
        assert high_no_mg == 450
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

    def test_generate_preview_respects_resolved_auto_mesh_mode(self, tmp_path, monkeypatch):
        """Preview should only build adaptive overlay when resolved mesh mode is adaptive."""
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

        adaptive_calls = []
        preview_meshes = []

        def fake_build_adaptive_mesh(*args, **kwargs):
            adaptive_calls.append(True)
            return "fake-adaptive-mesh"

        def fake_save_preview_image(*args, **kwargs):
            preview_meshes.append(kwargs.get("adaptive_mesh"))
            return str(tmp_path / "preview.png")

        monkeypatch.setattr(plugin_mod, "build_adaptive_mesh", fake_build_adaptive_mesh)
        monkeypatch.setattr(plugin_mod, "save_preview_image", fake_save_preview_image)
        monkeypatch.setattr(plugin_mod, "HAS_NATIVE_MG", True)

        # Auto mesh mode with native MG and small grid resolves to uniform -> no adaptive overlay
        plugin.generate_preview(
            {
                "res": 2.0,
                "compute_engine": "fast_native",
                "mesh_mode": "auto",
                "output_dir": str(tmp_path),
            },
            ["F.Cu", "B.Cu"],
        )
        assert adaptive_calls == []
        assert preview_meshes == [None]

        # Explicit adaptive mesh mode with fast engine builds adaptive overlay
        plugin.generate_preview(
            {
                "res": 2.0,
                "compute_engine": "fast_native",
                "mesh_mode": "adaptive",
                "output_dir": str(tmp_path),
            },
            ["F.Cu", "B.Cu"],
        )
        assert len(adaptive_calls) == 1
        assert preview_meshes == [None, "fake-adaptive-mesh"]

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
            n_nodes = solver_cap.shape[0]
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
        monkeypatch.setattr(plugin_mod, "restrict_heatsink_model", fake_restrict_heatsink_model)
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

    def test_heatsink_model_wiring_uniform_adaptive_legacy_and_metadata(self, tmp_path, monkeypatch):
        # Uniform matrix-free passes full heatsink_model without restricting
        rec_uniform = self._run_plugin_with_heatsink(
            tmp_path, monkeypatch, {"compute_engine": "fast_native", "mesh_mode": "uniform"}
        )
        assert len(rec_uniform["build_calls"]) == 1
        assert rec_uniform["restrict_calls"] == []
        assert rec_uniform["mf_kwargs"][0]["heatsink_model"]["kind"] == "full_heatsink_model"

        # Adaptive matrix-free restricts heatsink_model
        rec_adaptive = self._run_plugin_with_heatsink(
            tmp_path, monkeypatch, {"compute_engine": "fast_native", "mesh_mode": "adaptive"}
        )
        assert len(rec_adaptive["build_calls"]) == 1
        assert len(rec_adaptive["restrict_calls"]) == 1
        assert rec_adaptive["mf_kwargs"][0]["heatsink_model"]["kind"] == "restricted_heatsink_model"

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
