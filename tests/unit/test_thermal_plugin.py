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
    MockFootprint,
    MockPad,
    MockZone,
    VECTOR2I,
    EDA_RECT,
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
        assert target_cells == 12_500_000

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
        assert target_cells == 25_000_000

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
