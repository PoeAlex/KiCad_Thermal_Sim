"""
Unit tests for thermal_plugin helper functions.

These tests focus on initialization-time helper logic that was refactored
for performance, while keeping numerical behavior unchanged.
"""

import numpy as np

from ThermalSim.geometry_mapper import get_pad_pixels
from ThermalSim.thermal_plugin import (
    _build_power_vector,
    _build_sparse_pad_contributions,
)
from tests.mocks.pcbnew_mock import (
    MockBoard,
    MockFootprint,
    MockPad,
    VECTOR2I,
    EDA_RECT,
    F_Cu,
    B_Cu,
)


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
