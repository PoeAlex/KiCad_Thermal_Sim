"""
Unit tests for geometry_mapper module.

This module tests the PCB geometry to thermal grid mapping functions.
"""

import math
import pytest
import numpy as np

from ThermalSim.geometry_mapper import (
    FillContext,
    GeometryCancelledError,
    build_geometry_state,
    _bbox_to_grid_indices,
    _fill_box,
    _fill_via,
    _fill_heatsink,
    _fill_zone_mask_polygons,
    _scanline_fill_polygon,
    build_pad_distance_mask,
    get_pad_copper_layer_indices,
    get_pad_grid_mask,
    get_pad_pixels,
    get_pad_power_indices,
    get_pad_primary_layer_id,
    create_multilayer_maps,
)

from tests.mocks.pcbnew_mock import (
    MockBoard, MockLayerSet, MockPad, MockZone, MockFootprint, MockTrack, MockVia,
    VECTOR2I, EDA_RECT, F_Cu, B_Cu, In1_Cu, In2_Cu,
    PAD_ATTRIB_NPTH, PAD_ATTRIB_PTH, PAD_ATTRIB_SMD, Eco1_User,
    MockDrawing,
)


class TestFillContext:
    """Tests for FillContext dataclass."""

    def test_initialization(self):
        """Test FillContext initialization."""
        K = np.ones((2, 10, 10))
        V = np.ones((10, 10))
        H = np.zeros((10, 10))

        ctx = FillContext(
            K=K, V=V, H=H,
            area_mask=None,
            x_min=0.0, y_min=0.0,
            res=0.5,
            rows=10, cols=10
        )

        assert ctx.rows == 10
        assert ctx.cols == 10
        assert ctx.res == 0.5
        assert ctx.area_mask is None

    def test_with_area_mask(self):
        """Test FillContext with area mask."""
        K = np.ones((2, 10, 10))
        V = np.ones((10, 10))
        H = np.zeros((10, 10))
        mask = np.ones((10, 10), dtype=bool)
        mask[5:, :] = False  # Mask out bottom half

        ctx = FillContext(
            K=K, V=V, H=H,
            area_mask=mask,
            x_min=0.0, y_min=0.0,
            res=0.5,
            rows=10, cols=10
        )

        assert ctx.area_mask is not None
        assert ctx.area_mask.shape == (10, 10)


class TestBboxToGridIndices:
    """Tests for _bbox_to_grid_indices helper function."""

    @pytest.fixture
    def context(self):
        """Create test context."""
        K = np.ones((2, 20, 20))
        V = np.ones((20, 20))
        H = np.zeros((20, 20))
        return FillContext(
            K=K, V=V, H=H,
            area_mask=None,
            x_min=0.0, y_min=0.0,
            res=0.5,  # 0.5 mm resolution
            rows=20, cols=20
        )

    def test_basic_conversion(self, context):
        """Test basic bounding box to grid index conversion."""
        # 1mm x 1mm box starting at (1mm, 1mm)
        bbox = EDA_RECT(1000000, 1000000, 1000000, 1000000)  # nm units

        rs, re, cs, ce = _bbox_to_grid_indices(bbox, context)

        # At 0.5mm resolution, 1mm = 2 grid cells
        assert rs == 2  # y_min=1mm / 0.5mm
        assert cs == 2  # x_min=1mm / 0.5mm
        assert re >= rs  # Row end > row start
        assert ce >= cs  # Col end > col start

    def test_at_origin(self, context):
        """Test bounding box at origin."""
        bbox = EDA_RECT(0, 0, 500000, 500000)  # 0.5mm x 0.5mm

        rs, re, cs, ce = _bbox_to_grid_indices(bbox, context)

        assert rs == 0
        assert cs == 0
        assert re >= 1
        assert ce >= 1

    def test_clamping_to_grid(self, context):
        """Test that indices are clamped to grid bounds."""
        # Box extending beyond grid (grid is 20x20 at 0.5mm = 10mm x 10mm)
        bbox = EDA_RECT(9000000, 9000000, 5000000, 5000000)  # Starts at 9mm, extends to 14mm

        rs, re, cs, ce = _bbox_to_grid_indices(bbox, context)

        # Should be clamped to grid size
        assert re <= context.rows
        assert ce <= context.cols

    def test_negative_start_clamped(self, context):
        """Test that negative indices are clamped to zero."""
        # Box starting at negative coordinates
        bbox = EDA_RECT(-1000000, -1000000, 2000000, 2000000)

        rs, re, cs, ce = _bbox_to_grid_indices(bbox, context)

        assert rs >= 0
        assert cs >= 0


class TestFillBox:
    """Tests for _fill_box function."""

    @pytest.fixture
    def context(self):
        """Create test context."""
        K = np.ones((2, 20, 20))  # Start with 1.0 everywhere
        V = np.ones((20, 20))
        H = np.zeros((20, 20))
        return FillContext(
            K=K, V=V, H=H,
            area_mask=None,
            x_min=0.0, y_min=0.0,
            res=0.5,
            rows=20, cols=20
        )

    def test_fills_region(self, context):
        """Test that region is filled with value."""
        bbox = EDA_RECT(1000000, 1000000, 2000000, 2000000)  # 1mm to 3mm
        val = 400.0  # Copper conductivity

        _fill_box(context, 0, bbox, val)

        # Check that region has been filled
        assert np.max(context.K[0]) == 400.0

    def test_uses_maximum(self, context):
        """Test that fill uses maximum with existing value."""
        # Pre-fill a region with 200
        context.K[0, 5:10, 5:10] = 200.0

        # Now fill overlapping region with 400
        bbox = EDA_RECT(2000000, 2000000, 3000000, 3000000)  # Overlaps with prefilled
        _fill_box(context, 0, bbox, 400.0)

        # Should have 400 where filled, original values preserved elsewhere
        assert np.max(context.K[0, 5:10, 5:10]) >= 200.0

    def test_respects_area_mask(self):
        """Test that area mask is respected."""
        K = np.ones((2, 20, 20))
        V = np.ones((20, 20))
        H = np.zeros((20, 20))
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True  # Only center region is active

        context = FillContext(
            K=K, V=V, H=H,
            area_mask=mask,
            x_min=0.0, y_min=0.0,
            res=0.5,
            rows=20, cols=20
        )

        # Fill entire grid
        bbox = EDA_RECT(0, 0, 10000000, 10000000)
        _fill_box(context, 0, bbox, 400.0)

        # Only masked region should be filled
        assert np.max(context.K[0, 0:5, :]) == 1.0  # Outside mask
        assert np.max(context.K[0, 5:15, 5:15]) == 400.0  # Inside mask


class TestFillVia:
    """Tests for _fill_via function."""

    @pytest.fixture
    def context(self):
        """Create test context."""
        K = np.ones((2, 20, 20))
        V = np.ones((20, 20))
        H = np.zeros((20, 20))
        return FillContext(
            K=K, V=V, H=H,
            area_mask=None,
            x_min=0.0, y_min=0.0,
            res=0.5,
            rows=20, cols=20
        )

    def test_fills_via_map(self, context):
        """Test that via enhancement is applied."""
        bbox = EDA_RECT(2500000, 2500000, 500000, 500000)  # Small via
        via_factor = 50.0

        _fill_via(context, bbox, via_factor)

        # V map should have enhanced values
        assert np.max(context.V) == 50.0

    def test_via_uses_maximum(self, context):
        """Test that via fill uses maximum."""
        # Pre-fill with lower value
        context.V[5:10, 5:10] = 25.0

        bbox = EDA_RECT(2000000, 2000000, 3000000, 3000000)
        _fill_via(context, bbox, 50.0)

        # Should use maximum
        assert np.max(context.V) >= 50.0


class TestFillHeatsink:
    """Tests for _fill_heatsink function."""

    @pytest.fixture
    def context(self):
        """Create test context."""
        K = np.ones((2, 20, 20))
        V = np.ones((20, 20))
        H = np.zeros((20, 20))
        return FillContext(
            K=K, V=V, H=H,
            area_mask=None,
            x_min=0.0, y_min=0.0,
            res=0.5,
            rows=20, cols=20
        )

    def test_marks_heatsink_region(self, context):
        """Test that heatsink region is marked."""
        bbox = EDA_RECT(2000000, 2000000, 3000000, 3000000)

        _fill_heatsink(context, bbox)

        # H map should have 1.0 in region
        assert np.max(context.H) == 1.0
        assert np.sum(context.H > 0) > 0


class TestBuildPadDistanceMask:
    """Tests for build_pad_distance_mask function."""

    def test_returns_none_for_empty_pads(self):
        """Test returns None when no pads provided."""
        result = build_pad_distance_mask([], 50, 50, 0.0, 0.0, 0.5, 10.0)
        assert result is None

    def test_returns_none_for_zero_radius(self):
        """Test returns None when radius is zero."""
        pad = MockPad(position=VECTOR2I(5000000, 5000000))
        result = build_pad_distance_mask([pad], 50, 50, 0.0, 0.0, 0.5, 0.0)
        assert result is None

    def test_returns_none_for_none_radius(self):
        """Test returns None when radius is None."""
        pad = MockPad(position=VECTOR2I(5000000, 5000000))
        result = build_pad_distance_mask([pad], 50, 50, 0.0, 0.0, 0.5, None)
        assert result is None

    def test_creates_circular_mask(self):
        """Test that mask is approximately circular around pad."""
        # Pad at center of 25mm x 25mm grid (50x50 at 0.5mm)
        pad = MockPad(position=VECTOR2I(12500000, 12500000))  # Center
        radius_mm = 5.0
        res = 0.5

        result = build_pad_distance_mask([pad], 50, 50, 0.0, 0.0, res, radius_mm)

        assert result is not None
        assert result.shape == (50, 50)
        # Center should be True
        assert result[25, 25] == True
        # Far corners should be False
        assert result[0, 0] == False
        assert result[49, 49] == False

    def test_multiple_pads(self):
        """Test mask with multiple pads."""
        pad1 = MockPad(position=VECTOR2I(5000000, 5000000))
        pad2 = MockPad(position=VECTOR2I(20000000, 20000000))
        radius_mm = 3.0

        result = build_pad_distance_mask([pad1, pad2], 50, 50, 0.0, 0.0, 0.5, radius_mm)

        assert result is not None
        # Both pad regions should be included
        assert result[10, 10] == True  # Near pad1
        assert result[40, 40] == True  # Near pad2


class TestGetPadPixels:
    """Tests for get_pad_pixels function."""

    def test_returns_pixel_list(self):
        """Test that function returns list of pixel coordinates."""
        pad = MockPad(
            position=VECTOR2I(5000000, 5000000),
            bbox=EDA_RECT(4500000, 4500000, 1000000, 1000000)  # 1mm x 1mm
        )

        pixels = get_pad_pixels(pad, 50, 50, 0.0, 0.0, 0.5)

        assert isinstance(pixels, list)
        assert len(pixels) > 0
        for r, c in pixels:
            assert 0 <= r < 50
            assert 0 <= c < 50

    def test_pixel_count_matches_area(self):
        """Test that pixel count approximately matches pad area."""
        # 2mm x 2mm pad at 0.5mm resolution = ~16 pixels
        pad = MockPad(
            position=VECTOR2I(5000000, 5000000),
            bbox=EDA_RECT(4000000, 4000000, 2000000, 2000000)
        )

        pixels = get_pad_pixels(pad, 50, 50, 0.0, 0.0, 0.5)

        # Should be roughly 4x4 = 16 pixels (with some tolerance)
        assert 12 <= len(pixels) <= 25

    def test_small_pad(self):
        """Test with very small pad."""
        pad = MockPad(
            position=VECTOR2I(5000000, 5000000),
            bbox=EDA_RECT(4900000, 4900000, 200000, 200000)  # 0.2mm x 0.2mm
        )

        pixels = get_pad_pixels(pad, 50, 50, 0.0, 0.0, 0.5)

        # Should have at least 1 pixel
        assert len(pixels) >= 1


class TestCreateMultilayerMaps:
    """Tests for create_multilayer_maps function."""

    @pytest.fixture
    def basic_setup(self, temp_dir):
        """Create basic setup for map creation tests."""
        import os
        from tests.fixtures.sample_boards import SIMPLE_2_LAYER_STACKUP

        filepath = os.path.join(temp_dir, "test.kicad_pcb")
        with open(filepath, "w") as f:
            f.write(SIMPLE_2_LAYER_STACKUP)

        # Create mock board with geometry
        smd_pad = MockPad(
            position=VECTOR2I(10000000, 10000000),
            layer=F_Cu,
            attribute=PAD_ATTRIB_SMD,
            selected=True
        )
        pth_pad = MockPad(
            position=VECTOR2I(20000000, 20000000),
            layer=F_Cu,
            attribute=PAD_ATTRIB_PTH,
            selected=True
        )
        fp = MockFootprint(reference="U1", pads=[smd_pad, pth_pad])

        track = MockTrack(
            layer=F_Cu,
            bbox=EDA_RECT(5000000, 5000000, 15000000, 500000)
        )
        via = MockVia(
            bbox=EDA_RECT(15000000, 15000000, 500000, 500000),
            layers=[F_Cu, B_Cu]
        )

        zone = MockZone(
            layers=[F_Cu, B_Cu],
            bbox=EDA_RECT(0, 0, 30000000, 30000000),
            filled=True
        )

        board = MockBoard(
            filename=filepath,
            footprints=[fp],
            tracks=[track, via],
            zones=[zone],
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"}
        )

        settings = {
            'ignore_traces': False,
            'ignore_polygons': False,
            'limit_area': False,
            'pad_dist_mm': 0,
            'use_heatsink': False,
        }

        return {
            'board': board,
            'copper_ids': [F_Cu, B_Cu],
            'rows': 60,
            'cols': 60,
            'x_min': 0.0,
            'y_min': 0.0,
            'res': 0.5,
            'settings': settings,
            'k_fr4': 1.0,
            'k_cu_layers': [400.0, 400.0],
            'via_factor': 1300.0,
            'pads_list': [smd_pad, pth_pad],
        }

    def test_returns_three_arrays(self, basic_setup):
        """Test that function returns K, V, H arrays."""
        K, V, H = create_multilayer_maps(**basic_setup)

        assert K is not None
        assert V is not None
        assert H is not None

    def test_k_shape_matches_layers(self, basic_setup):
        """Test that K has correct shape."""
        K, V, H = create_multilayer_maps(**basic_setup)

        expected_shape = (2, basic_setup['rows'], basic_setup['cols'])
        assert K.shape == expected_shape

    def test_v_and_h_shape(self, basic_setup):
        """Test that V and H have correct 2D shape."""
        K, V, H = create_multilayer_maps(**basic_setup)

        expected_shape = (basic_setup['rows'], basic_setup['cols'])
        assert V.shape == expected_shape
        assert H.shape == expected_shape

    def test_k_has_copper_values(self, basic_setup):
        """Test that K contains copper conductivity values."""
        K, V, H = create_multilayer_maps(**basic_setup)

        # Should have some values > 1.0 (FR4 baseline)
        assert np.max(K) > 1.0

    def test_pth_pad_on_all_layers(self, basic_setup):
        """Test that PTH pads appear on all layers."""
        K, V, H = create_multilayer_maps(**basic_setup)

        # PTH pad is at (20mm, 20mm), grid position ~ (40, 40)
        # It should create copper on both layers
        # Check that both layers have elevated K values somewhere
        assert np.max(K[0]) > basic_setup['k_fr4']
        assert np.max(K[1]) > basic_setup['k_fr4']

    def test_via_increases_v_map(self, basic_setup):
        """Test that vias increase V map values."""
        K, V, H = create_multilayer_maps(**basic_setup)

        # Should have some V values > 1.0 where vias exist
        assert np.max(V) > 1.0

    def test_ignore_traces_setting(self, basic_setup):
        """Test ignore_traces setting."""
        setup = basic_setup.copy()
        setup['settings'] = setup['settings'].copy()
        setup['settings']['ignore_traces'] = True

        K, V, H = create_multilayer_maps(**setup)

        # With traces ignored, there should still be copper from pads/zones
        # but potentially less than with traces
        assert K is not None

    def test_heatsink_detection(self, basic_setup, temp_dir):
        """Test heatsink detection from User.Eco1 layer."""
        setup = basic_setup.copy()
        setup['settings'] = setup['settings'].copy()
        setup['settings']['use_heatsink'] = True

        # Add drawing on Eco1_User
        drawing = MockDrawing(
            layer=Eco1_User,
            bbox=EDA_RECT(10000000, 10000000, 10000000, 10000000)
        )
        setup['board']._drawings = [drawing]

        K, V, H = create_multilayer_maps(**setup)

        # H map should have some values set
        assert np.max(H) > 0

    def test_limit_area_creates_mask(self, basic_setup):
        """Test that limit_area creates distance mask."""
        setup = basic_setup.copy()
        setup['settings'] = setup['settings'].copy()
        setup['settings']['limit_area'] = True
        setup['settings']['pad_dist_mm'] = 5.0

        K, V, H = create_multilayer_maps(**setup)

        # Maps should still be valid
        assert K is not None
        # Some regions should be FR4 (not filled due to mask)
        # but near pads should be processed

    def test_wrapper_matches_internal_geometry_state(self, basic_setup):
        """Materialized maps should match the internal GeometryState."""
        state = build_geometry_state(
            board=basic_setup['board'],
            copper_ids=basic_setup['copper_ids'],
            rows=basic_setup['rows'],
            cols=basic_setup['cols'],
            x_min=basic_setup['x_min'],
            y_min=basic_setup['y_min'],
            res=basic_setup['res'],
            settings=basic_setup['settings'],
            via_factor=basic_setup['via_factor'],
            pads_list=basic_setup['pads_list'],
        )
        K, V, H = create_multilayer_maps(**basic_setup)

        expected_K = np.full(K.shape, basic_setup['k_fr4'], dtype=np.float64)
        for layer_idx, layer_value in enumerate(basic_setup['k_cu_layers']):
            expected_K[layer_idx, state.copper_mask[layer_idx]] = layer_value

        np.testing.assert_array_equal(K, expected_K)
        np.testing.assert_array_equal(V, state.via_map)
        np.testing.assert_array_equal(H, state.heatsink_mask.astype(np.float64))

    def test_zone_fill_mask_respects_hit_test_filled_area(self):
        """Zone fill should follow HitTestFilledArea cell selection."""
        fill_mask = {
            (1500000, 1500000),
            (2500000, 1500000),
        }
        zone = MockZone(
            layers=[F_Cu],
            bbox=EDA_RECT(1000000, 1000000, 2000000, 1000000),
            filled=True,
            fill_mask=fill_mask,
        )
        board = MockBoard(
            zones=[zone],
            layer_names={F_Cu: "F.Cu"},
        )
        settings = {
            'ignore_traces': False,
            'ignore_polygons': False,
            'limit_area': False,
            'pad_dist_mm': 0.0,
            'use_heatsink': False,
        }

        state = build_geometry_state(
            board=board,
            copper_ids=[F_Cu],
            rows=6,
            cols=6,
            x_min=0.0,
            y_min=0.0,
            res=1.0,
            settings=settings,
            via_factor=1300.0,
            pads_list=[],
        )

        assert state.copper_mask[0, 1, 1]
        assert state.copper_mask[0, 1, 2]
        assert np.count_nonzero(state.copper_mask[0]) == 2

    def test_diagonal_track_does_not_fill_its_bounding_box(self):
        """Exact track rasterization must not create false corner copper."""
        track = MockTrack(
            layer=F_Cu,
            bbox=EDA_RECT(750000, 750000, 8500000, 8500000),
            start=VECTOR2I(1000000, 1000000),
            end=VECTOR2I(9000000, 9000000),
            width=500000,
        )
        board = MockBoard(tracks=[track], layer_names={F_Cu: "F.Cu"})
        state = build_geometry_state(
            board=board,
            copper_ids=[F_Cu],
            rows=11,
            cols=11,
            x_min=0.0,
            y_min=0.0,
            res=1.0,
            settings={
                "ignore_traces": False,
                "ignore_polygons": False,
                "use_heatsink": False,
            },
            via_factor=1300.0,
            pads_list=[],
        )

        assert state.copper_mask[0, 1, 1]
        assert state.copper_mask[0, 8, 8]
        assert not state.copper_mask[0, 1, 8]
        assert not state.copper_mask[0, 8, 1]

    def test_mapping_error_does_not_return_partial_geometry(self):
        """Unexpected KiCad geometry errors must abort instead of returning."""
        class BrokenTrack:
            def GetLayer(self):
                raise RuntimeError("broken KiCad track")

        board = MockBoard(
            tracks=[BrokenTrack()],
            layer_names={F_Cu: "F.Cu"},
        )

        with pytest.raises(RuntimeError, match="refusing to use a partial map"):
            build_geometry_state(
                board=board,
                copper_ids=[F_Cu],
                rows=4,
                cols=4,
                x_min=0.0,
                y_min=0.0,
                res=1.0,
                settings={
                    "ignore_traces": False,
                    "ignore_polygons": False,
                    "use_heatsink": False,
                },
                via_factor=1300.0,
                pads_list=[],
            )

    def test_scanline_polygon_supports_holes(self):
        """Scanline fill can add an outline and remove an inner hole."""
        x_values = np.arange(10, dtype=np.float64) + 0.5
        y_values = np.arange(10, dtype=np.float64) + 0.5
        mask = np.zeros((10, 10), dtype=bool)
        outer = np.asarray([(1, 1), (9, 1), (9, 9), (1, 9)], dtype=np.float64)
        hole = np.asarray([(4, 4), (6, 4), (6, 6), (4, 6)], dtype=np.float64)

        _scanline_fill_polygon(mask, x_values, y_values, outer, True)
        _scanline_fill_polygon(mask, x_values, y_values, hole, False)

        assert mask[2, 2]
        assert not mask[4, 4]
        assert not mask[0, 0]

    def test_via_get_width_passes_layer_argument_for_kicad10(self):
        """KiCad 10 PCB_VIA.GetWidth expects a layer ID argument."""
        calls = []

        class KiCad10Via(MockVia):
            def GetPosition(self):
                return VECTOR2I(5500000, 5500000)

            def GetWidth(self, layer=None):
                calls.append(layer)
                if layer is None:
                    raise AssertionError("PCB_VIA::GetWidth called without a layer argument")
                return 800000

        via = KiCad10Via(
            bbox=EDA_RECT(5100000, 5100000, 800000, 800000),
            layers=[F_Cu, B_Cu],
        )
        board = MockBoard(
            tracks=[via],
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
        )

        state = build_geometry_state(
            board=board,
            copper_ids=[F_Cu, B_Cu],
            rows=10,
            cols=10,
            x_min=0.0,
            y_min=0.0,
            res=1.0,
            settings={
                "ignore_traces": False,
                "ignore_polygons": False,
                "use_heatsink": False,
            },
            via_factor=1300.0,
            pads_list=[],
        )

        assert calls == [F_Cu]
        assert state.via_map[5, 5] == 1300.0
        assert state.copper_mask[0, 5, 5]
        assert state.copper_mask[1, 5, 5]

    def test_geometry_state_uses_compact_dtypes(self):
        """GeometryState uses bool masks and float32 via_map for low RAM."""
        board = MockBoard(layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"})
        state = build_geometry_state(
            board=board,
            copper_ids=[F_Cu, B_Cu],
            rows=8,
            cols=8,
            x_min=0.0,
            y_min=0.0,
            res=1.0,
            settings={
                "ignore_traces": False,
                "ignore_polygons": False,
                "use_heatsink": False,
            },
            via_factor=1300.0,
            pads_list=[],
        )

        assert state.copper_mask.dtype == bool
        assert state.heatsink_mask.dtype == bool
        assert state.via_map.dtype == np.float32
        assert state.via_map.nbytes == 8 * 8 * 4


def _naive_scanline_fill_polygon(target, x_vals, y_vals, vertices, value=True):
    """Reference brute-force scanline fill for bit-exact equivalence checks."""
    if len(vertices) < 3:
        return
    x0 = vertices[:, 0]
    y0 = vertices[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    non_horizontal = y0 != y1
    if not np.any(non_horizontal):
        return
    x0 = x0[non_horizontal]
    y0 = y0[non_horizontal]
    x1 = x1[non_horizontal]
    y1 = y1[non_horizontal]
    for row_idx, y_value in enumerate(y_vals):
        crosses = ((y0 <= y_value) & (y_value < y1)) | (
            (y1 <= y_value) & (y_value < y0)
        )
        if not np.any(crosses):
            continue
        intersections = x0[crosses] + (
            (float(y_value) - y0[crosses])
            * (x1[crosses] - x0[crosses])
            / (y1[crosses] - y0[crosses])
        )
        intersections.sort()
        pair_count = intersections.size // 2
        if pair_count <= 0:
            continue
        intersections = intersections[: pair_count * 2]
        starts = np.searchsorted(x_vals, intersections[0::2], side="left")
        ends = np.searchsorted(x_vals, intersections[1::2], side="right")
        row = target[row_idx]
        for start, end in zip(starts, ends):
            if start < end:
                row[int(start):int(end)] = value


class _MockChain:
    def __init__(self, pts):
        self._pts = [VECTOR2I(int(x), int(y)) for x, y in pts]

    def PointCount(self):
        return len(self._pts)

    def CPoint(self, idx):
        return self._pts[idx]


class _MockPolySet:
    def __init__(self, outlines):
        # outlines: list of (outer_pts, [hole_pts, ...])
        self._outlines = outlines

    def OutlineCount(self):
        return len(self._outlines)

    def Outline(self, idx):
        return _MockChain(self._outlines[idx][0])

    def HoleCount(self, idx):
        return len(self._outlines[idx][1])

    def Hole(self, outline_idx, hole_idx):
        return _MockChain(self._outlines[outline_idx][1][hole_idx])


class _PolyZone:
    def __init__(self, outlines, layers):
        self._poly_set = _MockPolySet(outlines)
        self._layers = set(layers)

    def IsFilled(self):
        return True

    def IsOnLayer(self, lid):
        return lid in self._layers

    def GetFilledPolysList(self, lid):
        return self._poly_set


class TestFastZoneRasterizationAndProgress:
    """Tests for fast scanline rasterization, bit-exactness, and cancellation."""

    def test_scanline_bit_exact_concave_many_edges_and_oob(self):
        """Fast scanline fill is bit-exact with reference across concave/OOB/many-edge shapes."""
        x_vals = np.arange(80, dtype=np.float64) * 1e6 + 5e5
        y_vals = np.arange(70, dtype=np.float64) * 1e6 + 5e5

        # 120-vertex star-like concave polygon (>64 edges, >16 rows -> Y-strip path)
        angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        radii = np.where(np.arange(120) % 2 == 0, 32e6, 14e6)
        cx, cy = 40e6, 35e6
        star = np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])

        # Concave L-shape touching edges + out-of-bounds polygons
        l_shape = np.asarray(
            [
                (2e6, 2e6),
                (20e6, 2e6),
                (20e6, 8e6),
                (8e6, 8e6),
                (8e6, 25e6),
                (2e6, 25e6),
            ],
            dtype=np.float64,
        )
        oob_left = np.asarray([(-10e6, 5e6), (-2e6, 5e6), (-2e6, 15e6)], dtype=np.float64)
        oob_top = np.asarray([(10e6, 90e6), (20e6, 90e6), (15e6, 99e6)], dtype=np.float64)
        between_rows = np.asarray([(10e6, 10.6e6), (15e6, 10.6e6), (12e6, 10.9e6)], dtype=np.float64)

        for poly in (star, l_shape, oob_left, oob_top, between_rows):
            got = np.zeros((len(y_vals), len(x_vals)), dtype=bool)
            ref = np.zeros((len(y_vals), len(x_vals)), dtype=bool)
            _scanline_fill_polygon(got, x_vals, y_vals, poly, True)
            _naive_scanline_fill_polygon(ref, x_vals, y_vals, poly, True)
            np.testing.assert_array_equal(got, ref)

    def test_fill_zone_mask_polygons_bit_exact_multi_island_and_holes(self):
        """Local-bbox outline/hole fill matches brute-force scanline fill bit-for-bit."""
        x_vals = np.arange(60, dtype=np.float64) * 1e6 + 5e5
        y_vals = np.arange(50, dtype=np.float64) * 1e6 + 5e5

        # Island 1: concave polygon with 25 holes (including edge & OOB holes)
        outer_1 = [
            (5e6, 5e6), (55e6, 5e6), (55e6, 25e6), (40e6, 25e6),
            (40e6, 42e6), (5e6, 42e6),
        ]
        holes_1 = []
        for idx in range(25):
            hx = (7 + (idx % 6) * 5) * 1e6
            hy = (7 + (idx // 6) * 4) * 1e6
            holes_1.append([(hx, hy), (hx + 2e6, hy), (hx + 2e6, hy + 2e6), (hx, hy + 2e6)])
        # Boundary-overlapping hole and completely OOB hole
        holes_1.append([(4e6, 10e6), (8e6, 10e6), (8e6, 13e6), (4e6, 13e6)])
        holes_1.append([(200e6, 200e6), (205e6, 200e6), (205e6, 205e6), (200e6, 205e6)])

        # Island 2: small island near top-right + Island 3: completely OOB
        outer_2 = [(45e6, 30e6), (58e6, 30e6), (58e6, 46e6), (45e6, 46e6)]
        holes_2 = [[(49e6, 35e6), (53e6, 35e6), (53e6, 39e6), (49e6, 39e6)]]
        outer_3 = [(-20e6, -20e6), (-5e6, -20e6), (-5e6, -5e6)]

        outlines = [(outer_1, holes_1), (outer_2, holes_2), (outer_3, [])]
        zone = _PolyZone(outlines, [F_Cu])

        ref = np.zeros((len(y_vals), len(x_vals)), dtype=bool)
        for outer_pts, hole_list in outlines:
            inside = np.zeros_like(ref)
            _naive_scanline_fill_polygon(inside, x_vals, y_vals, np.asarray(outer_pts, dtype=np.float64), True)
            for hole_pts in hole_list:
                _naive_scanline_fill_polygon(inside, x_vals, y_vals, np.asarray(hole_pts, dtype=np.float64), False)
            ref |= inside

        got = np.zeros_like(ref)
        assert _fill_zone_mask_polygons(got, None, x_vals, y_vals, F_Cu, zone)
        np.testing.assert_array_equal(got, ref)

        # Also check with area_mask
        area_mask = np.zeros_like(ref)
        area_mask[10:40, 10:50] = True
        got_masked = np.zeros_like(ref)
        assert _fill_zone_mask_polygons(got_masked, area_mask, x_vals, y_vals, F_Cu, zone)
        np.testing.assert_array_equal(got_masked, ref & area_mask)

    def test_large_zone_1000x1000_with_500_holes_stress_under_1s(self):
        """1000x1000 zone with 2,000-vertex outline and 500 via holes completes in < 1.0 s."""
        import time

        x_vals = np.arange(1000, dtype=np.float64) * 1e5 + 5e4
        y_vals = np.arange(1000, dtype=np.float64) * 1e5 + 5e4

        angles = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
        radii = 4.5e7 + 2.0e6 * np.sin(8.0 * angles)
        outer = np.column_stack([5.0e7 + radii * np.cos(angles), 5.0e7 + radii * np.sin(angles)])

        hole_angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        holes = []
        for idx in range(500):
            gx = 1.5e7 + (idx % 25) * 2.8e6
            gy = 1.5e7 + (idx // 25) * 3.5e6
            hx = gx + 3.0e5 * np.cos(hole_angles)
            hy = gy + 3.0e5 * np.sin(hole_angles)
            holes.append(list(zip(hx, hy)))

        zone = _PolyZone([(outer, holes)], [F_Cu])
        mask = np.zeros((1000, 1000), dtype=bool)

        t0 = time.monotonic()
        ok = _fill_zone_mask_polygons(mask, None, x_vals, y_vals, F_Cu, zone)
        elapsed = time.monotonic() - t0

        assert ok
        assert np.any(mask)
        assert not mask[500, 500] or np.count_nonzero(mask) < 900_000
        assert elapsed < 1.0, f"Zone rasterization too slow: {elapsed:.3f}s"

    def test_build_geometry_state_progress_and_cancellation(self):
        """build_geometry_state emits track/pad/zone progress and raises GeometryCancelledError."""
        pad = MockPad(position=VECTOR2I(5000000, 5000000), layer=F_Cu, attribute=PAD_ATTRIB_SMD)
        fp = MockFootprint(reference="U1", pads=[pad])
        track = MockTrack(layer=F_Cu, bbox=EDA_RECT(1000000, 1000000, 5000000, 500000))
        zone = MockZone(layers=[F_Cu], bbox=EDA_RECT(0, 0, 10000000, 10000000), filled=True)
        drawing = MockDrawing(layer=Eco1_User, bbox=EDA_RECT(2000000, 2000000, 2000000, 2000000))
        board = MockBoard(
            footprints=[fp],
            tracks=[track],
            zones=[zone],
            drawings=[drawing],
            layer_names={F_Cu: "F.Cu"},
        )
        settings = {"ignore_traces": False, "ignore_polygons": False, "use_heatsink": True}

        events = []

        def on_progress(done, total, msg):
            events.append((done, total, msg))
            return True

        state = build_geometry_state(
            board=board,
            copper_ids=[F_Cu],
            rows=20,
            cols=20,
            x_min=0.0,
            y_min=0.0,
            res=0.5,
            settings=settings,
            via_factor=1300.0,
            pads_list=[pad],
            progress_callback=on_progress,
        )
        assert state.copper_mask.shape == (1, 20, 20)
        messages = [m for _, _, m in events]
        assert any("tracks" in m.lower() for m in messages)
        assert any("pads" in m.lower() for m in messages)
        assert any("zone" in m.lower() for m in messages)
        assert any("thermal pad" in m.lower() for m in messages)

        # Returning False from progress_callback raises GeometryCancelledError
        def cancel_on_zone(done, total, msg):
            if "zone" in msg.lower():
                return False
            return True

        with pytest.raises(GeometryCancelledError):
            build_geometry_state(
                board=board,
                copper_ids=[F_Cu],
                rows=20,
                cols=20,
                x_min=0.0,
                y_min=0.0,
                res=0.5,
                settings=settings,
                via_factor=1300.0,
                pads_list=[pad],
                progress_callback=cancel_on_zone,
            )

        # cancel_check=lambda: True raises GeometryCancelledError immediately
        with pytest.raises(GeometryCancelledError):
            build_geometry_state(
                board=board,
                copper_ids=[F_Cu],
                rows=20,
                cols=20,
                x_min=0.0,
                y_min=0.0,
                res=0.5,
                settings=settings,
                via_factor=1300.0,
                pads_list=[pad],
                cancel_check=lambda: True,
            )


class _ShapedMockPad(MockPad):
    """MockPad with KiCad-like shape, orientation, LayerSet, and offset APIs."""

    def __init__(
        self,
        *,
        position,
        size,
        shape,
        layer=F_Cu,
        layer_set=None,
        attribute=PAD_ATTRIB_SMD,
        orientation_deg=0.0,
        offset=None,
        shape_pos=None,
        roundrect_ratio=0.25,
        chamfer_ratio=0.25,
        chamfer_positions=0,
        parent=None,
        hit_fn=None,
        bbox=None,
    ):
        self._size = size
        self._shape = shape
        self._layer_set = layer_set
        self._orientation_deg = float(orientation_deg)
        self._offset = offset
        self._shape_pos = shape_pos
        self._roundrect_ratio = float(roundrect_ratio)
        self._chamfer_ratio = float(chamfer_ratio)
        self._chamfer_positions = int(chamfer_positions)
        self._parent = parent
        self._hit_fn = hit_fn

        if bbox is None:
            cx = position.x * 1e-6
            cy = position.y * 1e-6
            if offset is not None:
                a = math.radians(self._orientation_deg)
                ox, oy = offset.x * 1e-6, offset.y * 1e-6
                cx += ox * math.cos(a) + oy * math.sin(a)
                cy += -ox * math.sin(a) + oy * math.cos(a)
            elif shape_pos is not None:
                cx, cy = shape_pos.x * 1e-6, shape_pos.y * 1e-6
            w = size.x * 1e-6
            h = size.y * 1e-6
            a = math.radians(self._orientation_deg)
            bb_w = abs(w * math.cos(a)) + abs(h * math.sin(a))
            bb_h = abs(w * math.sin(a)) + abs(h * math.cos(a))
            bbox = EDA_RECT(
                int(round((cx - 0.5 * bb_w) * 1e6)),
                int(round((cy - 0.5 * bb_h) * 1e6)),
                int(round(bb_w * 1e6)),
                int(round(bb_h * 1e6)),
            )

        super().__init__(
            position=position,
            layer=layer,
            attribute=attribute,
            bbox=bbox,
        )

    def GetSize(self):
        return self._size

    def GetShape(self):
        return self._shape

    def GetOrientationDegrees(self):
        return self._orientation_deg

    def GetRoundRectRadiusRatio(self):
        return self._roundrect_ratio

    def GetChamferRectRatio(self):
        return self._chamfer_ratio

    def GetChamferPositions(self):
        return self._chamfer_positions

    def GetLayerSet(self):
        if self._layer_set is None:
            raise AttributeError("no layer_set")
        return self._layer_set

    def GetOffset(self):
        if self._offset is None:
            return VECTOR2I(0, 0)
        return self._offset

    def ShapePos(self, layer_id=None):
        if self._shape_pos is not None:
            return self._shape_pos
        raise TypeError("no ShapePos")

    def GetParent(self):
        return self._parent

    def HitTest(self, pos, accuracy=0, layer_id=None):
        if self._hit_fn is not None:
            return self._hit_fn(pos, accuracy, layer_id)
        raise AttributeError("no hit_fn")


class _KiCad10LayerSet(MockLayerSet):
    """LayerSet supporting both Contains() and CuStack()."""

    def CuStack(self):
        return sorted(self._layers)


class TestPadGeometryAndAlignment:
    """Tests for v0.4.5 pad geometry, LayerSet, rotation, offset, and alignment."""

    @pytest.fixture(autouse=True)
    def _ensure_pad_shape_consts(self, monkeypatch):
        import pcbnew
        import ThermalSim.geometry_mapper as gm

        for target_mod in (pcbnew, gm.pcbnew):
            for idx, name in enumerate(
                [
                    "PAD_SHAPE_CIRCLE",
                    "PAD_SHAPE_RECT",
                    "PAD_SHAPE_OVAL",
                    "PAD_SHAPE_ROUNDRECT",
                    "PAD_SHAPE_CHAMFERED_RECT",
                    "PAD_SHAPE_CUSTOM",
                ]
            ):
                if not hasattr(target_mod, name):
                    monkeypatch.setattr(target_mod, name, idx, raising=False)

    def test_rotated_pad_orientation_kicad_y_down_ccw(self):
        """Rotated 30 deg and 45 deg pads must follow KiCad +Y-down CCW convention."""
        import pcbnew

        # 6mm x 1mm rect centered at (10, 10), rotated +30 deg CCW (+Y down).
        # At dx=+2.0, dy=-1.15 -> (12.0, 8.85) lies along the rotated +X axis (INSIDE).
        # At dx=+2.0, dy=+1.15 -> (12.0, 11.15) is on the mirrored side (OUTSIDE).
        pad_30 = _ShapedMockPad(
            position=VECTOR2I(10_000_000, 10_000_000),
            size=VECTOR2I(6_000_000, 1_000_000),
            shape=pcbnew.PAD_SHAPE_RECT,
            orientation_deg=30.0,
            layer=F_Cu,
        )
        board = MockBoard(
            footprints=[MockFootprint("U1", [pad_30])],
            layer_names={F_Cu: "F.Cu"},
        )
        res = 0.25
        rows = cols = 80
        state = build_geometry_state(
            board=board,
            copper_ids=[F_Cu],
            rows=rows,
            cols=cols,
            x_min=0.0,
            y_min=0.0,
            res=res,
            settings={"ignore_traces": True, "ignore_polygons": True, "use_heatsink": False},
            via_factor=1300.0,
            pads_list=[pad_30],
        )

        r_in = int((8.85 - 0.0) / res)
        c_in = int((12.0 - 0.0) / res)
        r_out = int((11.15 - 0.0) / res)
        c_out = int((12.0 - 0.0) / res)
        assert state.copper_mask[0, r_in, c_in]
        assert not state.copper_mask[0, r_out, c_out]

        power_idx = get_pad_power_indices(pad_30, 0, rows, cols, 0.0, 0.0, res)
        assert power_idx.size > 0
        np.testing.assert_array_equal(
            np.sort(power_idx), np.flatnonzero(state.copper_mask[0])
        )

        # 45 deg rect: (+2, -2) is along the +45 CCW axis (inside), (+2, +2) is outside.
        pad_45 = _ShapedMockPad(
            position=VECTOR2I(10_000_000, 10_000_000),
            size=VECTOR2I(7_000_000, 1_000_000),
            shape=pcbnew.PAD_SHAPE_RECT,
            orientation_deg=45.0,
            layer=F_Cu,
        )
        rs, re, cs, ce, mask_45 = get_pad_grid_mask(pad_45, rows, cols, 0.0, 0.0, res)
        full_45 = np.zeros((rows, cols), dtype=bool)
        full_45[rs:re, cs:ce] = mask_45
        assert full_45[int(8.0 / res), int(12.0 / res)]
        assert not full_45[int(12.0 / res), int(12.0 / res)]

    def test_b_cu_smd_pad_uses_layerset_when_getlayer_returns_zero(self):
        """KiCad 10 B.Cu SMD pad with GetLayer()==0 must map exclusively to B.Cu."""
        import pcbnew

        b_cu_pad = _ShapedMockPad(
            position=VECTOR2I(5_000_000, 5_000_000),
            size=VECTOR2I(2_000_000, 2_000_000),
            shape=pcbnew.PAD_SHAPE_RECT,
            layer=F_Cu,  # Simulates KiCad 10 returning 0 (F.Cu) on B.Cu SMD pad
            layer_set=_KiCad10LayerSet([B_Cu]),
            attribute=PAD_ATTRIB_SMD,
        )
        npth_pad = _ShapedMockPad(
            position=VECTOR2I(15_000_000, 5_000_000),
            size=VECTOR2I(2_000_000, 2_000_000),
            shape=pcbnew.PAD_SHAPE_CIRCLE,
            layer=F_Cu,
            layer_set=_KiCad10LayerSet([]),
            attribute=PAD_ATTRIB_NPTH,
        )
        copper_ids = [F_Cu, B_Cu]
        assert get_pad_copper_layer_indices(b_cu_pad, copper_ids) == [1]
        assert get_pad_primary_layer_id(b_cu_pad, copper_ids) == B_Cu
        assert get_pad_copper_layer_indices(npth_pad, copper_ids) == []

        board = MockBoard(
            footprints=[MockFootprint("U1", [b_cu_pad, npth_pad])],
            layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"},
        )
        state = build_geometry_state(
            board=board,
            copper_ids=copper_ids,
            rows=40,
            cols=40,
            x_min=0.0,
            y_min=0.0,
            res=0.5,
            settings={"ignore_traces": True, "ignore_polygons": True, "use_heatsink": False},
            via_factor=1300.0,
            pads_list=[b_cu_pad],
        )

        assert not np.any(state.copper_mask[0])
        assert np.any(state.copper_mask[1])
        power_idx = get_pad_power_indices(
            b_cu_pad, 1, 40, 40, 0.0, 0.0, 0.5, layer_id=B_Cu
        )
        assert power_idx.size > 0
        assert np.all(state.copper_mask.reshape(-1)[power_idx])

    def test_offset_pads_use_shapepos_or_rotated_getoffset(self):
        """Offset pads must center copper and power at ShapePos / rotated GetOffset."""
        import pcbnew

        # Case A: ShapePos provided (e.g. KiCad 10 ShapePos(layer_id))
        pad_shapepos = _ShapedMockPad(
            position=VECTOR2I(10_000_000, 10_000_000),
            size=VECTOR2I(2_000_000, 2_000_000),
            shape=pcbnew.PAD_SHAPE_RECT,
            shape_pos=VECTOR2I(14_000_000, 10_000_000),
            layer=F_Cu,
        )
        # Case B: No ShapePos, uses GetOffset=(+4, 0) rotated 30 deg CCW -> (13.464, 8.0)
        pad_offset = _ShapedMockPad(
            position=VECTOR2I(10_000_000, 10_000_000),
            size=VECTOR2I(2_000_000, 2_000_000),
            shape=pcbnew.PAD_SHAPE_RECT,
            orientation_deg=30.0,
            offset=VECTOR2I(4_000_000, 0),
            layer=F_Cu,
        )

        for pad, expected_cx, expected_cy in [
            (pad_shapepos, 14.0, 10.0),
            (pad_offset, 10.0 + 4.0 * math.cos(math.radians(30.0)), 10.0 - 4.0 * math.sin(math.radians(30.0))),
        ]:
            board = MockBoard(footprints=[MockFootprint("U1", [pad])], layer_names={F_Cu: "F.Cu"})
            state = build_geometry_state(
                board=board,
                copper_ids=[F_Cu],
                rows=40,
                cols=40,
                x_min=0.0,
                y_min=0.0,
                res=0.5,
                settings={"ignore_traces": True, "ignore_polygons": True, "use_heatsink": False},
                via_factor=1300.0,
                pads_list=[pad],
            )
            r_exp = int(expected_cy / 0.5)
            c_exp = int(expected_cx / 0.5)
            assert state.copper_mask[0, r_exp, c_exp]
            assert not state.copper_mask[0, int(10.0 / 0.5), int(10.0 / 0.5)]
            power_idx = get_pad_power_indices(pad, 0, 40, 40, 0.0, 0.0, 0.5)
            assert np.all(state.copper_mask.reshape(-1)[power_idx])

    def test_subpixel_and_narrow_pads_keep_copper_and_power_aligned(self):
        """Sub-pixel and narrow between-row pads stay >=1 cell and 100% aligned."""
        import pcbnew

        # 0.12 x 0.12 mm sub-pixel pad off cell center at (10.1, 10.1) on 0.5 mm grid
        tiny_pad = _ShapedMockPad(
            position=VECTOR2I(10_100_000, 10_100_000),
            size=VECTOR2I(120_000, 120_000),
            shape=pcbnew.PAD_SHAPE_RECT,
        )
        # Narrow 1.7 x 0.35 mm horizontal pad midway between row centers (y=10.0 vs 9.75/10.25)
        narrow_pad = _ShapedMockPad(
            position=VECTOR2I(20_000_000, 10_000_000),
            size=VECTOR2I(1_700_000, 350_000),
            shape=pcbnew.PAD_SHAPE_RECT,
        )

        board = MockBoard(
            footprints=[MockFootprint("U1", [tiny_pad, narrow_pad])],
            layer_names={F_Cu: "F.Cu"},
        )
        state = build_geometry_state(
            board=board,
            copper_ids=[F_Cu],
            rows=60,
            cols=60,
            x_min=0.0,
            y_min=0.0,
            res=0.5,
            settings={"ignore_traces": True, "ignore_polygons": True, "use_heatsink": False},
            via_factor=1300.0,
            pads_list=[tiny_pad, narrow_pad],
        )

        tiny_power = get_pad_power_indices(tiny_pad, 0, 60, 60, 0.0, 0.0, 0.5)
        narrow_power = get_pad_power_indices(narrow_pad, 0, 60, 60, 0.0, 0.0, 0.5)
        assert tiny_power.size >= 1
        assert narrow_power.size >= 3
        assert np.all(state.copper_mask.reshape(-1)[tiny_power])
        assert np.all(state.copper_mask.reshape(-1)[narrow_power])

    def test_circle_oval_roundrect_chamfer_custom_power_subset_of_copper(self):
        """All pad shapes produce non-empty power indices strictly inside copper_mask."""
        import pcbnew

        custom_pad = _ShapedMockPad(
            position=VECTOR2I(35_000_000, 15_000_000),
            size=VECTOR2I(3_000_000, 3_000_000),
            shape=pcbnew.PAD_SHAPE_CUSTOM,
            hit_fn=lambda pos, acc, lid: (
                (pos.x * 1e-6 - 35.0) ** 2 + (pos.y * 1e-6 - 15.0) ** 2 <= 1.5 ** 2
            ),
        )
        pads = [
            _ShapedMockPad(
                position=VECTOR2I(5_000_000, 5_000_000),
                size=VECTOR2I(4_000_000, 4_000_000),
                shape=pcbnew.PAD_SHAPE_CIRCLE,
            ),
            _ShapedMockPad(
                position=VECTOR2I(15_000_000, 5_000_000),
                size=VECTOR2I(5_000_000, 2_000_000),
                shape=pcbnew.PAD_SHAPE_OVAL,
                orientation_deg=45.0,
            ),
            _ShapedMockPad(
                position=VECTOR2I(25_000_000, 5_000_000),
                size=VECTOR2I(4_000_000, 4_000_000),
                shape=pcbnew.PAD_SHAPE_ROUNDRECT,
                roundrect_ratio=0.25,
            ),
            _ShapedMockPad(
                position=VECTOR2I(15_000_000, 15_000_000),
                size=VECTOR2I(4_000_000, 4_000_000),
                shape=pcbnew.PAD_SHAPE_CHAMFERED_RECT,
                chamfer_ratio=0.25,
                chamfer_positions=0b1111,
            ),
            custom_pad,
        ]
        board = MockBoard(
            footprints=[MockFootprint("U1", pads)],
            layer_names={F_Cu: "F.Cu"},
        )
        state = build_geometry_state(
            board=board,
            copper_ids=[F_Cu],
            rows=80,
            cols=80,
            x_min=0.0,
            y_min=0.0,
            res=0.5,
            settings={"ignore_traces": True, "ignore_polygons": True, "use_heatsink": False},
            via_factor=1300.0,
            pads_list=pads,
        )

        # Circle bbox corner (3.25, 3.25) is outside radius 2.0 from (5.0, 5.0)
        assert not state.copper_mask[0, int(3.25 / 0.5), int(3.25 / 0.5)]
        # Roundrect corner (23.25, 3.25) is outside r=1.0 rounded corner
        assert not state.copper_mask[0, int(3.25 / 0.5), int(23.25 / 0.5)]

        for pad in pads:
            power_idx = get_pad_power_indices(pad, 0, 80, 80, 0.0, 0.0, 0.5)
            assert power_idx.size > 0
            assert np.all(state.copper_mask.reshape(-1)[power_idx])
