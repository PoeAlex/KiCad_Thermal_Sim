"""
Unit tests for geometry_mapper module.

This module tests the PCB geometry to thermal grid mapping functions.
"""

import pytest
import numpy as np

from ThermalSim.geometry_mapper import (
    FillContext,
    _bbox_to_grid_indices,
    _fill_box,
    _fill_via,
    _fill_heatsink,
    build_pad_distance_mask,
    get_pad_pixels,
    create_multilayer_maps,
)

from tests.mocks.pcbnew_mock import (
    MockBoard, MockPad, MockZone, MockFootprint, MockTrack, MockVia,
    VECTOR2I, EDA_RECT, F_Cu, B_Cu, In1_Cu, In2_Cu,
    PAD_ATTRIB_PTH, PAD_ATTRIB_SMD, Eco1_User,
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
