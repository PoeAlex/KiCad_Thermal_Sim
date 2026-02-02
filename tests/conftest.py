"""
pytest configuration and fixtures for ThermalSim tests.

This module provides:
- Automatic pcbnew mock injection
- Common fixtures for test data
- pytest markers for test categorization

IMPORTANT: The pcbnew mock must be installed before ANY ThermalSim
module imports, as several modules import pcbnew at module level.
"""

import os
import sys
import tempfile

# ============================================================================
# CRITICAL: Install pcbnew mock BEFORE any imports that might trigger
# ThermalSim module loading
# ============================================================================

# First, add the tests directory to path so we can import mocks
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_plugin_dir = os.path.dirname(_tests_dir)

# Temporarily add tests directory to import mocks
sys.path.insert(0, _tests_dir)

# Import and install mock pcbnew module
from mocks.pcbnew_mock import install_mock, uninstall_mock
from mocks.wx_mock import install_wx_mock, uninstall_wx_mock
install_mock()
install_wx_mock()

# Now add the plugin directory to path for ThermalSim imports
sys.path.insert(0, _plugin_dir)

# Remove tests directory from path (no longer needed)
sys.path.remove(_tests_dir)

# Now safe to import pytest and numpy
import pytest
import numpy as np


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "physics: marks tests as physics validation tests (may be slower)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_wx: marks tests that require wxPython"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_mock_pcbnew():
    """Ensure pcbnew and wx mocks are installed for all tests."""
    install_mock()
    install_wx_mock()
    yield
    uninstall_mock()
    uninstall_wx_mock()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_2layer_board_file(temp_dir):
    """Create a temporary 2-layer board file."""
    from tests.fixtures.sample_boards import SIMPLE_2_LAYER_STACKUP
    filepath = os.path.join(temp_dir, "test_2layer.kicad_pcb")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(SIMPLE_2_LAYER_STACKUP)
    return filepath


@pytest.fixture
def sample_4layer_board_file(temp_dir):
    """Create a temporary 4-layer board file."""
    from tests.fixtures.sample_boards import SIMPLE_4_LAYER_STACKUP
    filepath = os.path.join(temp_dir, "test_4layer.kicad_pcb")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(SIMPLE_4_LAYER_STACKUP)
    return filepath


@pytest.fixture
def sample_6layer_board_file(temp_dir):
    """Create a temporary 6-layer board file."""
    from tests.fixtures.sample_boards import SIMPLE_6_LAYER_STACKUP
    filepath = os.path.join(temp_dir, "test_6layer.kicad_pcb")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(SIMPLE_6_LAYER_STACKUP)
    return filepath


@pytest.fixture
def mock_board_2layer(sample_2layer_board_file):
    """Create a MockBoard with 2-layer stackup."""
    from tests.mocks.pcbnew_mock import MockBoard, MockFootprint, MockPad, VECTOR2I, F_Cu, B_Cu

    pad1 = MockPad(
        position=VECTOR2I(10000000, 10000000),
        layer=F_Cu,
        selected=True,
        number="1"
    )
    fp = MockFootprint(reference="U1", pads=[pad1])

    return MockBoard(
        filename=sample_2layer_board_file,
        footprints=[fp],
        layer_names={F_Cu: "F.Cu", B_Cu: "B.Cu"}
    )


@pytest.fixture
def mock_board_4layer(sample_4layer_board_file):
    """Create a MockBoard with 4-layer stackup."""
    from tests.mocks.pcbnew_mock import (
        MockBoard, MockFootprint, MockPad, MockZone, MockVia,
        VECTOR2I, EDA_RECT, F_Cu, In1_Cu, In2_Cu, B_Cu, PAD_ATTRIB_PTH
    )

    # PTH pad (spans all layers)
    pth_pad = MockPad(
        position=VECTOR2I(15000000, 15000000),
        layer=F_Cu,
        attribute=PAD_ATTRIB_PTH,
        selected=True,
        number="1"
    )

    # SMD pad on top layer
    smd_pad = MockPad(
        position=VECTOR2I(25000000, 15000000),
        layer=F_Cu,
        selected=True,
        number="2"
    )

    fp = MockFootprint(reference="U1", pads=[pth_pad, smd_pad])

    # Via
    via = MockVia(
        bbox=EDA_RECT(20000000, 20000000, 500000, 500000),
        layers=[F_Cu, In1_Cu, In2_Cu, B_Cu]
    )

    # Ground zone on all layers
    zone = MockZone(
        layers=[F_Cu, In1_Cu, In2_Cu, B_Cu],
        bbox=EDA_RECT(0, 0, 50000000, 50000000),
        filled=True,
        net_name="GND"
    )

    return MockBoard(
        filename=sample_4layer_board_file,
        footprints=[fp],
        tracks=[via],
        zones=[zone],
        layer_names={F_Cu: "F.Cu", In1_Cu: "In1.Cu", In2_Cu: "In2.Cu", B_Cu: "B.Cu"}
    )


@pytest.fixture
def default_settings():
    """Default simulation settings."""
    return {
        'power_str': '1.0',
        'time': 20.0,
        'amb': 25.0,
        'thick': 1.6,
        'res': 0.5,
        'show_all': True,
        'snapshots': False,
        'snap_count': 5,
        'output_dir': '',
        'ignore_traces': False,
        'ignore_polygons': False,
        'limit_area': False,
        'pad_dist_mm': 30.0,
        'use_heatsink': False,
        'pad_th': 1.0,
        'pad_k': 3.0,
        'pad_cap_areal': 0.0
    }


@pytest.fixture
def small_grid_params():
    """Small grid parameters for fast tests."""
    return {
        'rows': 20,
        'cols': 20,
        'x_min': 0.0,
        'y_min': 0.0,
        'res': 0.5,  # mm
    }


@pytest.fixture
def medium_grid_params():
    """Medium grid parameters for standard tests."""
    return {
        'rows': 50,
        'cols': 50,
        'x_min': 0.0,
        'y_min': 0.0,
        'res': 0.5,  # mm
    }


@pytest.fixture
def sample_conductivity_maps(small_grid_params):
    """Create sample K, V, H maps for solver tests."""
    rows = small_grid_params['rows']
    cols = small_grid_params['cols']
    layers = 2

    # K: FR4 baseline with some copper regions
    k_fr4 = 1.0
    k_cu = 400.0
    K = np.ones((layers, rows, cols)) * k_fr4

    # Add copper trace pattern (horizontal lines)
    K[0, rows//4:rows//4+2, :] = k_cu  # Top layer trace
    K[1, 3*rows//4:3*rows//4+2, :] = k_cu  # Bottom layer trace

    # V: Via map (unit by default, higher where vias exist)
    V = np.ones((rows, cols))
    V[rows//2-1:rows//2+2, cols//2-1:cols//2+2] = 50.0  # Via region

    # H: No heatsink
    H = np.zeros((rows, cols))

    return K, V, H


@pytest.fixture
def sample_stackup_info():
    """Sample stackup information for 2-layer board."""
    return {
        "board_thickness_mm": 1.6,
        "copper": [
            {"order": 0, "name": "F.Cu", "layer_id": 0, "thickness_mm": 0.035},
            {"order": 2, "name": "B.Cu", "layer_id": 31, "thickness_mm": 0.035},
        ],
        "dielectrics": [
            {"order": 1, "name": "dielectric", "type": "core", "thickness_mm": 1.53},
        ],
        "copper_ids": [0, 31],
        "dielectric_gaps_mm": [1.53],
        "file_layer_count": 3,
    }


@pytest.fixture
def sample_stackup_derived():
    """Sample derived stackup for report generation."""
    return {
        "total_thick_mm_used": 1.6,
        "stack_board_thick_mm": 1.6,
        "copper_thickness_mm_used": [0.035, 0.035],
        "gap_mm_used": [1.53],
        "gap_fallback_used": False,
    }


@pytest.fixture
def sample_k_norm_info():
    """Sample k_norm_info for report generation."""
    return {
        "strategy": "implicit_fvm_bdf2",
        "backend": "SciPy",
        "multi_phase": True,
        "N": 800,
        "nnz_K": 3200,
        "dt_base": 0.1,
        "steps_target": 200,
        "steps_total": 200,
        "factorization_s": 0.05,
        "factorizations": 3,
        "avg_solve_s": 0.001,
        "pin_w": 1.0,
        "pout_final_w": 0.98,
        "steady_rel_diff": 0.02,
        "t_fr4_eff_per_plane_mm": [0.765, 0.765],
    }
