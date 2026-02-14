"""
Tests for current path analysis module.

Covers physics validation (resistance, power conservation, bottleneck
detection) and unit tests (sigma map filtering, matrix symmetry,
cross-section uniformity, Q merging).
"""

import pytest
import numpy as np
import scipy.sparse as sp

from ThermalSim.current_analyzer import (
    CurrentPathPair,
    CurrentPathResult,
    build_net_conductivity_map,
    build_electrical_stiffness_matrix,
    solve_current_path,
    compute_current_density,
    compute_cross_section_profile,
    analyze_current_path,
    analyze_all_paths,
    merge_i2r_into_q,
    SIGMA_CU,
    SIGMA_BG,
)
from ThermalSim.geometry_mapper import get_pad_pixels
from tests.fixtures.current_path_configs import (
    simple_copper_bar_board,
    two_layer_via_board,
)
from tests.mocks.pcbnew_mock import (
    MockBoard, MockFootprint, MockPad, MockTrack, MockVia, MockZone,
    VECTOR2I, EDA_RECT, F_Cu, B_Cu, PAD_ATTRIB_SMD, PAD_ATTRIB_PTH,
    FromMM,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bar_board():
    """Simple copper bar board fixture."""
    return simple_copper_bar_board(length_mm=10.0, width_mm=2.0)


@pytest.fixture
def via_board():
    """Two-layer via board fixture."""
    return two_layer_via_board()


# ---------------------------------------------------------------------------
# Physics validation tests
# ---------------------------------------------------------------------------

class TestPhysicsValidation:
    """Physics-based validation of the current path solver."""

    @pytest.mark.physics
    def test_uniform_bar_resistance(self, bar_board):
        """
        Verify R ≈ ρL/(w*t) for a straight copper bar.

        For 10mm x 2mm x 35µm copper at σ = 5.8e7 S/m:
        R = L / (σ * w * t) = 0.010 / (5.8e7 * 0.002 * 35e-6) ≈ 2.46 mΩ
        """
        bb = bar_board
        res = 0.5  # mm
        length_mm = bb["length_mm"]
        width_mm = bb["width_mm"]
        t_cu = 35e-6  # m

        # Grid dimensions - ensure it covers the entire bar
        cols = int(length_mm / res) + 4
        rows = int(width_mm / res) + 4

        pair = CurrentPathPair(
            pad_a=bb["pad_a"],
            pad_b=bb["pad_b"],
            current_a=5.0,
            net_code=bb["net_code"],
            label="VCC bar",
        )

        result = analyze_current_path(
            pair=pair,
            K=np.ones((2, rows, cols)),
            V_map=np.ones((rows, cols)),
            board=bb["board"],
            copper_ids=bb["copper_ids"],
            rows=rows, cols=cols,
            x_min=-(res * 2), y_min=-(res * 2),
            res=res,
            copper_thickness_m=[t_cu, t_cu],
            gap_m=[1.53e-3],
            get_pad_pixels_func=get_pad_pixels,
        )

        # Analytical resistance
        R_analytical = length_mm * 1e-3 / (SIGMA_CU * width_mm * 1e-3 * t_cu)

        # Allow 35% tolerance due to grid discretization and pad geometry
        # (pad bounding boxes widen the conductor at the ends, lowering R)
        assert result.resistance_ohm > 0, "Resistance must be positive"
        assert abs(result.resistance_ohm - R_analytical) / R_analytical < 0.35, (
            f"R={result.resistance_ohm*1e3:.4f} mΩ vs analytical "
            f"R={R_analytical*1e3:.4f} mΩ (>{35}% error)"
        )

    @pytest.mark.physics
    def test_power_conservation(self, bar_board):
        """Verify ∫P_density·dV ≈ I²·R."""
        bb = bar_board
        res = 0.5
        length_mm = bb["length_mm"]
        width_mm = bb["width_mm"]
        t_cu = 35e-6

        cols = int(length_mm / res) + 4
        rows = int(width_mm / res) + 4

        current = 5.0
        pair = CurrentPathPair(
            pad_a=bb["pad_a"], pad_b=bb["pad_b"],
            current_a=current, net_code=bb["net_code"],
            label="power test",
        )

        result = analyze_current_path(
            pair=pair,
            K=np.ones((2, rows, cols)),
            V_map=np.ones((rows, cols)),
            board=bb["board"],
            copper_ids=bb["copper_ids"],
            rows=rows, cols=cols,
            x_min=-(res * 2), y_min=-(res * 2),
            res=res,
            copper_thickness_m=[t_cu, t_cu],
            gap_m=[1.53e-3],
            get_pad_pixels_func=get_pad_pixels,
        )

        # P = I²R from V-I measurement
        P_vi = result.power_loss_w

        # P from integrated Q_i2r
        P_q = float(np.sum(result.Q_i2r))

        # Both should agree within 20%
        if P_vi > 1e-9:
            rel_diff = abs(P_q - P_vi) / P_vi
            assert rel_diff < 0.20, (
                f"Power mismatch: P(V*I)={P_vi*1e3:.3f} mW, "
                f"P(Q_sum)={P_q*1e3:.3f} mW, diff={rel_diff:.1%}"
            )

    @pytest.mark.physics
    def test_two_layer_via_path(self, via_board):
        """Current flows through a via between two layers."""
        vb = via_board
        res = 0.5
        t_cu = 35e-6

        cols = int(12.0 / res) + 4
        rows = int(6.0 / res) + 4

        pair = CurrentPathPair(
            pad_a=vb["pad_a"], pad_b=vb["pad_b"],
            current_a=3.0, net_code=vb["net_code"],
            label="via path",
        )

        # Build V_map with high values at via location
        V_map = np.ones((rows, cols))
        # Via at ~(5mm, 2mm) from origin
        via_r = int(2.0 / res) + 2
        via_c = int(5.0 / res) + 2
        if via_r < rows and via_c < cols:
            V_map[max(0, via_r-1):via_r+2, max(0, via_c-1):via_c+2] = 50.0

        result = analyze_current_path(
            pair=pair,
            K=np.ones((2, rows, cols)),
            V_map=V_map,
            board=vb["board"],
            copper_ids=vb["copper_ids"],
            rows=rows, cols=cols,
            x_min=-res*2, y_min=-res*2,
            res=res,
            copper_thickness_m=[t_cu, t_cu],
            gap_m=[1.53e-3],
            get_pad_pixels_func=get_pad_pixels,
        )

        # Should have positive resistance and power
        assert result.resistance_ohm > 0
        assert result.power_loss_w > 0
        assert result.voltage_drop_v > 0

        # Current density should be non-zero on both layers
        J_top = result.J_magnitude[0]
        J_bot = result.J_magnitude[1]
        assert np.max(J_top) > 0, "No current on top layer"
        assert np.max(J_bot) > 0, "No current on bottom layer"


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSigmaMapNetFiltering:
    """Test that build_net_conductivity_map only fills target-net copper."""

    def test_only_target_net_gets_sigma(self):
        """Tracks on a different net should not appear in sigma map."""
        net_target = 1
        net_other = 2

        # Track on target net
        track_target = MockTrack(
            layer=F_Cu,
            start=VECTOR2I(FromMM(0), FromMM(1)),
            end=VECTOR2I(FromMM(5), FromMM(1)),
            width=FromMM(1.0),
            net_code=net_target,
        )
        # Track on different net
        track_other = MockTrack(
            layer=F_Cu,
            start=VECTOR2I(FromMM(0), FromMM(3)),
            end=VECTOR2I(FromMM(5), FromMM(3)),
            width=FromMM(1.0),
            net_code=net_other,
        )

        board = MockBoard(
            footprints=[],
            tracks=[track_target, track_other],
            zones=[],
        )

        sigma = build_net_conductivity_map(
            board, net_target, [F_Cu, B_Cu],
            rows=10, cols=12, x_min=0, y_min=0, res=0.5,
            copper_thickness_m=[35e-6, 35e-6],
        )

        # Target track region (y~1mm, row ~2) should have copper
        assert sigma[0, 2, 3] == SIGMA_CU

        # Other track region (y~3mm, row ~6) should be background
        assert sigma[0, 6, 3] == SIGMA_BG


class TestMatrixSymmetry:
    """Test electrical stiffness matrix properties."""

    def test_k_elec_is_symmetric(self):
        """Electrical conductance matrix should be symmetric."""
        rows, cols = 8, 8
        layer_count = 2
        sigma = np.full((layer_count, rows, cols), SIGMA_CU)
        V_map = np.ones((rows, cols))
        dx = dy = 0.5e-3
        t_cu = [35e-6, 35e-6]
        gap_m = [1.5e-3]

        K_elec = build_electrical_stiffness_matrix(
            sigma, V_map, layer_count, rows, cols, t_cu, gap_m, dx, dy,
        )

        # Check symmetry: K - K^T should have near-zero entries
        diff = K_elec - K_elec.T
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-10, (
            f"Matrix not symmetric: max |K - K^T| = {np.max(np.abs(diff.data)):.2e}"
        )

    def test_k_elec_row_sums_zero(self):
        """For an isolated system (no BCs), row sums should be ~0."""
        rows, cols = 6, 6
        layer_count = 1
        sigma = np.full((layer_count, rows, cols), SIGMA_CU)
        V_map = np.ones((rows, cols))
        dx = dy = 0.5e-3

        K_elec = build_electrical_stiffness_matrix(
            sigma, V_map, layer_count, rows, cols,
            [35e-6], [], dx, dy,
        )

        row_sums = np.abs(K_elec.sum(axis=1)).A1
        assert np.max(row_sums) < 1e-6, (
            f"Row sums not zero: max = {np.max(row_sums):.2e}"
        )


class TestCrossSectionProfile:
    """Test cross-section profiling."""

    def test_cross_section_uniform_bar(self):
        """Uniform bar should have roughly constant cross-section area."""
        # Simulate a uniform bar: sigma = SIGMA_CU everywhere, linear V
        rows, cols = 4, 20
        layer_count = 1
        t_cu = 35e-6
        dx = dy = 0.5e-3

        sigma = np.full((layer_count, rows, cols), SIGMA_CU)
        # Linear voltage field from 0 to 1V
        V = np.zeros((layer_count, rows, cols))
        for c in range(cols):
            V[0, :, c] = c / (cols - 1)

        J_mag = np.full((layer_count, rows, cols), 0.0)
        # J = sigma * dV/dx (uniform)
        dVdx = 1.0 / ((cols - 1) * dx)
        J_mag[0, :, :] = SIGMA_CU * dVdx

        profile = compute_cross_section_profile(
            V, sigma, J_mag, [t_cu], dx, dy, n_slices=10,
        )

        # All slices should have similar copper area
        areas = [s["copper_area_mm2"] for s in profile if s["n_cells"] > 0]
        if len(areas) > 2:
            mean_area = np.mean(areas)
            for a in areas:
                if mean_area > 0:
                    assert abs(a - mean_area) / mean_area < 0.5, (
                        f"Non-uniform cross-section: {a:.4f} vs mean {mean_area:.4f}"
                    )


class TestMergeI2r:
    """Test merging I²R power into thermal Q vector."""

    def test_merge_additive(self):
        """Multiple paths should add to Q independently."""
        N = 100
        Q = np.ones(N) * 0.5

        r1 = CurrentPathResult(
            resistance_ohm=0.001, voltage_drop_v=0.005,
            power_loss_w=0.025,
            V_field=np.zeros((1, 10, 10)),
            J_magnitude=np.zeros((1, 10, 10)),
            Q_i2r=np.ones(N) * 0.1,
        )
        r2 = CurrentPathResult(
            resistance_ohm=0.002, voltage_drop_v=0.010,
            power_loss_w=0.050,
            V_field=np.zeros((1, 10, 10)),
            J_magnitude=np.zeros((1, 10, 10)),
            Q_i2r=np.ones(N) * 0.2,
        )

        Q_new = merge_i2r_into_q(Q, [r1, r2])

        # Q should be 0.5 + 0.1 + 0.2 = 0.8
        np.testing.assert_allclose(Q_new, 0.8, atol=1e-10)

    def test_merge_preserves_original(self):
        """Original Q should not be modified."""
        N = 50
        Q = np.ones(N) * 1.0
        Q_copy = Q.copy()

        r = CurrentPathResult(
            resistance_ohm=0.001, voltage_drop_v=0.005,
            power_loss_w=0.025,
            V_field=np.zeros((1, 5, 10)),
            J_magnitude=np.zeros((1, 5, 10)),
            Q_i2r=np.ones(N) * 0.3,
        )

        merge_i2r_into_q(Q, [r])
        np.testing.assert_array_equal(Q, Q_copy)

    def test_merge_empty_results(self):
        """Empty results list should return unchanged Q."""
        Q = np.ones(20) * 2.0
        Q_new = merge_i2r_into_q(Q, [])
        np.testing.assert_array_equal(Q_new, Q)


class TestSolveCurrentPath:
    """Test the linear solve function directly."""

    def test_pinned_voltage_is_zero(self):
        """The pinned pad B cell should have V = 0."""
        rows, cols = 6, 6
        layer_count = 1
        sigma = np.full((layer_count, rows, cols), SIGMA_CU)
        V_map = np.ones((rows, cols))
        dx = dy = 0.5e-3

        K_elec = build_electrical_stiffness_matrix(
            sigma, V_map, layer_count, rows, cols,
            [35e-6], [], dx, dy,
        )

        pad_a_pixels = [(3, 0), (3, 1)]
        pad_b_pixels = [(3, 4), (3, 5)]

        V_flat, V_a, V_b = solve_current_path(
            K_elec, pad_a_pixels, pad_b_pixels,
            0, 0, 1.0, layer_count, rows, cols,
        )

        # V_a should be positive (current flows from A to B)
        assert V_a > V_b
        # V at pinned cell should be close to 0
        pin_idx = 0 * rows * cols + 3 * cols + 4
        assert abs(V_flat[pin_idx]) < 1e-10


class TestAnalyzeAllPaths:
    """Test the batch analysis wrapper."""

    def test_analyze_empty_list(self):
        """Empty list of pairs should return empty results."""
        results = analyze_all_paths(
            pairs=[],
            K=np.ones((1, 5, 5)),
            V_map=np.ones((5, 5)),
            board=MockBoard(),
            copper_ids=[F_Cu],
            rows=5, cols=5,
            x_min=0, y_min=0, res=1.0,
            copper_thickness_m=[35e-6],
            gap_m=[],
            get_pad_pixels_func=get_pad_pixels,
        )
        assert results == []


class TestBottleneckDetection:
    """Test that narrower sections produce higher current density."""

    @pytest.mark.physics
    def test_bottleneck_higher_j(self):
        """A narrow section in a conductor should show higher J.

        Directly builds a sigma map with a known bottleneck shape
        to avoid bounding-box overlap issues with MockTrack.
        """
        rows, cols = 20, 40
        layer_count = 1
        t_cu = 35e-6
        dx = dy = 0.25e-3  # 0.25mm resolution

        # Build sigma: wide-narrow-wide bar on single layer
        sigma = np.full((layer_count, rows, cols), SIGMA_BG)
        # Wide section (8 rows): cols 0-14
        sigma[0, 6:14, 0:15] = SIGMA_CU
        # Narrow section (2 rows): cols 15-24
        sigma[0, 9:11, 15:25] = SIGMA_CU
        # Wide section (8 rows): cols 25-39
        sigma[0, 6:14, 25:40] = SIGMA_CU

        V_map = np.ones((rows, cols))
        K_elec = build_electrical_stiffness_matrix(
            sigma, V_map, layer_count, rows, cols,
            [t_cu], [], dx, dy,
        )

        # Inject current at left wide section, extract at right
        pad_a_pix = [(r, 0) for r in range(6, 14)]
        pad_b_pix = [(r, 39) for r in range(6, 14)]

        V_flat, V_a, V_b = solve_current_path(
            K_elec, pad_a_pix, pad_b_pix,
            0, 0, 5.0, layer_count, rows, cols,
        )

        J_mag, _ = compute_current_density(
            V_flat, sigma, layer_count, rows, cols, dx, dy,
        )

        J = J_mag[0]

        # Sample J at center of narrow section (row 10, col 20)
        J_narrow = np.mean(J[9:11, 18:22])
        # Sample J at center of wide section (row 10, col 7)
        J_wide = np.mean(J[9:11, 5:10])

        # Narrow section should have higher J (ratio ~ wide_rows/narrow_rows = 4)
        assert J_narrow > 0, "Current density should be non-zero in narrow section"
        assert J_wide > 0, "Current density should be non-zero in wide section"
        assert J_narrow > J_wide * 1.5, (
            f"Narrow section J={J_narrow:.1f} should be > 1.5x wide J={J_wide:.1f}"
        )
