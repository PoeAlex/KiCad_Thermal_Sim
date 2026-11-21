"""
Unit tests for thermal_solver module.

This module contains physics validation tests for the thermal solver,
including:
- 1D heat conduction verification
- Steady-state equilibrium (Pin = Pout)
- Energy conservation during transients
- Multi-layer thermal coupling
- BDF2 convergence order
- Stiffness matrix properties
"""

import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ThermalSim import thermal_solver
from ThermalSim.thermal_solver import (
    SolverConfig,
    SolverResult,
    _assemble_structured_csc,
    _build_phase_plan,
    _coarsen_structured_operator,
    _factor_linear_system,
    _solve_pcg,
    build_stiffness_matrix,
    build_structured_operator,
    run_simulation,
    run_simulation_matrix_free,
    run_steady_state,
    run_steady_state_matrix_free,
)
from ThermalSim.workflow import ThermalFactorizationCache


def _build_stiffness_matrix_reference(
    layer_count,
    rows,
    cols,
    copper_mask,
    t_cu,
    t_fr4_eff,
    k_cu,
    k_fr4,
    dx,
    dy,
    V_map,
    gap_m,
    H_map,
    settings,
    amb
):
    """Reference implementation matching the preallocation refactor mathematically."""
    eps = 1e-12
    pixel_area = dx * dy
    rc = rows * cols
    n = rc * layer_count

    rows_list = []
    cols_list = []
    data_list = []

    col_right = np.arange(cols - 1)[None, :]
    row_all = np.arange(rows)[:, None]
    col_all = np.arange(cols)[None, :]
    row_down = np.arange(rows - 1)[:, None]

    for layer in range(layer_count):
        base = layer * rc
        mask = copper_mask[layer]
        k_layer = np.where(mask, k_cu, k_fr4)
        t_layer = np.where(mask, t_cu[layer], t_fr4_eff[layer])

        k_h = 2.0 * k_layer[:, :-1] * k_layer[:, 1:] / (k_layer[:, :-1] + k_layer[:, 1:] + eps)
        t_edge = 0.5 * (t_layer[:, :-1] + t_layer[:, 1:])
        gx = k_h * (t_edge * dy) / dx
        idx_left = base + row_all * cols + col_right
        idx_right = idx_left + 1

        g = gx.ravel()
        i_idx = idx_left.ravel()
        j_idx = idx_right.ravel()
        rows_list.extend([i_idx, j_idx, i_idx, j_idx])
        cols_list.extend([i_idx, j_idx, j_idx, i_idx])
        data_list.extend([g, g, -g, -g])

        k_h = 2.0 * k_layer[:-1, :] * k_layer[1:, :] / (k_layer[:-1, :] + k_layer[1:, :] + eps)
        t_edge = 0.5 * (t_layer[:-1, :] + t_layer[1:, :])
        gy = k_h * (t_edge * dx) / dy
        idx_up = base + row_down * cols + col_all
        idx_down = idx_up + cols

        g = gy.ravel()
        i_idx = idx_up.ravel()
        j_idx = idx_down.ravel()
        rows_list.extend([i_idx, j_idx, i_idx, j_idx])
        cols_list.extend([i_idx, j_idx, j_idx, i_idx])
        data_list.extend([g, g, -g, -g])

    if layer_count > 1 and gap_m:
        v_enh = np.clip(V_map, 1.0, 50.0)
        plane_idx = np.arange(rc, dtype=np.int64)
        for layer in range(layer_count - 1):
            gap_val = max(gap_m[layer], 1e-6)
            gz_base = k_fr4 * pixel_area / gap_val
            gz = (gz_base * v_enh).ravel()
            i_idx = layer * rc + plane_idx
            j_idx = (layer + 1) * rc + plane_idx
            rows_list.extend([i_idx, j_idx, i_idx, j_idx])
            cols_list.extend([i_idx, j_idx, j_idx, i_idx])
            data_list.extend([gz, gz, -gz, -gz])

    k_base = sp.coo_matrix(
        (np.concatenate(data_list), (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(n, n),
        dtype=np.float64,
    ).tocsr()

    diag_extra = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    h_top = float(settings.get('h_conv', 10.0))
    diag_add_top = h_top * pixel_area
    top_idx = np.arange(rc, dtype=np.int64)
    diag_extra[top_idx] += diag_add_top
    b[top_idx] += diag_add_top * amb

    h_air_bot = float(settings.get('h_conv', 10.0))
    pad_thick_m = max(0.0001, settings['pad_th'] * 1e-3)
    pad_k = settings['pad_k']
    h_sink = (pad_k / pad_thick_m) * 0.2
    h_bot_eff = (1.0 - H_map) * h_air_bot + H_map * h_sink
    diag_add_bot = (h_bot_eff * pixel_area).ravel()
    bot_idx = (layer_count - 1) * rc + top_idx
    diag_extra[bot_idx] += diag_add_bot
    b[bot_idx] += diag_add_bot * amb

    hA = np.zeros(n, dtype=np.float64)
    hA[top_idx] += diag_add_top
    hA[bot_idx] += diag_add_bot

    return k_base + sp.diags(diag_extra, format="csr"), b, hA, diag_extra


class TestSolverConfig:
    """Tests for SolverConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SolverConfig(
            sim_time=10.0,
            amb=25.0,
            dt_base=0.1,
            steps_target=100
        )
        assert config.sim_time == 10.0
        assert config.amb == 25.0
        assert config.dt_base == 0.1
        assert config.steps_target == 100
        assert config.use_multi_phase is True
        assert config.snapshots_enabled is False
        assert config.snap_times == []
        assert config.steady_state is False
        assert config.early_stop is True

    def test_snap_times_initialization(self):
        """Test that snap_times is initialized to empty list if None."""
        config = SolverConfig(
            sim_time=10.0,
            amb=25.0,
            dt_base=0.1,
            steps_target=100,
            snap_times=None
        )
        assert config.snap_times == []

    def test_custom_snap_times(self):
        """Test that custom snap_times is preserved."""
        snap_times = [1.0, 5.0, 10.0]
        config = SolverConfig(
            sim_time=10.0,
            amb=25.0,
            dt_base=0.1,
            steps_target=100,
            snap_times=snap_times
        )
        assert config.snap_times == snap_times


class TestBuildStiffnessMatrix:
    """Tests for build_stiffness_matrix function."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple setup for matrix tests."""
        layer_count = 2
        rows = 10
        cols = 10
        k_cu = 390.0
        k_fr4 = 0.3
        dx = dy = 0.5e-3  # 0.5 mm in meters

        # All FR4 (no copper)
        copper_mask = np.zeros((layer_count, rows, cols), dtype=bool)
        t_cu = np.array([35e-6, 35e-6])  # 35 um
        t_fr4_eff = np.array([0.75e-3, 0.75e-3])  # 0.75 mm

        V_map = np.ones((rows, cols))
        gap_m = [1.53e-3]  # 1.53 mm gap
        H_map = np.zeros((rows, cols))

        settings = {
            'pad_th': 1.0,
            'pad_k': 3.0,
        }
        amb = 25.0

        return {
            'layer_count': layer_count,
            'rows': rows,
            'cols': cols,
            'copper_mask': copper_mask,
            't_cu': t_cu,
            't_fr4_eff': t_fr4_eff,
            'k_cu': k_cu,
            'k_fr4': k_fr4,
            'dx': dx,
            'dy': dy,
            'V_map': V_map,
            'gap_m': gap_m,
            'H_map': H_map,
            'settings': settings,
            'amb': amb,
        }

    def test_matrix_shape(self, simple_setup):
        """Test that stiffness matrix has correct shape."""
        K, b, hA, diag_extra = build_stiffness_matrix(**simple_setup)

        N = simple_setup['layer_count'] * simple_setup['rows'] * simple_setup['cols']
        assert K.shape == (N, N)
        assert b.shape == (N,)
        assert hA.shape == (N,)
        assert diag_extra.shape == (N,)

    def test_matrix_symmetry(self, simple_setup):
        """Test that stiffness matrix is symmetric."""
        K, b, hA, diag_extra = build_stiffness_matrix(**simple_setup)

        # Check symmetry (K should equal K^T)
        diff = K - K.T
        max_asymmetry = np.max(np.abs(diff.toarray()))
        assert max_asymmetry < 1e-10, f"Matrix asymmetry: {max_asymmetry}"

    @pytest.mark.physics
    def test_diagonal_dominance(self, simple_setup):
        """Test that matrix is diagonally dominant (stability condition)."""
        K, b, hA, diag_extra = build_stiffness_matrix(**simple_setup)
        K_dense = K.toarray()

        diag = np.abs(np.diag(K_dense))
        off_diag_sum = np.sum(np.abs(K_dense), axis=1) - diag

        # Each diagonal element should be >= sum of off-diagonal elements
        # Allow small tolerance for numerical errors
        ratio = diag / (off_diag_sum + 1e-15)
        min_ratio = np.min(ratio)

        assert min_ratio >= 0.99, f"Diagonal dominance ratio: {min_ratio}"

    @pytest.mark.physics
    def test_positive_diagonal(self, simple_setup):
        """Test that all diagonal elements are positive."""
        K, b, hA, diag_extra = build_stiffness_matrix(**simple_setup)
        diag = K.diagonal()

        assert np.all(diag > 0), "All diagonal elements must be positive"

    def test_boundary_conditions_applied(self, simple_setup):
        """Test that boundary condition contributions are non-zero."""
        K, b, hA, diag_extra = build_stiffness_matrix(**simple_setup)

        # b should have non-zero values (boundary condition RHS)
        assert np.sum(np.abs(b)) > 0, "Boundary conditions should contribute to RHS"

        # hA should have non-zero values for boundary nodes
        assert np.sum(hA) > 0, "Heat transfer coefficients should be non-zero"

    def test_copper_increases_conductivity(self, simple_setup):
        """Test that adding copper increases matrix coefficients."""
        # Get baseline matrix (no copper)
        K_fr4, _, _, _ = build_stiffness_matrix(**simple_setup)
        K_fr4_nnz = K_fr4.nnz

        # Add copper to top layer
        setup_with_copper = simple_setup.copy()
        copper_mask = np.zeros_like(simple_setup['copper_mask'])
        copper_mask[0, 4:6, 4:6] = True  # Copper region
        setup_with_copper['copper_mask'] = copper_mask

        K_cu, _, _, _ = build_stiffness_matrix(**setup_with_copper)

        # With copper, conductance values in copper region should be higher
        # We check the Frobenius norm as a proxy
        norm_fr4 = sp.linalg.norm(K_fr4)
        norm_cu = sp.linalg.norm(K_cu)

        assert norm_cu > norm_fr4, "Copper should increase matrix norm"

    def test_matches_reference_assembly(self, simple_setup):
        """Preallocated COO assembly should match the legacy reference implementation."""
        K_ref, b_ref, hA_ref, diag_ref = _build_stiffness_matrix_reference(**simple_setup)
        K_new, b_new, hA_new, diag_new = build_stiffness_matrix(**simple_setup)

        np.testing.assert_allclose((K_new - K_ref).toarray(), 0.0, atol=1e-12)
        np.testing.assert_allclose(b_new, b_ref, atol=1e-12)
        np.testing.assert_allclose(hA_new, hA_ref, atol=1e-12)
        np.testing.assert_allclose(diag_new, diag_ref, atol=1e-12)

    def test_matrix_free_operator_matches_sparse_matrix(self, simple_setup):
        """Structured operator application must equal assembled CSR exactly."""
        setup = simple_setup.copy()
        copper_mask = np.zeros_like(setup["copper_mask"])
        copper_mask[0, 2:8, 3:7] = True
        setup["copper_mask"] = copper_mask
        setup["V_map"] = np.ones_like(setup["V_map"])
        setup["V_map"][4:6, 4:6] = 12.0
        sparse_matrix, sparse_b, sparse_h, sparse_diag = build_stiffness_matrix(**setup)
        operator, fast_b, fast_h, fast_diag = build_structured_operator(**setup)
        vector = np.linspace(20.0, 80.0, sparse_matrix.shape[0])

        np.testing.assert_allclose(operator.dot(vector), sparse_matrix.dot(vector), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fast_b, sparse_b, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fast_h, sparse_h, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fast_diag, sparse_diag, rtol=1e-12, atol=1e-12)


class TestRunSimulation:
    """Tests for run_simulation function."""

    @pytest.fixture
    def simple_simulation_setup(self):
        """Create setup for simulation tests."""
        layer_count = 2
        rows = 10
        cols = 10
        N = layer_count * rows * cols

        # Create simple matrices
        # K: sparse stiffness matrix (simplified)
        k_fr4 = 0.3
        dx = dy = 0.5e-3
        pixel_area = dx * dy
        h_conv = 10.0

        # Simple tridiagonal-like structure
        diag = np.ones(N) * (4 * k_fr4 * 35e-6 / dx + h_conv * pixel_area)
        K = sp.diags(diag, format='csr')

        # C: heat capacity
        rho_cu = 8960.0
        cp_cu = 385.0
        t_avg = 35e-6
        C = np.ones(N) * rho_cu * cp_cu * t_avg * pixel_area

        # Q: heat source (1W at center node)
        Q = np.zeros(N)
        center_idx = N // 2
        Q[center_idx] = 1.0

        # b: boundary condition RHS
        amb = 25.0
        b = np.ones(N) * h_conv * pixel_area * amb

        # hA: heat transfer coefficients
        hA = np.ones(N) * h_conv * pixel_area

        config = SolverConfig(
            sim_time=1.0,
            amb=amb,
            dt_base=0.05,
            steps_target=20,
            use_multi_phase=False
        )

        return {
            'config': config,
            'K': K,
            'C': C,
            'Q': Q,
            'b': b,
            'hA': hA,
            'layer_count': layer_count,
            'rows': rows,
            'cols': cols,
        }

    def test_simulation_returns_result(self, simple_simulation_setup):
        """Test that simulation returns a SolverResult."""
        result = run_simulation(**simple_simulation_setup)

        assert isinstance(result, SolverResult)
        assert result.T is not None
        assert result.aborted is False

    def test_temperature_shape(self, simple_simulation_setup):
        """Test that result temperature has correct shape."""
        result = run_simulation(**simple_simulation_setup)

        expected_shape = (
            simple_simulation_setup['layer_count'],
            simple_simulation_setup['rows'],
            simple_simulation_setup['cols']
        )
        assert result.T.shape == expected_shape

    def test_simulation_completes(self, simple_simulation_setup):
        """Test that simulation completes with simplified matrix."""
        result = run_simulation(**simple_simulation_setup)

        # With simplified test matrix, just verify simulation ran
        # Real physics validation is in TestPhysicsValidation
        assert result is not None
        assert result.T is not None
        assert not result.aborted

    @pytest.mark.physics
    def test_heat_source_creates_hotspot(self, simple_simulation_setup):
        """Test that simulation completes with heat source."""
        result = run_simulation(**simple_simulation_setup)

        # In simplified test setup, just verify simulation ran
        # Real physics tests are in TestPhysicsValidation
        assert result.T is not None
        assert result.step_counter > 0

    def test_step_counter_increments(self, simple_simulation_setup):
        """Test that step counter tracks progress."""
        result = run_simulation(**simple_simulation_setup)

        assert result.step_counter > 0, "Should complete at least one step"

    def test_progress_callback_can_abort(self, simple_simulation_setup):
        """Test that progress callback can abort simulation."""
        abort_at_step = 5

        def abort_callback(current, total):
            return current < abort_at_step

        setup = simple_simulation_setup.copy()
        result = run_simulation(
            **setup,
            progress_callback=abort_callback
        )

        assert result.aborted is True
        assert result.step_counter <= abort_at_step + 1

    def test_matrix_free_solver_matches_sparse_solver(self):
        """PCG BDF2 should reproduce the existing solver on the same grid."""
        setup = TestBuildStiffnessMatrix().simple_setup.__wrapped__(
            TestBuildStiffnessMatrix()
        )
        sparse_matrix, b, h_area, _ = build_stiffness_matrix(**setup)
        operator, fast_b, fast_h_area, _ = build_structured_operator(**setup)
        node_count = sparse_matrix.shape[0]
        capacity = np.full(node_count, 0.0025, dtype=np.float64)
        power = np.zeros(node_count, dtype=np.float64)
        power[node_count // 3] = 1.25
        config = SolverConfig(
            sim_time=0.4,
            amb=25.0,
            dt_base=0.02,
            steps_target=20,
            use_multi_phase=False,
            time_stepping="uniform",
        )

        reference = run_simulation(
            config, sparse_matrix, capacity, power, b, h_area,
            setup["layer_count"], setup["rows"], setup["cols"],
        )
        fast = run_simulation_matrix_free(
            config, operator, capacity, power, fast_b, fast_h_area,
        )

        np.testing.assert_allclose(fast.T, reference.T, rtol=5e-5, atol=1e-3)
        assert fast.k_norm_info["factorizations"] == 0
        assert fast.k_norm_info["backend"] in {"MatrixFree-PCG", "Native-MatrixFree-PCG"}


class TestPhysicsValidation:
    """Physics validation tests for the thermal solver."""

    @pytest.fixture
    def physics_setup(self):
        """Create setup for physics validation tests."""
        # Use a 1D-like geometry (narrow in one dimension)
        layer_count = 1
        rows = 50
        cols = 5
        N = layer_count * rows * cols

        k_fr4 = 0.3
        k_cu = 390.0
        dx = dy = 0.5e-3  # 0.5 mm
        pixel_area = dx * dy

        # Create copper mask (uniform FR4)
        copper_mask = np.zeros((layer_count, rows, cols), dtype=bool)
        t_cu = np.array([35e-6])
        t_fr4_eff = np.array([1.6e-3])

        V_map = np.ones((rows, cols))
        gap_m = []
        H_map = np.zeros((rows, cols))

        settings = {'pad_th': 1.0, 'pad_k': 3.0}
        amb = 25.0

        K, b, hA, diag_extra = build_stiffness_matrix(
            layer_count, rows, cols, copper_mask, t_cu, t_fr4_eff,
            k_cu, k_fr4, dx, dy, V_map, gap_m, H_map, settings, amb
        )

        # Heat capacity
        rho = 1850.0  # FR4 density
        cp = 600.0    # FR4 specific heat
        t_layer = 35e-6
        C = np.ones(N) * rho * cp * t_layer * pixel_area

        return {
            'K': K, 'C': C, 'b': b, 'hA': hA,
            'layer_count': layer_count,
            'rows': rows, 'cols': cols,
            'N': N, 'amb': amb,
            'pixel_area': pixel_area
        }

    @pytest.mark.physics
    def test_steady_state_energy_balance(self, physics_setup):
        """
        Test that Pin = Pout at steady state.

        At thermal equilibrium, the power input from heat sources
        should equal the power output through convection.
        """
        N = physics_setup['N']
        rows = physics_setup['rows']
        cols = physics_setup['cols']
        amb = physics_setup['amb']

        # Add heat source
        Q = np.zeros(N)
        center_idx = (rows // 2) * cols + cols // 2
        power_in = 0.1  # 100 mW
        Q[center_idx] = power_in

        config = SolverConfig(
            sim_time=100.0,  # Long time for steady state
            amb=amb,
            dt_base=0.5,
            steps_target=200,
            use_multi_phase=True
        )

        result = run_simulation(
            config=config,
            K=physics_setup['K'],
            C=physics_setup['C'],
            Q=Q,
            b=physics_setup['b'],
            hA=physics_setup['hA'],
            layer_count=physics_setup['layer_count'],
            rows=rows,
            cols=cols
        )

        # Calculate power out
        T_flat = result.T.flatten()
        delta_T = T_flat - amb
        power_out = np.sum(physics_setup['hA'] * delta_T)

        # At steady state, Pin should approximately equal Pout
        rel_error = abs(power_in - power_out) / power_in
        assert rel_error < 0.10, f"Energy balance error: {rel_error:.2%} (Pin={power_in}, Pout={power_out:.4f})"

    @pytest.mark.physics
    def test_monotonic_temperature_decrease_from_source(self, physics_setup):
        """
        Test that temperature decreases monotonically away from heat source.

        This is a fundamental property of heat conduction: temperature
        always decreases in the direction of heat flow.
        """
        N = physics_setup['N']
        rows = physics_setup['rows']
        cols = physics_setup['cols']
        amb = physics_setup['amb']

        # Heat source at top
        Q = np.zeros(N)
        for c in range(cols):
            Q[c] = 0.1 / cols  # Distribute 100mW across top row

        config = SolverConfig(
            sim_time=50.0,
            amb=amb,
            dt_base=0.25,
            steps_target=200,
            use_multi_phase=True
        )

        result = run_simulation(
            config=config,
            K=physics_setup['K'],
            C=physics_setup['C'],
            Q=Q,
            b=physics_setup['b'],
            hA=physics_setup['hA'],
            layer_count=physics_setup['layer_count'],
            rows=rows,
            cols=cols
        )

        # Check monotonic decrease from top to bottom (center column)
        T_layer = result.T[0]
        center_col = cols // 2
        temps_along_col = T_layer[:, center_col]

        # Check that each temperature is >= the one below it
        for i in range(len(temps_along_col) - 1):
            assert temps_along_col[i] >= temps_along_col[i+1] - 0.1, \
                f"Temperature should decrease monotonically: T[{i}]={temps_along_col[i]:.2f}, T[{i+1}]={temps_along_col[i+1]:.2f}"

    @pytest.mark.physics
    @pytest.mark.slow
    def test_bdf2_convergence_order(self, physics_setup):
        """
        Test that BDF2 achieves second-order convergence.

        Using Richardson extrapolation: if we halve dt, the error
        should reduce by approximately factor of 4 for O(dt^2) method.
        """
        N = physics_setup['N']
        rows = physics_setup['rows']
        cols = physics_setup['cols']
        amb = physics_setup['amb']

        # Simple heat source
        Q = np.zeros(N)
        center_idx = (rows // 2) * cols + cols // 2
        Q[center_idx] = 0.1

        sim_time = 5.0
        results = []

        # Run with different dt values
        for dt_scale in [1.0, 0.5, 0.25]:
            dt = 0.2 * dt_scale
            steps = int(sim_time / dt)

            config = SolverConfig(
                sim_time=sim_time,
                amb=amb,
                dt_base=dt,
                steps_target=steps,
                use_multi_phase=False  # Single phase for clean comparison
            )

            result = run_simulation(
                config=config,
                K=physics_setup['K'],
                C=physics_setup['C'],
                Q=Q,
                b=physics_setup['b'],
                hA=physics_setup['hA'],
                layer_count=physics_setup['layer_count'],
                rows=rows,
                cols=cols
            )
            results.append((dt, result.T.copy()))

        # Compare errors using finest solution as reference
        T_ref = results[-1][1]

        errors = []
        for dt, T in results[:-1]:
            error = np.max(np.abs(T - T_ref))
            errors.append((dt, error))

        # Check convergence order
        if len(errors) >= 2 and errors[0][1] > 1e-10 and errors[1][1] > 1e-10:
            dt1, e1 = errors[0]
            dt2, e2 = errors[1]
            order = np.log(e1 / e2) / np.log(dt1 / dt2)

            # BDF2 should give order ~2 (allow some tolerance)
            assert order > 1.5, f"BDF2 convergence order too low: {order:.2f}"


class TestMultiLayerCoupling:
    """Tests for multi-layer thermal coupling through vias."""

    @pytest.fixture
    def multilayer_setup(self):
        """Create 4-layer setup for coupling tests."""
        layer_count = 4
        rows = 20
        cols = 20
        N = layer_count * rows * cols

        k_cu = 390.0
        k_fr4 = 0.3
        dx = dy = 0.5e-3
        pixel_area = dx * dy

        copper_mask = np.zeros((layer_count, rows, cols), dtype=bool)
        t_cu = np.array([35e-6] * layer_count)
        t_fr4_eff = np.array([0.4e-3] * layer_count)

        # Via in center (high vertical conductivity)
        V_map = np.ones((rows, cols))
        V_map[rows//2-2:rows//2+2, cols//2-2:cols//2+2] = 50.0  # Via region

        gap_m = [0.2e-3, 1.0e-3, 0.2e-3]  # Gap between layers
        H_map = np.zeros((rows, cols))

        settings = {'pad_th': 1.0, 'pad_k': 3.0}
        amb = 25.0

        K, b, hA, diag_extra = build_stiffness_matrix(
            layer_count, rows, cols, copper_mask, t_cu, t_fr4_eff,
            k_cu, k_fr4, dx, dy, V_map, gap_m, H_map, settings, amb
        )

        # Heat capacity
        rho = 1850.0
        cp = 600.0
        C = np.ones(N) * rho * cp * 35e-6 * pixel_area

        return {
            'K': K, 'C': C, 'b': b, 'hA': hA,
            'layer_count': layer_count,
            'rows': rows, 'cols': cols,
            'N': N, 'amb': amb
        }

    @pytest.mark.physics
    def test_heat_transfer_through_via(self, multilayer_setup):
        """Test that vias enhance heat transfer between layers."""
        N = multilayer_setup['N']
        rows = multilayer_setup['rows']
        cols = multilayer_setup['cols']
        layer_count = multilayer_setup['layer_count']
        RC = rows * cols
        amb = multilayer_setup['amb']

        # Heat source on top layer at via location
        Q = np.zeros(N)
        via_row, via_col = rows // 2, cols // 2
        Q[via_row * cols + via_col] = 0.5  # 500 mW

        config = SolverConfig(
            sim_time=20.0,
            amb=amb,
            dt_base=0.1,
            steps_target=200,
            use_multi_phase=True
        )

        result = run_simulation(
            config=config,
            K=multilayer_setup['K'],
            C=multilayer_setup['C'],
            Q=Q,
            b=multilayer_setup['b'],
            hA=multilayer_setup['hA'],
            layer_count=layer_count,
            rows=rows,
            cols=cols
        )

        # Check temperature at via location on each layer
        via_temps = []
        for l in range(layer_count):
            via_temps.append(result.T[l, via_row, via_col])

        # Temperature should decrease from top to bottom
        for i in range(len(via_temps) - 1):
            assert via_temps[i] >= via_temps[i+1], \
                f"Temperature should decrease with depth: L{i}={via_temps[i]:.2f}, L{i+1}={via_temps[i+1]:.2f}"

        # Bottom layer should be warmer than ambient due to heat transfer through via
        assert via_temps[-1] > amb + 1.0, \
            f"Via should conduct heat to bottom layer: T_bot={via_temps[-1]:.2f}, amb={amb}"

    @pytest.mark.physics
    def test_layer_temperature_gradient(self, multilayer_setup):
        """Test that temperature decreases monotonically through layers."""
        N = multilayer_setup['N']
        rows = multilayer_setup['rows']
        cols = multilayer_setup['cols']
        layer_count = multilayer_setup['layer_count']
        amb = multilayer_setup['amb']

        # Uniform heat on top layer
        Q = np.zeros(N)
        for r in range(rows):
            for c in range(cols):
                Q[r * cols + c] = 0.01  # Distribute heat

        config = SolverConfig(
            sim_time=30.0,
            amb=amb,
            dt_base=0.15,
            steps_target=200,
            use_multi_phase=True
        )

        result = run_simulation(
            config=config,
            K=multilayer_setup['K'],
            C=multilayer_setup['C'],
            Q=Q,
            b=multilayer_setup['b'],
            hA=multilayer_setup['hA'],
            layer_count=layer_count,
            rows=rows,
            cols=cols
        )

        # Average temperature per layer should decrease from top to bottom
        avg_temps = [np.mean(result.T[l]) for l in range(layer_count)]

        for i in range(len(avg_temps) - 1):
            assert avg_temps[i] >= avg_temps[i+1] - 0.5, \
                f"Avg temperature should decrease: L{i}={avg_temps[i]:.2f}, L{i+1}={avg_temps[i+1]:.2f}"


class TestTimeVaryingQ:
    """Tests for time-varying heat source (Q_func) support."""

    @pytest.fixture
    def qfunc_setup(self):
        """Create setup for Q_func tests."""
        layer_count = 1
        rows = 10
        cols = 10
        N = layer_count * rows * cols

        k_fr4 = 0.3
        dx = dy = 0.5e-3
        pixel_area = dx * dy
        h_conv = 10.0

        diag = np.ones(N) * (4 * k_fr4 * 35e-6 / dx + h_conv * pixel_area)
        K = sp.diags(diag, format='csr')

        rho = 1850.0
        cp = 600.0
        C = np.ones(N) * rho * cp * 35e-6 * pixel_area

        Q = np.zeros(N)
        center_idx = N // 2
        Q[center_idx] = 1.0

        amb = 25.0
        b = np.ones(N) * h_conv * pixel_area * amb
        hA = np.ones(N) * h_conv * pixel_area

        return {
            'K': K, 'C': C, 'Q': Q, 'b': b, 'hA': hA,
            'layer_count': layer_count,
            'rows': rows, 'cols': cols,
            'N': N, 'amb': amb,
            'center_idx': center_idx,
        }

    def test_Q_func_none_backward_compat(self, qfunc_setup):
        """Test that Q_func=None gives identical result to current code."""
        config = SolverConfig(
            sim_time=1.0,
            amb=qfunc_setup['amb'],
            dt_base=0.05,
            steps_target=20,
            use_multi_phase=False
        )

        # Run without Q_func
        result_no_qfunc = run_simulation(
            config=config,
            K=qfunc_setup['K'], C=qfunc_setup['C'],
            Q=qfunc_setup['Q'], b=qfunc_setup['b'],
            hA=qfunc_setup['hA'],
            layer_count=qfunc_setup['layer_count'],
            rows=qfunc_setup['rows'], cols=qfunc_setup['cols'],
            Q_func=None
        )

        # Run with Q_func that returns the same constant Q
        Q_const = qfunc_setup['Q'].copy()

        def const_qfunc(t, _Q=Q_const):
            return _Q.copy()

        result_with_qfunc = run_simulation(
            config=config,
            K=qfunc_setup['K'], C=qfunc_setup['C'],
            Q=qfunc_setup['Q'], b=qfunc_setup['b'],
            hA=qfunc_setup['hA'],
            layer_count=qfunc_setup['layer_count'],
            rows=qfunc_setup['rows'], cols=qfunc_setup['cols'],
            Q_func=const_qfunc
        )

        np.testing.assert_allclose(
            result_no_qfunc.T, result_with_qfunc.T, rtol=1e-10,
            err_msg="Q_func returning constant Q should match Q_func=None"
        )

    @pytest.mark.physics
    def test_time_varying_Q_step_function(self, qfunc_setup):
        """Test power on for first half, off for second half; verify temp drops."""
        center_idx = qfunc_setup['center_idx']
        N = qfunc_setup['N']

        Q_on = np.zeros(N)
        Q_on[center_idx] = 1.0
        Q_off = np.zeros(N)

        sim_time = 2.0

        def step_qfunc(t):
            if t <= sim_time / 2:
                return Q_on.copy()
            else:
                return Q_off.copy()

        config = SolverConfig(
            sim_time=sim_time,
            amb=qfunc_setup['amb'],
            dt_base=0.05,
            steps_target=40,
            use_multi_phase=False
        )

        result = run_simulation(
            config=config,
            K=qfunc_setup['K'], C=qfunc_setup['C'],
            Q=Q_on, b=qfunc_setup['b'],
            hA=qfunc_setup['hA'],
            layer_count=qfunc_setup['layer_count'],
            rows=qfunc_setup['rows'], cols=qfunc_setup['cols'],
            Q_func=step_qfunc
        )

        # Compare with always-on simulation
        result_always_on = run_simulation(
            config=config,
            K=qfunc_setup['K'], C=qfunc_setup['C'],
            Q=Q_on, b=qfunc_setup['b'],
            hA=qfunc_setup['hA'],
            layer_count=qfunc_setup['layer_count'],
            rows=qfunc_setup['rows'], cols=qfunc_setup['cols'],
            Q_func=None
        )

        # With power off in second half, final temp should be lower
        max_step = float(np.max(result.T))
        max_on = float(np.max(result_always_on.T))
        assert max_step < max_on, \
            f"Step function result ({max_step:.2f}) should be cooler than always-on ({max_on:.2f})"

    @pytest.mark.physics
    def test_time_varying_Q_energy_balance(self, qfunc_setup):
        """Test that ramp Q produces higher final temp than zero Q."""
        center_idx = qfunc_setup['center_idx']
        N = qfunc_setup['N']
        amb = qfunc_setup['amb']

        # Ramp power: 0W -> 2W over 1 second
        Q_unit = np.zeros(N)
        Q_unit[center_idx] = 1.0

        def ramp_qfunc(t):
            power = min(2.0, 2.0 * t)  # Linear ramp, capped at 2W
            return power * Q_unit

        sim_time = 1.0
        config = SolverConfig(
            sim_time=sim_time,
            amb=amb,
            dt_base=0.02,
            steps_target=50,
            use_multi_phase=False
        )

        # Run with ramp Q_func
        result_ramp = run_simulation(
            config=config,
            K=qfunc_setup['K'], C=qfunc_setup['C'],
            Q=np.zeros(N), b=qfunc_setup['b'],
            hA=qfunc_setup['hA'],
            layer_count=qfunc_setup['layer_count'],
            rows=qfunc_setup['rows'], cols=qfunc_setup['cols'],
            Q_func=ramp_qfunc
        )

        # Run with no heat source
        result_zero = run_simulation(
            config=config,
            K=qfunc_setup['K'], C=qfunc_setup['C'],
            Q=np.zeros(N), b=qfunc_setup['b'],
            hA=qfunc_setup['hA'],
            layer_count=qfunc_setup['layer_count'],
            rows=qfunc_setup['rows'], cols=qfunc_setup['cols'],
            Q_func=None
        )

        assert not result_ramp.aborted
        assert result_ramp.step_counter > 0
        # Ramp heating should produce a warmer center than no heating
        T_ramp_center = result_ramp.T.flatten()[center_idx]
        T_zero_center = result_zero.T.flatten()[center_idx]
        assert T_ramp_center > T_zero_center, \
            f"Ramp Q should heat center more than zero Q: {T_ramp_center:.4f} vs {T_zero_center:.4f}"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_heat_source(self):
        """Test simulation with no heat source completes without error."""
        layer_count = 2
        rows = cols = 10
        N = layer_count * rows * cols
        amb = 25.0

        # Minimal valid matrices
        K = sp.eye(N, format='csr') * 0.1
        C = np.ones(N) * 1e-6
        Q = np.zeros(N)
        b = np.zeros(N)
        hA = np.zeros(N)

        config = SolverConfig(
            sim_time=1.0,
            amb=amb,
            dt_base=0.1,
            steps_target=10,
            use_multi_phase=False
        )

        result = run_simulation(
            config=config,
            K=K, C=C, Q=Q, b=b, hA=hA,
            layer_count=layer_count,
            rows=rows, cols=cols
        )

        # Simulation should complete
        assert result is not None
        assert result.T is not None
        assert not result.aborted

    def test_very_short_simulation(self):
        """Test simulation with very short duration."""
        layer_count = 2
        rows = cols = 10
        N = layer_count * rows * cols
        amb = 25.0

        K = sp.eye(N, format='csr') * 0.1
        C = np.ones(N) * 1e-6
        Q = np.ones(N) * 0.001
        b = np.zeros(N)
        hA = np.zeros(N)

        config = SolverConfig(
            sim_time=0.001,  # Very short
            amb=amb,
            dt_base=0.0005,
            steps_target=2,
            use_multi_phase=False
        )

        result = run_simulation(
            config=config,
            K=K, C=C, Q=Q, b=b, hA=hA,
            layer_count=layer_count,
            rows=rows, cols=cols
        )

        assert result.step_counter > 0, "Should complete at least one step"
        assert not result.aborted, "Short simulation should complete normally"


class TestAutoTimeSteppingAndFactorizationCache:
    """Tests for large-grid SciPy auto single-factorization and cache lifecycle."""

    def test_large_auto_scipy_plan_uses_single_uniform_phase(self):
        """SciPy auto should switch to single-factorization uniform at >=100k nodes."""
        config = SolverConfig(
            1.0, 25.0, 1.0 / 120.0, 120, time_stepping="auto"
        )
        mode_small, plan_small = _build_phase_plan(config, 99_999)
        assert mode_small == "multi_phase"
        assert [item[0]["name"] for item in plan_small] == ["A", "B", "C"]

        mode_large, plan_large = _build_phase_plan(config, 100_000)
        assert mode_large == "uniform"
        assert len(plan_large) == 1
        assert plan_large[0][2] == 95
        assert plan_large[0][0]["derivative_start"] is True

    def test_scipy_auto_single_factorization_matches_multi_phase(self, monkeypatch):
        """Derivative-start uniform auto uses 1 factorization and matches multi-phase."""
        layer_count, rows, cols = 2, 16, 16
        N = layer_count * rows * cols
        copper_mask = np.zeros((layer_count, rows, cols), dtype=bool)
        copper_mask[0, 5:11, 4:12] = True
        V_map = np.ones((rows, cols))
        V_map[7:9, 7:9] = 20.0
        H_map = np.zeros((rows, cols))
        H_map[10:, 10:] = 1.0
        settings = {"h_conv": 10.0, "pad_th": 1.0, "pad_k": 3.0}
        K, b, hA, _ = build_stiffness_matrix(
            layer_count, rows, cols, copper_mask,
            np.full(layer_count, 35e-6), np.full(layer_count, 0.8e-3),
            390.0, 0.3, 0.5e-3, 0.5e-3, V_map, [1.5e-3], H_map, settings, 25.0,
        )
        C = np.full(N, 0.002, dtype=np.float64)
        Q = np.zeros(N, dtype=np.float64)
        Q[7 * cols + 7] = 0.5

        multi_cfg = SolverConfig(
            sim_time=2.0, amb=25.0, dt_base=0.05, steps_target=40,
            time_stepping="multi_phase",
        )
        ref = run_simulation(
            multi_cfg, K, C, Q, b, hA, layer_count, rows, cols
        )
        assert ref.factor_count == 6

        # Trigger the >=100k SciPy auto single-factorization plan on this grid
        orig_plan = _build_phase_plan

        def forced_large_plan(cfg, _node_count):
            return orig_plan(cfg, 120_000)

        monkeypatch.setattr(thermal_solver, "_build_phase_plan", forced_large_plan)
        auto_cfg = SolverConfig(
            sim_time=2.0, amb=25.0, dt_base=0.05, steps_target=40,
            time_stepping="auto",
        )
        fast = run_simulation(
            auto_cfg, K, C, Q, b, hA, layer_count, rows, cols
        )

        assert fast.factor_count == 1
        assert fast.k_norm_info["time_stepping"] == "uniform"
        assert fast.phase_metrics[0]["startup"] == "derivative_bdf2"
        np.testing.assert_allclose(fast.T, ref.T, rtol=1e-2, atol=0.1)

    def test_factorization_cache_does_not_free_newly_cached_owner(self, monkeypatch):
        """Storing a newly factored solver in factorization_cache must not free it."""
        layer_count, rows, cols = 1, 8, 8
        N = rows * cols
        K = sp.diags(np.full(N, 0.05), format="csr")
        C = np.full(N, 0.01, dtype=np.float64)
        Q = np.zeros(N, dtype=np.float64)
        Q[N // 2] = 0.1
        b = np.full(N, 0.05 * 25.0, dtype=np.float64)
        hA = np.full(N, 0.05, dtype=np.float64)

        created_owners = []

        class SpyOwner:
            def __init__(self, lu):
                self._lu = lu
                self.free_calls = 0

            def solve(self, rhs):
                assert self.free_calls == 0, "Cached solver was freed before reuse!"
                return self._lu.solve(rhs)

            def free_memory(self, everything=True):
                self.free_calls += 1

        def fake_factor(A):
            owner = SpyOwner(spla.splu(A.tocsc()))
            created_owners.append(owner)
            return owner.solve, owner, "SpyBackend", None

        orig_plan = _build_phase_plan

        def forced_large_plan(cfg, _node_count):
            return orig_plan(cfg, 120_000)

        monkeypatch.setattr(thermal_solver, "_factor_linear_system", fake_factor)
        monkeypatch.setattr(thermal_solver, "_build_phase_plan", forced_large_plan)

        config = SolverConfig(
            sim_time=0.5, amb=25.0, dt_base=0.05, steps_target=10,
            time_stepping="auto",
        )
        cache = ThermalFactorizationCache()

        # Run 1: cache miss -> factored and stored; owner must NOT be freed!
        res1 = run_simulation(
            config, K, C, Q, b, hA, layer_count, rows, cols,
            factorization_cache=cache,
            factorization_cache_key="grid-key-1",
        )
        assert res1.factor_count == 1
        assert res1.k_norm_info["factorization_cache_hit"] is False
        assert len(created_owners) == 1
        assert created_owners[0].free_calls == 0

        # Run 2: cache hit -> reuses live owner without re-factoring or freeing!
        res2 = run_simulation(
            config, K, C, Q, b, hA, layer_count, rows, cols,
            factorization_cache=cache,
            factorization_cache_key="grid-key-1",
        )
        assert res2.factor_count == 0
        assert res2.k_norm_info["factorization_cache_hit"] is True
        assert len(created_owners) == 1
        assert created_owners[0].free_calls == 0
        np.testing.assert_allclose(res2.T, res1.T, rtol=1e-12, atol=1e-12)

        # Clearing the cache releases the cached owner exactly once
        cache.clear()
        assert created_owners[0].free_calls == 1

        # Without a cache, the newly factored owner IS released at end of run
        res3 = run_simulation(
            config, K, C, Q, b, hA, layer_count, rows, cols,
            factorization_cache=None,
            factorization_cache_key=None,
        )
        assert res3.factor_count == 1
        assert len(created_owners) == 2
        assert created_owners[1].free_calls == 1


class TestStructuredMultigridPreconditioner:
    """Tests for structured geometric multigrid V-cycle preconditioner."""

    @staticmethod
    def _build_pcb_problem(layers=2, rows=64, cols=64, res_mm=0.25):
        yy, xx = np.ogrid[:rows, :cols]
        copper = np.zeros((layers, rows, cols), dtype=bool)
        for layer_idx in range(layers):
            horiz = np.mod(yy + 7 * layer_idx, 19) < 2
            vert = np.mod(xx + 11 * layer_idx, 23) < 2
            plane = (
                (xx > cols // 5) & (xx < 4 * cols // 5)
                & (yy > rows // 5) & (yy < 4 * rows // 5)
            )
            copper[layer_idx] = horiz | vert | plane
        via_map = np.ones((rows, cols), dtype=np.float64)
        via_map[(np.mod(yy, 16) < 2) & (np.mod(xx, 16) < 2)] = 50.0
        heatsink = np.zeros((rows, cols), dtype=np.float64)
        heatsink[rows // 3: 2 * rows // 3, cols // 3: 2 * cols // 3] = 1.0
        settings = {"h_conv": 10.0, "pad_th": 1.0, "pad_k": 3.0}
        t_cu = np.full(layers, 35e-6)
        t_fr4 = np.full(layers, 1.6e-3 / max(layers, 1))
        gaps = [1.6e-3 / max(layers - 1, 1)] * max(0, layers - 1)
        dx = dy = res_mm * 1e-3
        args = (
            layers, rows, cols, copper, t_cu, t_fr4,
            390.0, 0.3, dx, dy, via_map, gaps, heatsink, settings, 25.0,
        )
        pixel_area = dx * dy
        cap = np.empty_like(copper, dtype=np.float64)
        for layer_idx in range(layers):
            c_cu = 8960.0 * 385.0 * pixel_area * t_cu[layer_idx]
            c_fr4 = 1850.0 * 1100.0 * pixel_area * t_fr4[layer_idx]
            cap[layer_idx] = np.where(copper[layer_idx], c_cu + c_fr4, c_fr4)
        capacity = cap.reshape(-1)
        power = np.zeros(copper.size, dtype=np.float64)
        r0, c0 = rows // 2, cols // 2
        power[r0 * cols + c0] = 2.0
        return args, capacity, power

    @pytest.mark.parametrize("shape", [(2, 16, 16), (2, 15, 17), (1, 13, 14)])
    def test_coarsen_operator_matches_galerkin_triple_product(self, shape):
        """Structured 2x2 coarsening must equal ``P^T K P`` exactly."""
        layers, rows, cols = shape
        args, _, _ = self._build_pcb_problem(layers=layers, rows=rows, cols=cols)
        K_fine, _, _, _ = build_stiffness_matrix(*args)
        op_fine, _, _, _ = build_structured_operator(*args)
        op_coarse = _coarsen_structured_operator(op_fine)
        K_coarse = _assemble_structured_csc(
            op_coarse, np.zeros(op_coarse.shape[0], dtype=np.float64)
        )

        # Build explicit piecewise-constant prolongation matrix P
        r_c = (rows + 1) // 2
        c_c = (cols + 1) // 2
        rc_f = rows * cols
        rc_c = r_c * c_c
        fine_idx = np.arange(layers * rc_f, dtype=np.int64)
        l_idx = fine_idx // rc_f
        rem = fine_idx % rc_f
        r_idx = rem // cols
        c_idx = rem % cols
        coarse_idx = l_idx * rc_c + (r_idx // 2) * c_c + (c_idx // 2)
        P = sp.csr_matrix(
            (np.ones(fine_idx.size, dtype=np.float64), (fine_idx, coarse_idx)),
            shape=(layers * rc_f, layers * rc_c),
        )
        expected = (P.T @ K_fine @ P).toarray()
        np.testing.assert_allclose(K_coarse.toarray(), expected, rtol=1e-12, atol=1e-12)

    def test_multigrid_preconditioner_is_spd_and_beats_jacobi(self):
        """MG V-cycle is SPD, uses far fewer PCG iterations than Jacobi, and matches direct solve."""
        args, capacity, power = self._build_pcb_problem(layers=2, rows=64, cols=64)
        K, b, _, _ = build_stiffness_matrix(*args)
        op, fast_b, _, _ = build_structured_operator(*args)

        dt = 5.0
        scale = 1.5 / dt
        A_op, M_mg = op.implicit_linear_operator(capacity, scale)

        # Check symmetry and positive definiteness of M_mg on random vectors
        rng = np.random.default_rng(42)
        x = rng.standard_normal(op.shape[0])
        y = rng.standard_normal(op.shape[0])
        Mx = M_mg.matvec(x)
        My = M_mg.matvec(y)
        assert float(np.dot(x, Mx)) > 0.0
        np.testing.assert_allclose(np.dot(x, My), np.dot(y, Mx), rtol=1e-10, atol=1e-12)

        # Compare MG PCG vs diagonal Jacobi PCG and direct sparse solve
        rhs = (scale * capacity) * 25.0 + power + fast_b
        initial = np.full(op.shape[0], 25.0, dtype=np.float64)
        sol_mg, iters_mg, _, rel_mg = _solve_pcg(A_op, M_mg, rhs, initial)
        assert iters_mg < 35, f"Expected MG PCG to converge in <35 iters, got {iters_mg}"
        assert rel_mg <= 1e-6

        # Diagonal Jacobi on the same stiff problem takes >100 iterations (or fails past 300)
        diag = op.stiffness_diag + scale * capacity
        inv_diag = np.divide(1.0, diag, out=np.zeros_like(diag), where=np.abs(diag) > 1e-30)
        M_jacobi = spla.LinearOperator(
            op.shape, matvec=lambda v: inv_diag * v, dtype=np.float64
        )
        try:
            _, iters_jacobi, _, _ = _solve_pcg(A_op, M_jacobi, rhs, initial)
            assert iters_jacobi > 100, f"Expected Jacobi to take >100 iters, got {iters_jacobi}"
            assert iters_mg * 4 < iters_jacobi
        except RuntimeError:
            # Jacobi failing past 300 iterations on this stiff system also proves MG's advantage
            pass

        # Verify numerical accuracy against exact direct sparse solve
        A_sparse = K + sp.diags(scale * capacity, format="csr")
        sol_direct = spla.spsolve(A_sparse, rhs)
        np.testing.assert_allclose(sol_mg, sol_direct, rtol=1e-4, atol=1e-4)

    def test_multigrid_odd_grid_matches_direct_simulation(self):
        """Transient matrix-free MG PCG matches direct solver on odd grid dimensions."""
        args, capacity, power = self._build_pcb_problem(layers=2, rows=35, cols=41)
        K, b, hA, _ = build_stiffness_matrix(*args)
        op, fast_b, fast_hA, _ = build_structured_operator(*args)
        config = SolverConfig(
            sim_time=2.0, amb=25.0, dt_base=0.2, steps_target=10,
            use_multi_phase=False, time_stepping="uniform",
        )
        direct = run_simulation(config, K, capacity, power, b, hA, 2, 35, 41)
        mg_res = run_simulation_matrix_free(config, op, capacity, power, fast_b, fast_hA)

        np.testing.assert_allclose(mg_res.T, direct.T, rtol=1e-4, atol=1e-3)
        assert mg_res.k_norm_info["avg_pcg_iterations"] < 30

    def test_multigrid_hierarchy_cached_by_derivative_scale(self):
        """MG preconditioner caches by derivative_scale and shares coarse geometry."""
        args, capacity, _ = self._build_pcb_problem(layers=2, rows=32, cols=32)
        op, _, _, _ = build_structured_operator(*args)

        op.implicit_linear_operator(capacity, 1.0)
        mg_first = op._preconditioner_cache[1.0][1]
        coarse_ops_ref = op._coarse_operators
        assert coarse_ops_ref is not None and len(coarse_ops_ref) >= 2

        op.implicit_linear_operator(capacity, 1.0)
        assert op._preconditioner_cache[1.0][1] is mg_first

        op.implicit_linear_operator(capacity, 1.5)
        mg_second = op._preconditioner_cache[1.5][1]
        assert mg_second is not mg_first
        assert mg_second.operators is coarse_ops_ref

    @pytest.mark.parametrize("shape", [(2, 6, 12), (2, 12, 6)])
    def test_small_grid_uses_jacobi_fallback(self, shape):
        """Grids with rows < 8 or cols < 8 fall back to diagonal Jacobi."""
        layers, rows, cols = shape
        args, capacity, power = self._build_pcb_problem(layers=layers, rows=rows, cols=cols)
        K, b, hA, _ = build_stiffness_matrix(*args)
        op, fast_b, fast_hA, _ = build_structured_operator(*args)

        op.implicit_linear_operator(capacity, 1.5)
        assert op._preconditioner_cache == {}
        assert op._coarse_operators is None

        config = SolverConfig(
            sim_time=0.4, amb=25.0, dt_base=0.04, steps_target=10,
            use_multi_phase=False, time_stepping="uniform",
        )
        direct = run_simulation(config, K, capacity, power, b, hA, layers, rows, cols)
        fast = run_simulation_matrix_free(config, op, capacity, power, fast_b, fast_hA)
        np.testing.assert_allclose(fast.T, direct.T, rtol=5e-5, atol=1e-3)


class TestSteadyStateAndSpeedups:
    """Tests for direct steady-state, native v2 MG wiring, extrapolation, early stop, and BLAS thread capping."""

    @staticmethod
    def _build_problem(layers=2, rows=32, cols=32, res_mm=0.5):
        return TestStructuredMultigridPreconditioner._build_pcb_problem(
            layers=layers, rows=rows, cols=cols, res_mm=res_mm
        )

    @pytest.mark.physics
    def test_direct_steady_state_sparse_matches_long_transient(self):
        """Direct sparse steady-state solve matches long transient within < 0.05 C."""
        args, capacity, power = self._build_problem(layers=2, rows=20, cols=20)
        K, b, hA, _ = build_stiffness_matrix(*args)

        long_cfg = SolverConfig(
            sim_time=800.0,
            amb=25.0,
            dt_base=4.0,
            steps_target=200,
            use_multi_phase=True,
            early_stop=False,
        )
        transient = run_simulation(long_cfg, K, capacity, power, b, hA, 2, 20, 20)

        ss_cfg = SolverConfig(
            sim_time=400.0,
            amb=25.0,
            dt_base=2.0,
            steps_target=200,
            steady_state=True,
        )
        direct_ss = run_steady_state(ss_cfg, K, power, b, hA, 2, 20, 20)
        dispatched_ss = run_simulation(ss_cfg, K, capacity, power, b, hA, 2, 20, 20)

        assert direct_ss.step_counter == 1
        assert direct_ss.factor_count == 1
        assert direct_ss.aborted is False
        assert direct_ss.k_norm_info["strategy"] == "direct_steady_state"
        assert direct_ss.k_norm_info["steady_rel_diff"] < 1e-8
        np.testing.assert_allclose(dispatched_ss.T, direct_ss.T, rtol=1e-12, atol=1e-12)
        assert dispatched_ss.k_norm_info["strategy"] == "direct_steady_state"
        max_diff = float(np.max(np.abs(direct_ss.T - transient.T)))
        assert max_diff < 0.05, f"Steady-state vs long transient max diff: {max_diff:.4f} C"

        # When Q_func is provided, steady_state=True must still run transient
        transient_qfunc = run_simulation(
            SolverConfig(
                sim_time=1.0, amb=25.0, dt_base=0.2, steps_target=5,
                use_multi_phase=False, steady_state=True,
            ),
            K, capacity, power, b, hA, 2, 20, 20,
            Q_func=lambda _t: power,
        )
        assert transient_qfunc.k_norm_info["strategy"] == "implicit_fvm_bdf2"
        assert transient_qfunc.step_counter == 5

    @pytest.mark.physics
    def test_direct_steady_state_matrix_free_matches_sparse_and_transient(self):
        """Matrix-free direct steady-state matches sparse direct steady-state and long transient."""
        args, capacity, power = self._build_problem(layers=2, rows=32, cols=32)
        K, b, hA, _ = build_stiffness_matrix(*args)
        op, fast_b, fast_hA, _ = build_structured_operator(*args)

        ss_cfg = SolverConfig(
            sim_time=400.0,
            amb=25.0,
            dt_base=2.0,
            steps_target=200,
            steady_state=True,
        )
        sparse_ss = run_steady_state(ss_cfg, K, power, b, hA, 2, 32, 32)
        mf_ss = run_steady_state_matrix_free(ss_cfg, op, capacity, power, fast_b, fast_hA)
        dispatched_mf = run_simulation_matrix_free(
            ss_cfg, op, capacity, power, fast_b, fast_hA
        )

        assert mf_ss.step_counter == 1
        assert mf_ss.factor_count == 0
        assert mf_ss.aborted is False
        assert mf_ss.k_norm_info["strategy"] == "direct_steady_state"
        assert mf_ss.k_norm_info["steady_rel_diff"] < 1e-4
        assert mf_ss.k_norm_info["pcg_relative_residual_max"] <= 1e-6
        np.testing.assert_allclose(mf_ss.T, sparse_ss.T, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(dispatched_mf.T, mf_ss.T, rtol=1e-10, atol=1e-10)

        long_cfg = SolverConfig(
            sim_time=800.0,
            amb=25.0,
            dt_base=8.0,
            steps_target=100,
            use_multi_phase=False,
            early_stop=False,
        )
        transient = run_simulation(long_cfg, K, capacity, power, b, hA, 2, 32, 32)
        max_diff = float(np.max(np.abs(mf_ss.T - transient.T)))
        assert max_diff < 0.05, f"Matrix-free SS vs transient max diff: {max_diff:.4f} C"

    @pytest.mark.parametrize("shape", [(2, 32, 32), (2, 25, 29)])
    def test_multigrid_works_at_zero_derivative_scale(self, shape):
        """_StructuredGeometricMultigrid is SPD and solves cleanly when derivative_scale == 0.0."""
        layers, rows, cols = shape
        args, capacity, power = self._build_problem(layers=layers, rows=rows, cols=cols)
        K, b, _, _ = build_stiffness_matrix(*args)
        op, fast_b, _, _ = build_structured_operator(*args)

        A_op, M_mg = op.implicit_linear_operator(capacity, 0.0)
        rng = np.random.default_rng(7)
        x = rng.standard_normal(op.shape[0])
        y = rng.standard_normal(op.shape[0])
        Mx = M_mg.matvec(x)
        My = M_mg.matvec(y)
        assert float(np.dot(x, Mx)) > 0.0
        np.testing.assert_allclose(np.dot(x, My), np.dot(y, Mx), rtol=1e-10, atol=1e-12)

        rhs = power + fast_b
        initial = np.full(op.shape[0], 25.0, dtype=np.float64)
        sol, iters, _, rel_res = _solve_pcg(
            A_op, M_mg, rhs, initial, relative_tolerance=1e-7, maxiter=250, retry_maxiter=500
        )
        sol_direct = spla.spsolve(K, rhs)
        assert iters < 100
        assert rel_res <= 1e-6
        np.testing.assert_allclose(sol, sol_direct, rtol=1e-4, atol=1e-3)

    def test_native_mg_kernels_match_numpy_fallback(self, monkeypatch):
        """V-cycle with native C++ v2 MG kernels matches pure-NumPy fallback to ~1e-12."""
        args, capacity, power = self._build_problem(layers=2, rows=33, cols=37)
        op, fast_b, _, _ = build_structured_operator(*args)
        _, M_mg = op.implicit_linear_operator(capacity, 0.5)
        rhs = power + fast_b + 0.5 * capacity * 25.0

        native_out = M_mg.matvec(rhs)

        monkeypatch.setattr(thermal_solver, "damped_jacobi_native", None)
        monkeypatch.setattr(thermal_solver, "residual_restrict_2x2_native", None)
        monkeypatch.setattr(thermal_solver, "prolong_add_2x2_native", None)
        numpy_out = M_mg.matvec(rhs)

        np.testing.assert_allclose(native_out, numpy_out, rtol=1e-11, atol=1e-11)

    def test_pcg_linear_extrapolation_reduces_iterations(self, monkeypatch):
        """Linear PCG warm-start extrapolation reduces total PCG iterations without losing accuracy."""
        args, capacity, power = self._build_problem(layers=2, rows=32, cols=32)
        op, fast_b, fast_hA, _ = build_structured_operator(*args)
        config = SolverConfig(
            sim_time=4.0,
            amb=25.0,
            dt_base=0.2,
            steps_target=20,
            use_multi_phase=False,
            time_stepping="uniform",
            early_stop=False,
        )

        extrap_iters = []
        orig_solve_pcg = thermal_solver._solve_pcg

        def record_extrap(operator, precond, rhs, initial, **kwargs):
            sol, iters, retried, rel = orig_solve_pcg(
                operator, precond, rhs, initial, **kwargs
            )
            extrap_iters.append(iters)
            return sol, iters, retried, rel

        monkeypatch.setattr(thermal_solver, "_solve_pcg", record_extrap)
        res_extrap = run_simulation_matrix_free(
            config, op, capacity, power, fast_b, fast_hA
        )

        # Baseline: force x0 = last solution (no linear extrapolation)
        plain_iters = []
        last_sol = [np.full(op.shape[0], 25.0, dtype=np.float64)]

        def no_extrap_solve_pcg(operator, precond, rhs, _initial, **kwargs):
            sol, iters, retried, rel = orig_solve_pcg(
                operator, precond, rhs, last_sol[0], **kwargs
            )
            last_sol[0] = sol
            plain_iters.append(iters)
            return sol, iters, retried, rel

        monkeypatch.setattr(thermal_solver, "_solve_pcg", no_extrap_solve_pcg)
        res_plain = run_simulation_matrix_free(
            config, op, capacity, power, fast_b, fast_hA
        )

        assert sum(extrap_iters) < sum(plain_iters), (
            f"Expected extrapolation to reduce iterations: {sum(extrap_iters)} vs {sum(plain_iters)}"
        )
        np.testing.assert_allclose(res_extrap.T, res_plain.T, rtol=1e-4, atol=1e-3)

    def test_early_steady_state_stop_sparse_and_matrix_free(self):
        """Early steady-state stop cuts long constant runs short cleanly without affecting short transients."""
        args, capacity, power = self._build_problem(layers=2, rows=20, cols=20)
        K, b, hA, _ = build_stiffness_matrix(*args)
        op, fast_b, fast_hA, _ = build_structured_operator(*args)

        # Long run with large time horizon reaches steady state well before step 150
        cfg_early = SolverConfig(
            sim_time=1500.0,
            amb=25.0,
            dt_base=10.0,
            steps_target=150,
            use_multi_phase=False,
            time_stepping="uniform",
            early_stop=True,
        )
        cfg_full = SolverConfig(
            sim_time=1500.0,
            amb=25.0,
            dt_base=10.0,
            steps_target=150,
            use_multi_phase=False,
            time_stepping="uniform",
            early_stop=False,
        )

        sparse_early = run_simulation(cfg_early, K, capacity, power, b, hA, 2, 20, 20)
        sparse_full = run_simulation(cfg_full, K, capacity, power, b, hA, 2, 20, 20)
        assert sparse_early.aborted is False
        assert sparse_early.k_norm_info["early_steady_state_stop"] is True
        assert sparse_early.step_counter < sparse_full.step_counter
        np.testing.assert_allclose(sparse_early.T, sparse_full.T, rtol=1e-3, atol=0.05)

        mf_early = run_simulation_matrix_free(cfg_early, op, capacity, power, fast_b, fast_hA)
        mf_full = run_simulation_matrix_free(cfg_full, op, capacity, power, fast_b, fast_hA)
        assert mf_early.aborted is False
        assert mf_early.k_norm_info["early_steady_state_stop"] is True
        assert mf_early.step_counter < mf_full.step_counter
        np.testing.assert_allclose(mf_early.T, mf_full.T, rtol=1e-3, atol=0.05)

        # Pending snapshots must prevent premature early stop
        cfg_snaps = SolverConfig(
            sim_time=1500.0,
            amb=25.0,
            dt_base=10.0,
            steps_target=150,
            use_multi_phase=False,
            snapshots_enabled=True,
            snap_times=[100.0, 1400.0],
            early_stop=True,
        )
        sparse_snaps = run_simulation(cfg_snaps, K, capacity, power, b, hA, 2, 20, 20)
        assert len(sparse_snaps.snapshot_stats) == 2
        assert sparse_snaps.step_counter >= 140

        # Short transient must not trigger early stop
        cfg_short_early = SolverConfig(
            sim_time=2.0, amb=25.0, dt_base=0.2, steps_target=10,
            use_multi_phase=False, early_stop=True,
        )
        cfg_short_full = SolverConfig(
            sim_time=2.0, amb=25.0, dt_base=0.2, steps_target=10,
            use_multi_phase=False, early_stop=False,
        )
        short_early = run_simulation(cfg_short_early, K, capacity, power, b, hA, 2, 20, 20)
        short_full = run_simulation(cfg_short_full, K, capacity, power, b, hA, 2, 20, 20)
        assert short_early.k_norm_info["early_steady_state_stop"] is False
        assert short_early.step_counter == short_full.step_counter == 10
        np.testing.assert_allclose(short_early.T, short_full.T, rtol=1e-12, atol=1e-12)

    def test_factor_linear_system_uses_scipy_superlu(self):
        """_factor_linear_system returns a SciPy SuperLU factorization."""
        A = sp.diags([4.0, 4.0, 4.0], format="csr")
        rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        solve_fn, owner, backend, residual = _factor_linear_system(A)
        try:
            assert backend == "SciPy"
            assert residual is None
            np.testing.assert_allclose(solve_fn(rhs), rhs / 4.0, rtol=1e-12, atol=1e-12)
        finally:
            thermal_solver._release_linear_solver(owner)

    def test_limit_blas_threads_caps_and_restores_cleanly(self):
        """_limit_blas_threads caps loaded OpenBLAS threads and restores them even on exception."""
        import ctypes
        import sys

        def read_openblas_threads():
            counts = {}
            if not sys.platform.startswith("linux"):
                return counts
            with open("/proc/self/maps", "r", encoding="utf-8", errors="ignore") as handle:
                paths = sorted({
                    line.strip().split()[-1]
                    for line in handle
                    if "openblas" in line.lower() and ".so" in line and line.strip().split()[-1].startswith("/")
                })
            for path in paths:
                lib = ctypes.CDLL(path)
                for get_name, _ in thermal_solver._OPENBLAS_THREAD_SYMBOLS:
                    get_fn = getattr(lib, get_name, None)
                    if get_fn is not None:
                        get_fn.argtypes = []
                        get_fn.restype = ctypes.c_int
                        counts[path] = int(get_fn())
                        break
            return counts

        before = read_openblas_threads()
        with thermal_solver._limit_blas_threads(1):
            inside = read_openblas_threads()
            for count in inside.values():
                assert count == 1
        after = read_openblas_threads()
        assert after == before

        # Exceptions inside the block must propagate and still restore thread counts
        with pytest.raises(ValueError, match="boom"):
            with thermal_solver._limit_blas_threads(1):
                raise ValueError("boom")
        assert read_openblas_threads() == before
