"""Tests for conservative adaptive thermal-grid reduction."""

import numpy as np

from ThermalSim.adaptive_mesh import (
    build_adaptive_mesh,
    build_adaptive_system,
    restrict_heatsink_model,
)
from ThermalSim.thermal_solver import (
    SolverConfig,
    build_heatsink_model,
    build_stiffness_matrix,
    build_structured_operator,
    run_simulation_matrix_free,
    run_steady_state,
    run_steady_state_matrix_free,
)


def _fine_system(rows=16, cols=16, layers=2):
    copper = np.zeros((layers, rows, cols), dtype=bool)
    settings = {"pad_th": 1.0, "pad_k": 3.0, "h_conv": 10.0}
    operator, boundary, h_area, _ = build_structured_operator(
        layer_count=layers,
        rows=rows,
        cols=cols,
        copper_mask=copper,
        t_cu=np.full(layers, 35e-6),
        t_fr4_eff=np.full(layers, 0.8e-3),
        k_cu=390.0,
        k_fr4=0.3,
        dx=0.1e-3,
        dy=0.1e-3,
        V_map=np.ones((rows, cols)),
        gap_m=[1.5e-3] * max(0, layers - 1),
        H_map=np.zeros((rows, cols)),
        settings=settings,
        amb=25.0,
    )
    return copper, operator, boundary, h_area


def test_homogeneous_grid_is_reduced():
    copper, _, _, _ = _fine_system()
    mesh = build_adaptive_mesh(
        copper,
        np.ones(copper.shape[1:]),
        np.zeros(copper.shape[1:], dtype=bool),
        max_cell_ratio=8,
    )

    assert mesh.leaf_count == 4
    assert mesh.reduction_ratio == 64.0


def test_material_boundary_remains_fine():
    copper, _, _, _ = _fine_system()
    copper[0, :, 7:9] = True
    mesh = build_adaptive_mesh(
        copper,
        np.ones(copper.shape[1:]),
        np.zeros(copper.shape[1:], dtype=bool),
        max_cell_ratio=8,
    )

    boundary_leaf_ids = np.unique(mesh.leaf_map[:, 6:10])
    boundary_leaves = mesh.leaves[boundary_leaf_ids]
    assert np.all(boundary_leaves[:, 1] - boundary_leaves[:, 0] <= 2)
    assert np.all(boundary_leaves[:, 3] - boundary_leaves[:, 2] <= 2)
    assert mesh.leaf_count < mesh.fine_cell_count


def test_restriction_preserves_total_energy_and_power():
    copper, operator, boundary, h_area = _fine_system()
    source_mask = np.zeros(copper.shape, dtype=bool)
    source_mask[0, 3:5, 3:5] = True
    mesh = build_adaptive_mesh(
        copper,
        np.ones(copper.shape[1:]),
        np.zeros(copper.shape[1:], dtype=bool),
        source_mask=source_mask,
        max_cell_ratio=8,
    )
    node_count = copper.size
    capacity = np.linspace(0.001, 0.003, node_count)
    power = np.zeros(node_count)
    power[np.flatnonzero(source_mask.reshape(-1))] = 0.25
    system = build_adaptive_system(
        operator, capacity, power, boundary, h_area, mesh
    )

    np.testing.assert_allclose(np.sum(system.capacity), np.sum(capacity))
    np.testing.assert_allclose(np.sum(system.power), np.sum(power))
    np.testing.assert_allclose(np.sum(system.boundary_rhs), np.sum(boundary))
    np.testing.assert_allclose(np.sum(system.h_area), np.sum(h_area))


def test_adaptive_operator_preserves_uniform_temperature_balance():
    copper, operator, boundary, h_area = _fine_system()
    mesh = build_adaptive_mesh(
        copper,
        np.ones(copper.shape[1:]),
        np.zeros(copper.shape[1:], dtype=bool),
        max_cell_ratio=8,
    )
    capacity = np.ones(copper.size)
    power = np.zeros(copper.size)
    system = build_adaptive_system(
        operator, capacity, power, boundary, h_area, mesh
    )
    uniform = np.full(system.operator.shape[0], 25.0)

    np.testing.assert_allclose(
        system.operator.dot(uniform),
        system.boundary_rhs,
        rtol=1e-12,
        atol=1e-12,
    )


def test_adaptive_temperature_stays_within_engineering_tolerance():
    """Adaptive and requested-grid peaks should agree within one degree."""
    rows = cols = 32
    layers = 2
    copper = np.zeros((layers, rows, cols), dtype=bool)
    copper[0, 14:18, 3:29] = True
    copper[1, 6:26, 8:24] = True
    via_map = np.ones((rows, cols))
    via_map[15:17, 15:17] = 30.0
    heatsink = np.zeros((rows, cols), dtype=bool)
    settings = {"pad_th": 1.0, "pad_k": 3.0, "h_conv": 10.0}
    operator, boundary, h_area, _ = build_structured_operator(
        layers, rows, cols, copper,
        np.full(layers, 35e-6), np.full(layers, 0.8e-3),
        390.0, 0.3, 0.25e-3, 0.25e-3,
        via_map, [1.5e-3], heatsink.astype(float), settings, 25.0,
    )
    capacity = np.full(copper.size, 0.002, dtype=np.float64)
    power = np.zeros(copper.size, dtype=np.float64)
    source_mask = np.zeros_like(copper)
    source_mask[0, 15:17, 15:17] = True
    power[np.flatnonzero(source_mask.reshape(-1))] = 0.05
    mesh = build_adaptive_mesh(
        copper, via_map, heatsink, source_mask, max_cell_ratio=4
    )
    adaptive = build_adaptive_system(
        operator, capacity, power, boundary, h_area, mesh
    )
    config = SolverConfig(
        sim_time=0.2,
        amb=25.0,
        dt_base=0.01,
        steps_target=20,
        use_multi_phase=False,
        time_stepping="uniform",
    )

    fine_result = run_simulation_matrix_free(
        config, operator, capacity, power, boundary, h_area
    )
    adaptive_result = run_simulation_matrix_free(
        config,
        adaptive.operator,
        adaptive.capacity,
        adaptive.power,
        adaptive.boundary_rhs,
        adaptive.h_area,
    )
    prolonged = mesh.prolong(adaptive_result.T.reshape(-1), layers)

    assert abs(float(np.max(prolonged)) - float(np.max(fine_result.T))) <= 1.0
    fine_energy = float(np.sum(capacity * (fine_result.T.reshape(-1) - 25.0)))
    adaptive_energy = float(np.sum(
        adaptive.capacity * (adaptive_result.T.reshape(-1) - 25.0)
    ))
    assert abs(adaptive_energy - fine_energy) / max(abs(fine_energy), 1e-12) <= 0.01


def test_coarse_block_conductance_matches_exact_finite_volume():
    """Conductance between two 8x8 homogeneous leaves equals one fine-cell conductance."""
    copper, operator, boundary, h_area = _fine_system(rows=16, cols=16, layers=1)
    mesh = build_adaptive_mesh(
        copper,
        np.ones((16, 16)),
        np.zeros((16, 16), dtype=bool),
        max_cell_ratio=8,
    )
    assert mesh.leaf_count == 4
    system = build_adaptive_system(
        operator,
        np.ones(copper.size),
        np.zeros(copper.size),
        boundary,
        h_area,
        mesh,
    )
    left_id = int(mesh.leaf_map[0, 0])
    right_id = int(mesh.leaf_map[0, 8])
    actual_g = float(-system.operator.matrix[left_id, right_id])
    expected_g = float(operator.gx[0, 0, 7])
    np.testing.assert_allclose(actual_g, expected_g, rtol=1e-12)


def test_adaptive_copper_plane_steady_state_accuracy():
    """64x64 homogeneous 1-layer copper plane error vs fine direct grid stays < 2 C."""
    from ThermalSim.thermal_solver import build_stiffness_matrix, run_simulation

    rows = cols = 64
    layers = 1
    copper = np.ones((layers, rows, cols), dtype=bool)
    via_map = np.ones((rows, cols))
    heatsink = np.zeros((rows, cols), dtype=bool)
    dx = dy = 0.5e-3
    t_cu = np.array([35e-6])
    t_fr4_eff = np.array([1.6e-3])
    settings = {"pad_th": 1.0, "pad_k": 3.0, "h_conv": 20.0}
    amb = 25.0

    K_fine, b_fine, hA_fine, _ = build_stiffness_matrix(
        layers, rows, cols, copper, t_cu, t_fr4_eff,
        390.0, 0.3, dx, dy, via_map, [], heatsink.astype(float), settings, amb,
    )
    operator, b_op, hA_op, _ = build_structured_operator(
        layers, rows, cols, copper, t_cu, t_fr4_eff,
        390.0, 0.3, dx, dy, via_map, [], heatsink.astype(float), settings, amb,
    )

    pixel_area = dx * dy
    c_cell = 8960.0 * 385.0 * pixel_area * t_cu[0] + 1850.0 * 1100.0 * pixel_area * t_fr4_eff[0]
    capacity = np.full(copper.size, c_cell, dtype=np.float64)
    source_mask = np.zeros_like(copper)
    source_mask[0, :, :2] = True
    power = np.zeros(copper.size, dtype=np.float64)
    power[np.flatnonzero(source_mask.reshape(-1))] = 2.0 / float(np.sum(source_mask))

    mesh = build_adaptive_mesh(
        copper, via_map, heatsink, source_mask=source_mask, max_cell_ratio=8
    )
    assert mesh.reduction_ratio > 5.0
    adaptive = build_adaptive_system(
        operator, capacity, power, b_op, hA_op, mesh
    )

    config = SolverConfig(
        sim_time=60.0,
        amb=amb,
        dt_base=1.0,
        steps_target=60,
        use_multi_phase=False,
        time_stepping="uniform",
    )
    fine_res = run_simulation(
        config, K_fine, capacity, power, b_fine, hA_fine, layers, rows, cols
    )
    adaptive_res = run_simulation_matrix_free(
        config,
        adaptive.operator,
        adaptive.capacity,
        adaptive.power,
        adaptive.boundary_rhs,
        adaptive.h_area,
    )
    prolonged = mesh.prolong(adaptive_res.T.reshape(-1), layers)
    leaf_areas = (
        (mesh.leaves[:, 1] - mesh.leaves[:, 0])
        * (mesh.leaves[:, 3] - mesh.leaves[:, 2])
    ).astype(np.float64)
    fine_leaf_avg = mesh.restrict_sum(fine_res.T.reshape(-1), layers) / leaf_areas
    peak_err = abs(float(np.max(prolonged)) - float(np.max(fine_res.T)))
    min_err = abs(float(np.min(prolonged)) - float(np.min(fine_res.T)))
    leaf_avg_err = float(np.max(np.abs(adaptive_res.T.reshape(-1) - fine_leaf_avg)))

    assert peak_err < 2.0, f"Peak temperature error too large: {peak_err:.3f} C"
    assert min_err < 2.0, f"Min temperature error too large: {min_err:.3f} C"
    assert leaf_avg_err < 2.0, f"Leaf-averaged error too large: {leaf_avg_err:.3f} C"


def test_full_board_heatsink_achieves_strong_reduction():
    """Full-board heatsink_mask should coarsen uniformly with reduction_ratio > 10."""
    rows = cols = 64
    copper = np.ones((2, rows, cols), dtype=bool)
    via_map = np.ones((rows, cols))
    heatsink = np.ones((rows, cols), dtype=bool)

    mesh = build_adaptive_mesh(copper, via_map, heatsink, max_cell_ratio=8)

    assert mesh.reduction_ratio > 10.0
    assert mesh.leaf_count == 64


def test_partial_heatsink_boundary_remains_fine_while_interior_coarsens():
    """Heatsink boundaries should stay fine on both sides while interior coarsens."""
    rows = cols = 64
    copper = np.ones((1, rows, cols), dtype=bool)
    via_map = np.ones((rows, cols))
    heatsink = np.zeros((rows, cols), dtype=bool)
    heatsink[16:48, 16:48] = True

    mesh = build_adaptive_mesh(copper, via_map, heatsink, max_cell_ratio=8)

    # Both sides of heatsink edges must remain 1x1
    left_edge_ids = np.unique(mesh.leaf_map[16:48, 15:17])
    right_edge_ids = np.unique(mesh.leaf_map[16:48, 47:49])
    top_edge_ids = np.unique(mesh.leaf_map[15:17, 16:48])
    bot_edge_ids = np.unique(mesh.leaf_map[47:49, 16:48])
    edge_ids = np.unique(
        np.concatenate((left_edge_ids, right_edge_ids, top_edge_ids, bot_edge_ids))
    )
    edge_leaves = mesh.leaves[edge_ids]
    assert np.all(edge_leaves[:, 1] - edge_leaves[:, 0] == 1)
    assert np.all(edge_leaves[:, 3] - edge_leaves[:, 2] == 1)

    # Interior far from edges should coarsen beyond 1x1
    interior_ids = np.unique(mesh.leaf_map[24:40, 24:40])
    interior_leaves = mesh.leaves[interior_ids]
    assert np.max(interior_leaves[:, 1] - interior_leaves[:, 0]) >= 4
    assert np.max(interior_leaves[:, 3] - interior_leaves[:, 2]) >= 4
    assert mesh.reduction_ratio > 2.0


def test_restrict_heatsink_model_preserves_conductance_and_matches_steady_state():
    """restrict_heatsink_model preserves sum(g) and matches exact augmented adaptive solve."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    assert restrict_heatsink_model(None, None, 2) is None

    layers, rows, cols = 2, 32, 32
    copper = np.ones((layers, rows, cols), dtype=bool)
    via_map = np.ones((rows, cols), dtype=np.float64)
    via_map[14:18, 14:18] = 20.0
    heatsink = np.zeros((rows, cols), dtype=bool)
    heatsink[8:24, 8:24] = True
    dx = dy = 0.5e-3
    t_cu = np.full(layers, 35e-6)
    t_fr4_eff = np.full(layers, 0.8e-3)
    settings = {
        "h_conv": 10.0,
        "pad_th": 1.0,
        "pad_k": 3.0,
        "pad_contact_factor": 0.3,
        "use_heatsink": True,
        "pad_sink_mode": "heatsink",
        "heatsink_rth_kw": 3.5,
        "heatsink_material": "aluminium",
        "heatsink_mass_g": 40.0,
    }
    amb = 25.0

    operator, b_op, hA_op, _ = build_structured_operator(
        layers, rows, cols, copper, t_cu, t_fr4_eff,
        390.0, 0.3, dx, dy, via_map, [1.0e-3], heatsink.astype(float), settings, amb,
    )
    K_fine, b_fine, hA_fine, _ = build_stiffness_matrix(
        layers, rows, cols, copper, t_cu, t_fr4_eff,
        390.0, 0.3, dx, dy, via_map, [1.0e-3], heatsink.astype(float), settings, amb,
    )
    capacity = np.full(copper.size, 0.002, dtype=np.float64)
    source_mask = np.zeros_like(copper)
    source_mask[0, 15:17, 15:17] = True
    power = np.zeros(copper.size, dtype=np.float64)
    power[np.flatnonzero(source_mask.reshape(-1))] = 0.5

    mesh = build_adaptive_mesh(
        copper, via_map, heatsink, source_mask=source_mask, max_cell_ratio=8
    )
    adaptive = build_adaptive_system(
        operator, capacity, power, b_op, hA_op, mesh
    )

    hs_fine = build_heatsink_model(
        layers, rows, cols, dx, dy, heatsink.astype(float), settings, amb
    )
    hs_adaptive = restrict_heatsink_model(hs_fine, mesh, layers)
    assert hs_adaptive is not None
    np.testing.assert_allclose(
        np.sum(hs_adaptive.pad_conductance),
        np.sum(hs_fine.pad_conductance),
        rtol=1e-12,
    )
    assert hs_adaptive.rth_kw == hs_fine.rth_kw
    assert hs_adaptive.capacity_j_k == hs_fine.capacity_j_k
    assert hs_adaptive.initial_temp_c == hs_fine.initial_temp_c
    assert hs_adaptive.mode == hs_fine.mode

    # Compare adaptive MG-PCG steady-state against augmented direct solve on adaptive matrix
    g_a = hs_adaptive.pad_conductance
    g_hs_a = hs_adaptive.g_hs_amb
    A_aug = sp.bmat(
        [
            [adaptive.operator.matrix, sp.csc_matrix(-g_a.reshape(-1, 1))],
            [sp.csc_matrix(-g_a.reshape(1, -1)), sp.csc_matrix([[g_hs_a + np.sum(g_a)]])],
        ],
        format="csc",
    )
    rhs_aug = np.concatenate((adaptive.power + adaptive.boundary_rhs, [g_hs_a * amb]))
    sol_aug = spla.spsolve(A_aug, rhs_aug)
    T_adapt_truth = sol_aug[:-1]
    T_hs_truth = float(sol_aug[-1])

    ss_cfg = SolverConfig(
        sim_time=10.0, amb=amb, dt_base=1.0, steps_target=10, steady_state=True
    )
    res_adapt = run_steady_state_matrix_free(
        ss_cfg,
        adaptive.operator,
        adaptive.capacity,
        adaptive.power,
        adaptive.boundary_rhs,
        adaptive.h_area,
        heatsink_model=hs_adaptive,
    )
    res_fine = run_steady_state(
        ss_cfg, K_fine, power, b_fine, hA_fine, layers, rows, cols, heatsink_model=hs_fine
    )

    assert float(np.max(np.abs(res_adapt.T.reshape(-1) - T_adapt_truth))) < 1e-5
    assert abs(res_adapt.k_norm_info["heatsink_temp_c"] - T_hs_truth) < 1e-5
    assert abs(
        res_adapt.k_norm_info["heatsink_temp_c"] - res_fine.k_norm_info["heatsink_temp_c"]
    ) < 0.2
