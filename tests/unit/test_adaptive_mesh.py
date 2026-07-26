"""Tests for conservative adaptive thermal-grid reduction."""

import numpy as np

from ThermalSim.adaptive_mesh import build_adaptive_mesh, build_adaptive_system
from ThermalSim.thermal_solver import (
    SolverConfig,
    build_structured_operator,
    run_simulation_matrix_free,
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
