"""Adaptive grid reduction for large ThermalSim models.

The adaptive model keeps the requested fine grid as the geometry reference,
then merges only rectangular regions that contain no material boundary, via,
thermal-pad boundary, or heat-source cell. Conductances and capacities are
summed conservatively so the reduced graph preserves energy balance.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class AdaptiveMesh:
    """Shared 2D leaf mesh used by every copper layer."""

    rows: int
    cols: int
    leaf_map: np.ndarray
    leaves: np.ndarray
    max_cell_ratio: int

    @property
    def leaf_count(self):
        """Return the number of adaptive cells per layer."""
        return int(self.leaves.shape[0])

    @property
    def fine_cell_count(self):
        """Return the equivalent number of fine cells per layer."""
        return int(self.rows * self.cols)

    @property
    def reduction_ratio(self):
        """Return fine-cell count divided by adaptive leaf count."""
        return self.fine_cell_count / max(self.leaf_count, 1)

    def restrict_sum(self, values, layer_count):
        """Conservatively sum a fine-grid field into adaptive leaves."""
        fine = np.asarray(values, dtype=np.float64).reshape(
            layer_count, self.rows * self.cols
        )
        ids = self.leaf_map.reshape(-1)
        reduced = np.empty((layer_count, self.leaf_count), dtype=np.float64)
        for layer_idx in range(layer_count):
            reduced[layer_idx] = np.bincount(
                ids,
                weights=fine[layer_idx],
                minlength=self.leaf_count,
            )
        return reduced.reshape(-1)

    def prolong(self, values, layer_count):
        """Expand adaptive cell values back to the requested fine grid."""
        adaptive = np.asarray(values, dtype=np.float64).reshape(
            layer_count, self.leaf_count
        )
        return adaptive[:, self.leaf_map]


@dataclass
class AdaptiveSparseOperator:
    """Iterative-solver interface for the reduced adaptive graph."""

    matrix: sp.csr_matrix
    layer_count: int
    leaf_count: int
    mesh: AdaptiveMesh

    def __post_init__(self):
        self._preconditioner_cache = {}

    @property
    def rows(self):
        return 1

    @property
    def prefer_derivative_start(self):
        """Adaptive runs avoid a second expensive multigrid setup."""
        return True

    @property
    def cols(self):
        return self.leaf_count

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def nnz_estimate(self):
        return int(self.matrix.nnz)

    def dot(self, vector):
        """Apply the reduced stiffness matrix."""
        return self.matrix.dot(vector)

    def implicit_linear_operator(self, capacity, derivative_scale):
        """Return an iterative representation of ``K + scale*C``."""
        capacity = np.asarray(capacity, dtype=np.float64)
        diagonal = self.matrix.diagonal() + derivative_scale * capacity
        shape = self.shape
        system_matrix = self.matrix + sp.diags(
            derivative_scale * capacity, format="csr"
        )

        def matvec(vector):
            return system_matrix.dot(vector)

        operator = spla.LinearOperator(shape, matvec=matvec, dtype=np.float64)
        cache_key = float(derivative_scale)
        multigrid = self._preconditioner_cache.get(cache_key)
        if multigrid is None:
            multigrid = _GeometricMultigrid(
                system_matrix, self.mesh, self.layer_count
            )
            self._preconditioner_cache = {cache_key: multigrid}
        preconditioner = spla.LinearOperator(
            shape,
            matvec=multigrid.solve,
            dtype=np.float64,
        )
        return operator, preconditioner


class _GeometricMultigrid:
    """Fixed linear V-cycle preconditioner for the adaptive thermal graph."""

    def __init__(self, matrix, mesh, layer_count, max_levels=8):
        self.matrices = [matrix.tocsr()]
        self.prolongations = []
        self.inverse_diagonals = []
        self.omega = 0.72

        leaf_centres_r = 0.5 * (mesh.leaves[:, 0] + mesh.leaves[:, 1])
        leaf_centres_c = 0.5 * (mesh.leaves[:, 2] + mesh.leaves[:, 3])
        centres_r = np.tile(leaf_centres_r, layer_count)
        centres_c = np.tile(leaf_centres_c, layer_count)
        layer_ids = np.repeat(np.arange(layer_count), mesh.leaf_count)
        bucket_size = 2

        for _ in range(max_levels - 1):
            current = self.matrices[-1]
            node_count = current.shape[0]
            diagonal = current.diagonal()
            self.inverse_diagonals.append(np.divide(
                1.0,
                diagonal,
                out=np.zeros_like(diagonal),
                where=np.abs(diagonal) > 1e-30,
            ))
            if node_count <= 1500:
                break

            inverse = None
            for _attempt in range(4):
                row_bucket = np.floor_divide(centres_r.astype(np.int64), bucket_size)
                col_bucket = np.floor_divide(centres_c.astype(np.int64), bucket_size)
                column_span = int(np.max(col_bucket)) + 1
                row_span = int(np.max(row_bucket)) + 1
                keys = (
                    layer_ids.astype(np.int64) * row_span * column_span
                    + row_bucket * column_span
                    + col_bucket
                )
                _, inverse = np.unique(keys, return_inverse=True)
                coarse_count = int(np.max(inverse)) + 1
                if coarse_count <= int(node_count * 0.85):
                    break
                bucket_size *= 2
            if inverse is None or coarse_count >= node_count:
                break

            prolongation = sp.csr_matrix(
                (
                    np.ones(node_count, dtype=np.float64),
                    (np.arange(node_count), inverse),
                ),
                shape=(node_count, coarse_count),
            )
            coarse_matrix = (
                prolongation.T @ current @ prolongation
            ).tocsr()
            self.prolongations.append(prolongation)
            self.matrices.append(coarse_matrix)

            counts = np.bincount(inverse, minlength=coarse_count)
            centres_r = np.bincount(
                inverse, weights=centres_r, minlength=coarse_count
            ) / counts
            centres_c = np.bincount(
                inverse, weights=centres_c, minlength=coarse_count
            ) / counts
            layer_ids = np.rint(np.bincount(
                inverse, weights=layer_ids, minlength=coarse_count
            ) / counts).astype(np.int64)
            bucket_size *= 2

        if len(self.inverse_diagonals) < len(self.matrices):
            diagonal = self.matrices[-1].diagonal()
            self.inverse_diagonals.append(np.divide(
                1.0,
                diagonal,
                out=np.zeros_like(diagonal),
                where=np.abs(diagonal) > 1e-30,
            ))
        try:
            self.coarse_solver = spla.factorized(self.matrices[-1].tocsc())
        except Exception:
            inverse = self.inverse_diagonals[-1]
            self.coarse_solver = lambda rhs: inverse * rhs

    def _smooth(self, level, rhs, initial, iterations=2):
        matrix = self.matrices[level]
        inverse = self.inverse_diagonals[level]
        value = initial
        for _ in range(iterations):
            value = value + self.omega * inverse * (rhs - matrix.dot(value))
        return value

    def _cycle(self, level, rhs):
        if level == len(self.matrices) - 1:
            return np.asarray(self.coarse_solver(rhs), dtype=np.float64)
        value = self._smooth(level, rhs, np.zeros_like(rhs))
        residual = rhs - self.matrices[level].dot(value)
        prolongation = self.prolongations[level]
        coarse_rhs = prolongation.T.dot(residual)
        value += prolongation.dot(self._cycle(level + 1, coarse_rhs))
        return self._smooth(level, rhs, value)

    def solve(self, rhs):
        """Apply one symmetric V-cycle."""
        return self._cycle(0, np.asarray(rhs, dtype=np.float64))


@dataclass
class AdaptiveThermalSystem:
    """Reduced operator and fields ready for the iterative solver."""

    mesh: AdaptiveMesh
    operator: AdaptiveSparseOperator
    capacity: np.ndarray
    power: np.ndarray
    boundary_rhs: np.ndarray
    h_area: np.ndarray

    def wrap_power_function(
        self,
        function: Optional[Callable[[float], np.ndarray]],
        layer_count: int,
    ):
        """Restrict a time-dependent fine-grid source function."""
        if function is None:
            return None
        return lambda time_value: self.mesh.restrict_sum(
            function(time_value), layer_count
        )


def _integral_image(mask):
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)))
    return integral.cumsum(axis=0).cumsum(axis=1)


def _region_sum(integral, r0, r1, c0, c1):
    return (
        integral[r1, c1]
        - integral[r0, c1]
        - integral[r1, c0]
        + integral[r0, c0]
    )


def _split_leaf(leaf):
    r0, r1, c0, c1 = (int(value) for value in leaf)
    height = r1 - r0
    width = c1 - c0
    if height <= 1 and width <= 1:
        return [leaf]
    if height > 1 and width > 1:
        rm = r0 + height // 2
        cm = c0 + width // 2
        return [
            (r0, rm, c0, cm),
            (r0, rm, cm, c1),
            (rm, r1, c0, cm),
            (rm, r1, cm, c1),
        ]
    if height > 1:
        rm = r0 + height // 2
        return [(r0, rm, c0, c1), (rm, r1, c0, c1)]
    cm = c0 + width // 2
    return [(r0, r1, c0, cm), (r0, r1, cm, c1)]


def _leaf_map_from_rectangles(rows, cols, leaves):
    leaf_map = np.empty((rows, cols), dtype=np.int32)
    for leaf_id, (r0, r1, c0, c1) in enumerate(leaves):
        leaf_map[r0:r1, c0:c1] = leaf_id
    return leaf_map


def _balance_rectangles(rows, cols, leaves):
    """Enforce a maximum 2:1 size transition between touching leaves."""
    leaves = list(leaves)
    for _ in range(16):
        leaf_map = _leaf_map_from_rectangles(rows, cols, leaves)
        sizes = np.asarray([
            max(int(r1) - int(r0), int(c1) - int(c0))
            for r0, r1, c0, c1 in leaves
        ])
        pairs = []
        if cols > 1:
            left = leaf_map[:, :-1]
            right = leaf_map[:, 1:]
            changed = left != right
            if np.any(changed):
                pairs.append(np.column_stack((left[changed], right[changed])))
        if rows > 1:
            upper = leaf_map[:-1, :]
            lower = leaf_map[1:, :]
            changed = upper != lower
            if np.any(changed):
                pairs.append(np.column_stack((upper[changed], lower[changed])))
        if not pairs:
            return leaves, leaf_map
        touching = np.unique(np.concatenate(pairs, axis=0), axis=0)
        refine = set()
        for first, second in touching:
            first_size = sizes[first]
            second_size = sizes[second]
            if first_size > 2 * second_size:
                refine.add(int(first))
            elif second_size > 2 * first_size:
                refine.add(int(second))
        if not refine:
            return leaves, leaf_map
        balanced = []
        for leaf_id, leaf in enumerate(leaves):
            balanced.extend(_split_leaf(leaf) if leaf_id in refine else [leaf])
        leaves = balanced
    return leaves, _leaf_map_from_rectangles(rows, cols, leaves)


def build_adaptive_mesh(
    copper_mask,
    via_map,
    heatsink_mask,
    source_mask=None,
    max_cell_ratio=8,
):
    """Build a balanced shared mesh refined at all physical feature edges."""
    copper_mask = np.asarray(copper_mask, dtype=bool)
    _, rows, cols = copper_mask.shape
    max_cell_ratio = max(1, int(max_cell_ratio))
    feature = np.zeros((rows, cols), dtype=bool)

    if cols > 1:
        boundary = np.any(
            copper_mask[:, :, :-1] != copper_mask[:, :, 1:],
            axis=0,
        )
        feature[:, :-1] |= boundary
        feature[:, 1:] |= boundary
    if rows > 1:
        boundary = np.any(
            copper_mask[:, :-1, :] != copper_mask[:, 1:, :],
            axis=0,
        )
        feature[:-1, :] |= boundary
        feature[1:, :] |= boundary
    feature |= np.asarray(via_map) > 1.0
    heatsink = np.asarray(heatsink_mask, dtype=bool)
    feature |= heatsink
    if cols > 1:
        feature[:, 1:] |= heatsink[:, 1:] != heatsink[:, :-1]
    if rows > 1:
        feature[1:, :] |= heatsink[1:, :] != heatsink[:-1, :]
    if source_mask is not None:
        source = np.asarray(source_mask, dtype=bool)
        if source.ndim == 3:
            source = np.any(source, axis=0)
        feature |= source
    feature = ndi.binary_dilation(feature, iterations=1)
    integral = _integral_image(feature)

    leaves = []

    def refine(rectangle):
        r0, r1, c0, c1 = rectangle
        if _region_sum(integral, r0, r1, c0, c1) <= 0:
            leaves.append(rectangle)
            return
        if r1 - r0 <= 1 and c1 - c0 <= 1:
            leaves.append(rectangle)
            return
        for child in _split_leaf(rectangle):
            refine(child)

    for r0 in range(0, rows, max_cell_ratio):
        r1 = min(rows, r0 + max_cell_ratio)
        for c0 in range(0, cols, max_cell_ratio):
            c1 = min(cols, c0 + max_cell_ratio)
            refine((r0, r1, c0, c1))

    leaves, leaf_map = _balance_rectangles(rows, cols, leaves)
    return AdaptiveMesh(
        rows=rows,
        cols=cols,
        leaf_map=leaf_map,
        leaves=np.asarray(leaves, dtype=np.int32),
        max_cell_ratio=max_cell_ratio,
    )


def _aggregate_edges(first, second, conductance, leaf_count):
    if first.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    lower = np.minimum(first, second).astype(np.int64, copy=False)
    upper = np.maximum(first, second).astype(np.int64, copy=False)
    keys = lower * np.int64(leaf_count) + upper
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    summed = np.bincount(inverse, weights=conductance)
    return (
        unique_keys // leaf_count,
        unique_keys % leaf_count,
        summed,
    )


def build_adaptive_system(
    fine_operator,
    capacity,
    power,
    boundary_rhs,
    h_area,
    mesh,
):
    """Aggregate a structured fine-grid operator into an adaptive graph."""
    layer_count = fine_operator.layer_count
    rows = fine_operator.rows
    cols = fine_operator.cols
    leaf_count = mesh.leaf_count
    leaf_map = mesh.leaf_map
    global_first = []
    global_second = []
    global_conductance = []

    for layer_idx in range(layer_count):
        first_parts = []
        second_parts = []
        conductance_parts = []
        if cols > 1:
            left = leaf_map[:, :-1]
            right = leaf_map[:, 1:]
            boundary = left != right
            first_parts.append(left[boundary])
            second_parts.append(right[boundary])
            conductance_parts.append(fine_operator.gx[layer_idx][boundary])
        if rows > 1:
            upper = leaf_map[:-1, :]
            lower = leaf_map[1:, :]
            boundary = upper != lower
            first_parts.append(upper[boundary])
            second_parts.append(lower[boundary])
            conductance_parts.append(fine_operator.gy[layer_idx][boundary])
        if first_parts:
            first, second, conductance = _aggregate_edges(
                np.concatenate(first_parts),
                np.concatenate(second_parts),
                np.concatenate(conductance_parts),
                leaf_count,
            )
            global_first.append(layer_idx * leaf_count + first)
            global_second.append(layer_idx * leaf_count + second)
            global_conductance.append(conductance)

    ids = leaf_map.reshape(-1)
    if layer_count > 1 and fine_operator.gz.size:
        for layer_idx in range(layer_count - 1):
            conductance = np.bincount(
                ids,
                weights=fine_operator.gz[layer_idx].reshape(-1),
                minlength=leaf_count,
            )
            active = conductance > 0.0
            leaves = np.flatnonzero(active)
            global_first.append(layer_idx * leaf_count + leaves)
            global_second.append((layer_idx + 1) * leaf_count + leaves)
            global_conductance.append(conductance[active])

    if global_first:
        first = np.concatenate(global_first).astype(np.int64, copy=False)
        second = np.concatenate(global_second).astype(np.int64, copy=False)
        conductance = np.concatenate(global_conductance).astype(np.float64, copy=False)
    else:
        first = np.empty(0, dtype=np.int64)
        second = np.empty(0, dtype=np.int64)
        conductance = np.empty(0, dtype=np.float64)

    reduced_boundary_diag = mesh.restrict_sum(
        fine_operator.boundary_diag, layer_count
    )
    diagonal = reduced_boundary_diag.copy()
    np.add.at(diagonal, first, conductance)
    np.add.at(diagonal, second, conductance)
    node_count = layer_count * leaf_count
    row_indices = np.concatenate((first, second, np.arange(node_count)))
    col_indices = np.concatenate((second, first, np.arange(node_count)))
    data = np.concatenate((-conductance, -conductance, diagonal))
    matrix = sp.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(node_count, node_count),
    ).tocsr()

    return AdaptiveThermalSystem(
        mesh=mesh,
        operator=AdaptiveSparseOperator(
            matrix, layer_count, leaf_count, mesh
        ),
        capacity=mesh.restrict_sum(capacity, layer_count),
        power=mesh.restrict_sum(power, layer_count),
        boundary_rhs=mesh.restrict_sum(boundary_rhs, layer_count),
        h_area=mesh.restrict_sum(h_area, layer_count),
    )
