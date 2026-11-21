"""Tests for the optional native CPU-core loader and C++ operator."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ThermalSim import native_core, thermal_solver


def test_native_loader_has_safe_python_fallback():
    result = native_core.apply_structured_native(
        1,
        1,
        1,
        np.empty((1, 1, 0)),
        np.empty((1, 0, 1)),
        np.empty((0, 1, 1)),
        np.zeros(1),
        np.ones(1),
    )

    if native_core.HAS_NATIVE_CORE:
        np.testing.assert_array_equal(result, np.zeros(1))
    else:
        assert result is None


def test_native_source_and_build_definition_exist():
    root = Path(native_core.__file__).resolve().parent
    assert (root / "native" / "CMakeLists.txt").is_file()
    assert (root / "native" / "thermalsim_core.cpp").is_file()


@pytest.mark.parametrize(
    "platform,expected_names",
    [
        ("win32", ["thermalsim_core.dll"]),
        ("linux", ["libthermalsim_core.so", "thermalsim_core.so"]),
        ("darwin", ["libthermalsim_core.dylib", "thermalsim_core.dylib"]),
    ],
)
def test_candidate_paths_platform_specific(monkeypatch, platform, expected_names):
    monkeypatch.delenv("THERMALSIM_NATIVE_CORE", raising=False)
    monkeypatch.setattr(native_core.sys, "platform", platform)

    root = Path(native_core.__file__).resolve().parent
    paths = list(native_core._candidate_paths())
    expected = [root / name for name in expected_names] + [
        root / "native" / "bin" / name for name in expected_names
    ]
    assert paths == expected


def test_candidate_paths_respects_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom_core.so"
    monkeypatch.setenv("THERMALSIM_NATIVE_CORE", str(custom))
    paths = list(native_core._candidate_paths())
    assert paths[0] == custom


def test_as_contiguous_f64_fast_path():
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert native_core._as_contiguous_f64(arr) is arr

    strided = arr[:, ::2]
    assert not strided.flags.c_contiguous
    copied_strided = native_core._as_contiguous_f64(strided)
    assert copied_strided is not strided
    assert copied_strided.flags.c_contiguous
    assert copied_strided.dtype == np.float64
    np.testing.assert_array_equal(copied_strided, strided)

    f32 = np.arange(6, dtype=np.float32)
    copied_f32 = native_core._as_contiguous_f64(f32)
    assert copied_f32 is not f32
    assert copied_f32.dtype == np.float64
    assert copied_f32.flags.c_contiguous
    np.testing.assert_array_equal(copied_f32, f32.astype(np.float64))


_HAS_LINUX_GPP = sys.platform.startswith("linux") and shutil.which("g++") is not None


@pytest.fixture(scope="module")
def compiled_native_core(tmp_path_factory):
    if not _HAS_LINUX_GPP:
        pytest.skip("Requires Linux with g++ available")
    root = Path(native_core.__file__).resolve().parent
    src = root / "native" / "thermalsim_core.cpp"
    build_dir = tmp_path_factory.mktemp("native_core_build")
    so_path = build_dir / "libthermalsim_core.so"
    cmd = [
        "g++",
        "-O3",
        "-fno-fast-math",
        "-fopenmp",
        "-shared",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(src),
        "-o",
        str(so_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"g++ compile failed:\n{proc.stderr}"
    return native_core.NativeCore(so_path)


def _numpy_reference_dot(layer_count, rows, cols, gx, gy, gz, boundary, vector):
    op = thermal_solver.StructuredThermalOperator(
        layer_count=layer_count,
        rows=rows,
        cols=cols,
        gx=gx,
        gy=gy,
        gz=gz,
        boundary_diag=boundary,
        stiffness_diag=np.zeros_like(boundary),
    )
    old_has_native = thermal_solver.HAS_NATIVE_CORE
    thermal_solver.HAS_NATIVE_CORE = False
    try:
        return op.dot(vector)
    finally:
        thermal_solver.HAS_NATIVE_CORE = old_has_native


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize(
    "layer_count,rows,cols",
    [
        (1, 1, 1),
        (1, 1, 5),
        (1, 5, 1),
        (3, 1, 1),
        (1, 2, 2),
        (1, 7, 13),
        (2, 8, 9),
        (4, 25, 31),
        (4, 80, 100),
    ],
)
def test_compiled_core_matches_numpy_operator(
    compiled_native_core, layer_count, rows, cols
):
    rng = np.random.default_rng(42 + layer_count * 1000 + rows * 10 + cols)
    gx = rng.uniform(0.1, 5.0, size=(layer_count, rows, max(cols - 1, 0)))
    gy = rng.uniform(0.1, 5.0, size=(layer_count, max(rows - 1, 0), cols))
    gz = rng.uniform(0.05, 2.0, size=(max(layer_count - 1, 0), rows, cols))
    boundary = rng.uniform(0.01, 1.0, size=layer_count * rows * cols)
    vector = rng.normal(25.0, 10.0, size=layer_count * rows * cols)

    expected = _numpy_reference_dot(
        layer_count, rows, cols, gx, gy, gz, boundary, vector
    )
    actual = compiled_native_core.apply_structured(
        layer_count, rows, cols, gx, gy, gz, boundary, vector
    )

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
def test_compiled_core_avoids_copy_and_handles_non_contiguous(
    compiled_native_core, monkeypatch
):
    rng = np.random.default_rng(123)
    layer_count, rows, cols = 2, 6, 7
    gx = rng.uniform(0.1, 2.0, size=(layer_count, rows, cols - 1))
    gy = rng.uniform(0.1, 2.0, size=(layer_count, rows - 1, cols))
    gz = rng.uniform(0.1, 2.0, size=(layer_count - 1, rows, cols))
    boundary = rng.uniform(0.1, 1.0, size=layer_count * rows * cols)
    vector = rng.normal(size=layer_count * rows * cols)

    # Contiguous float64 fast-path should never call np.ascontiguousarray.
    def _forbid_copy(*args, **kwargs):
        raise AssertionError("Unexpected array copy on contiguous float64 inputs")

    with monkeypatch.context() as mp:
        mp.setattr(native_core.np, "ascontiguousarray", _forbid_copy)
        out = compiled_native_core.apply_structured(
            layer_count, rows, cols, gx, gy, gz, boundary, vector
        )

    expected = _numpy_reference_dot(
        layer_count, rows, cols, gx, gy, gz, boundary, vector
    )
    np.testing.assert_allclose(out, expected, rtol=1e-13, atol=1e-13)

    # Non-contiguous float32 vector should still convert cleanly and match.
    padded = np.empty(vector.size * 2, dtype=np.float32)
    padded[::2] = vector.astype(np.float32)
    strided_f32 = padded[::2]
    out_f32 = compiled_native_core.apply_structured(
        layer_count, rows, cols, gx, gy, gz, boundary, strided_f32
    )
    expected_f32 = _numpy_reference_dot(
        layer_count,
        rows,
        cols,
        gx,
        gy,
        gz,
        boundary,
        strided_f32.astype(np.float64),
    )
    np.testing.assert_allclose(out_f32, expected_f32, rtol=1e-13, atol=1e-13)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
def test_compiled_core_rejects_invalid_dimensions(compiled_native_core):
    empty = np.empty(0, dtype=np.float64)
    ones = np.ones(1, dtype=np.float64)
    with pytest.raises(RuntimeError):
        compiled_native_core.apply_structured(0, 1, 1, empty, empty, empty, ones, ones)
    with pytest.raises(RuntimeError):
        compiled_native_core.apply_structured(1, 1, 2, empty, empty, empty, np.ones(2), np.ones(2))


def _make_random_problem(layer_count, rows, cols, seed):
    rng = np.random.default_rng(seed)
    gx = rng.uniform(0.1, 5.0, size=(layer_count, rows, max(cols - 1, 0)))
    gy = rng.uniform(0.1, 5.0, size=(layer_count, max(rows - 1, 0), cols))
    gz = rng.uniform(0.05, 2.0, size=(max(layer_count - 1, 0), rows, cols))
    node_count = layer_count * rows * cols
    diag_mass = rng.uniform(0.05, 1.5, size=node_count)
    # Build true matrix diagonal for realistic omega * inv(diag)
    diag = diag_mass.reshape(layer_count, rows, cols).copy()
    if cols > 1:
        diag[:, :, :-1] += gx
        diag[:, :, 1:] += gx
    if rows > 1:
        diag[:, :-1, :] += gy
        diag[:, 1:, :] += gy
    if layer_count > 1:
        diag[:-1, :, :] += gz
        diag[1:, :, :] += gz
    omega_inv_diag = np.ascontiguousarray(0.72 / diag.reshape(-1), dtype=np.float64)
    rhs = rng.normal(5.0, 3.0, size=node_count)
    x0 = rng.normal(25.0, 8.0, size=node_count)
    return gx, gy, gz, diag_mass, omega_inv_diag, rhs, x0


def _numpy_reference_damped_jacobi(
    layer_count, rows, cols, gx, gy, gz, diag_mass, omega_inv_diag, rhs, x0, iterations, zero_init
):
    if zero_init:
        if iterations == 0:
            return np.zeros_like(rhs, dtype=np.float64)
        value = omega_inv_diag * rhs
        remaining = iterations - 1
    else:
        value = np.array(x0, dtype=np.float64, copy=True)
        remaining = iterations
    for _ in range(remaining):
        ax = _numpy_reference_dot(layer_count, rows, cols, gx, gy, gz, diag_mass, value)
        value = value + omega_inv_diag * (rhs - ax)
    return value


def _numpy_reference_prolong_add(layer_count, rows, cols, coarse_x, fine_x):
    out = np.array(fine_x, dtype=np.float64, copy=True)
    coarse_rows = (rows + 1) // 2
    coarse_cols = (cols + 1) // 2
    coarse_view = coarse_x.reshape(layer_count, coarse_rows, coarse_cols)
    value_view = out.reshape(layer_count, rows, cols)
    value_view[:, 0::2, 0::2] += coarse_view
    if cols > 1:
        value_view[:, 0::2, 1::2] += coarse_view[:, :, : cols // 2]
    if rows > 1:
        value_view[:, 1::2, 0::2] += coarse_view[:, : rows // 2, :]
    if rows > 1 and cols > 1:
        value_view[:, 1::2, 1::2] += coarse_view[:, : rows // 2, : cols // 2]
    return out


_MG_SHAPES = [
    (1, 1, 1),
    (1, 1, 5),
    (1, 5, 1),
    (3, 1, 1),
    (2, 7, 9),
    (4, 32, 33),
    (4, 64, 64),
    (4, 129, 131),
]


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
def test_compiled_core_reports_v2_and_mg_support(compiled_native_core):
    assert compiled_native_core.version == 2
    assert compiled_native_core.has_mg is True


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _MG_SHAPES)
@pytest.mark.parametrize("zero_init,iterations", [(True, 1), (True, 2), (False, 2), (False, 0), (True, 0)])
def test_compiled_damped_jacobi_matches_numpy(
    compiled_native_core, layer_count, rows, cols, zero_init, iterations
):
    gx, gy, gz, diag_mass, omega_inv_diag, rhs, x0 = _make_random_problem(
        layer_count, rows, cols, seed=100 + layer_count * 100 + rows * 7 + cols
    )
    expected = _numpy_reference_damped_jacobi(
        layer_count, rows, cols, gx, gy, gz, diag_mass, omega_inv_diag, rhs, x0, iterations, zero_init
    )

    if zero_init:
        actual = compiled_native_core.damped_jacobi(
            layer_count,
            rows,
            cols,
            gx,
            gy,
            gz,
            diag_mass,
            omega_inv_diag,
            rhs,
            None,
            iterations,
            zero_init=True,
        )
    else:
        buf = x0.copy()
        actual = compiled_native_core.damped_jacobi(
            layer_count,
            rows,
            cols,
            gx,
            gy,
            gz,
            diag_mass,
            omega_inv_diag,
            rhs,
            buf,
            iterations,
            zero_init=False,
        )
        assert actual is buf

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _MG_SHAPES + [(1, 3, 1050)])
def test_compiled_residual_restrict_2x2_matches_numpy(
    compiled_native_core, layer_count, rows, cols
):
    gx, gy, gz, diag_mass, _, rhs, x0 = _make_random_problem(
        layer_count, rows, cols, seed=200 + layer_count * 100 + rows * 7 + cols
    )
    ax = _numpy_reference_dot(layer_count, rows, cols, gx, gy, gz, diag_mass, x0)
    expected = thermal_solver._restrict_2x2(
        (rhs - ax).reshape(layer_count, rows, cols)
    ).reshape(-1)

    actual = compiled_native_core.residual_restrict_2x2(
        layer_count, rows, cols, gx, gy, gz, diag_mass, rhs, x0
    )

    assert actual.shape == expected.shape
    assert actual.flags.c_contiguous
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _MG_SHAPES)
def test_compiled_prolong_add_2x2_matches_numpy(
    compiled_native_core, layer_count, rows, cols
):
    rng = np.random.default_rng(300 + layer_count * 100 + rows * 7 + cols)
    coarse_rows = (rows + 1) // 2
    coarse_cols = (cols + 1) // 2
    coarse_x = rng.normal(1.5, 4.0, size=layer_count * coarse_rows * coarse_cols)
    fine_x = rng.normal(25.0, 5.0, size=layer_count * rows * cols)

    expected = _numpy_reference_prolong_add(layer_count, rows, cols, coarse_x, fine_x)
    actual = fine_x.copy()
    compiled_native_core.prolong_add_2x2(layer_count, rows, cols, coarse_x, actual)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
def test_compiled_mg_module_helpers_and_zero_copy(compiled_native_core, monkeypatch):
    layer_count, rows, cols = 2, 7, 9
    gx, gy, gz, diag_mass, omega_inv_diag, rhs, x0 = _make_random_problem(
        layer_count, rows, cols, seed=404
    )
    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)

    def _forbid_copy(*args, **kwargs):
        raise AssertionError("Unexpected array copy on contiguous float64 inputs")

    with monkeypatch.context() as mp:
        mp.setattr(native_core.np, "ascontiguousarray", _forbid_copy)
        val = native_core.damped_jacobi_native(
            layer_count,
            rows,
            cols,
            gx,
            gy,
            gz,
            diag_mass,
            omega_inv_diag,
            rhs,
            None,
            2,
            zero_init=True,
        )
        assert val is not None
        coarse_rhs = native_core.residual_restrict_2x2_native(
            layer_count, rows, cols, gx, gy, gz, diag_mass, rhs, val
        )
        assert coarse_rhs is not None
        ok = native_core.prolong_add_2x2_native(
            layer_count, rows, cols, coarse_rhs * 0.1, val
        )
        assert ok is True
        val2 = native_core.damped_jacobi_native(
            layer_count,
            rows,
            cols,
            gx,
            gy,
            gz,
            diag_mass,
            omega_inv_diag,
            rhs,
            val,
            2,
            zero_init=False,
        )
        assert val2 is val

    # Reference verification of the same sequence
    ref = _numpy_reference_damped_jacobi(
        layer_count, rows, cols, gx, gy, gz, diag_mass, omega_inv_diag, rhs, x0, 2, True
    )
    ref_ax = _numpy_reference_dot(layer_count, rows, cols, gx, gy, gz, diag_mass, ref)
    ref_coarse = thermal_solver._restrict_2x2(
        (rhs - ref_ax).reshape(layer_count, rows, cols)
    ).reshape(-1)
    ref = _numpy_reference_prolong_add(layer_count, rows, cols, ref_coarse * 0.1, ref)
    ref = _numpy_reference_damped_jacobi(
        layer_count, rows, cols, gx, gy, gz, diag_mass, omega_inv_diag, rhs, ref, 2, False
    )
    np.testing.assert_allclose(val2, ref, rtol=1e-13, atol=1e-13)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
def test_compiled_mg_rejects_invalid_arguments(compiled_native_core):
    empty = np.empty(0, dtype=np.float64)
    ones = np.ones(1, dtype=np.float64)
    with pytest.raises(ValueError):
        compiled_native_core.damped_jacobi(
            1, 1, 1, empty, empty, empty, ones, ones, ones, None, 2, zero_init=False
        )
    with pytest.raises(ValueError):
        compiled_native_core.damped_jacobi(
            1, 1, 1, empty, empty, empty, ones, ones, ones, ones.copy(), -1, zero_init=False
        )
    with pytest.raises(RuntimeError):
        compiled_native_core.residual_restrict_2x2(
            1, 1, 2, empty, empty, empty, np.ones(2), np.ones(2), np.ones(2)
        )
    with pytest.raises(RuntimeError):
        compiled_native_core.prolong_add_2x2(1, 2, 2, np.ones(2), np.ones(4))


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
def test_v1_library_loads_cleanly_with_mg_fallback(tmp_path, monkeypatch):
    v1_src = tmp_path / "v1_stub.cpp"
    v1_so = tmp_path / "libthermalsim_core_v1.so"
    v1_src.write_text(
        """
        extern "C" __attribute__((visibility("default")))
        int thermalsim_core_version() { return 1; }

        extern "C" __attribute__((visibility("default")))
        int thermalsim_apply_structured(
            int layer_count, int rows, int cols,
            const double*, const double*, const double*,
            const double* boundary_diag, const double* input, double* output
        ) {
            if (layer_count <= 0 || rows <= 0 || cols <= 0 || !boundary_diag || !input || !output) {
                return 1;
            }
            int n = layer_count * rows * cols;
            for (int i = 0; i < n; ++i) {
                output[i] = boundary_diag[i] * input[i];
            }
            return 0;
        }
        """,
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["g++", "-O2", "-shared", "-fPIC", str(v1_src), "-o", str(v1_so)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"v1 stub compile failed:\n{proc.stderr}"

    v1_core = native_core.NativeCore(v1_so)
    assert v1_core.version == 1
    assert v1_core.has_mg is False

    empty = np.empty(0, dtype=np.float64)
    out = v1_core.apply_structured(
        1, 1, 1, empty, empty, empty, np.array([2.0]), np.array([3.5])
    )
    np.testing.assert_array_equal(out, np.array([7.0]))

    monkeypatch.setattr(native_core, "NATIVE_CORE", v1_core)
    assert (
        native_core.damped_jacobi_native(
            1, 1, 1, empty, empty, empty, np.ones(1), np.ones(1), np.ones(1), None, 2, zero_init=True
        )
        is None
    )
    assert (
        native_core.residual_restrict_2x2_native(
            1, 1, 1, empty, empty, empty, np.ones(1), np.ones(1), np.ones(1)
        )
        is None
    )
    fine = np.zeros(1, dtype=np.float64)
    assert native_core.prolong_add_2x2_native(1, 1, 1, np.ones(1), fine) is False
    np.testing.assert_array_equal(fine, np.zeros(1))
