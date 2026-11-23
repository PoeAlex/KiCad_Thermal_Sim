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
def test_compiled_core_reports_v4_and_pcg_support(compiled_native_core):
    assert compiled_native_core.version == 4
    assert compiled_native_core.has_mg is True
    assert compiled_native_core.has_compact is True
    assert compiled_native_core.has_pcg is True
    assert native_core.HAS_NATIVE_CORE is True
    assert native_core.HAS_NATIVE_MG is True
    assert native_core.HAS_NATIVE_COMPACT is True
    assert native_core.HAS_NATIVE_PCG is True


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
    assert v1_core.has_compact is False
    assert v1_core.has_pcg is False

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

    # v3 compact/f32 helpers also cleanly fall back on v1 libraries.
    u8_one = np.ones(1, dtype=np.uint8)
    f32_one = np.ones(1, dtype=np.float32)
    lut3 = np.ones(3, dtype=np.float64)
    lut2 = np.ones(2, dtype=np.float64)
    assert (
        native_core.apply_compact_native(
            1, 1, 1, u8_one, None, u8_one, lut3, lut3, None, lut2, 0.1, 0.2, 0.3, 0.4, 1.0, np.ones(1)
        )
        is None
    )
    assert (
        native_core.damped_jacobi_compact_f32_native(
            1, 1, 1, u8_one, None, u8_one, lut3, lut3, None, lut2, 0.1, 0.2, 0.3, 0.4, 1.0, 0.7, f32_one, None, 2, zero_init=True
        )
        is None
    )
    assert (
        native_core.residual_restrict_2x2_compact_f32_native(
            1, 1, 1, u8_one, None, u8_one, lut3, lut3, None, lut2, 0.1, 0.2, 0.3, 0.4, 1.0, f32_one, f32_one
        )
        is None
    )
    assert (
        native_core.coarsen_compact_to_f32_native(
            1, 1, 1, u8_one, None, u8_one, lut3, lut3, None, lut2, 0.1, 0.2, 0.3, 0.4
        )
        is None
    )
    empty_f32 = np.empty(0, dtype=np.float32)
    assert (
        native_core.damped_jacobi_f32_native(
            1, 1, 1, empty_f32, empty_f32, empty_f32, f32_one, f32_one, f32_one, None, 2, zero_init=True
        )
        is None
    )
    assert (
        native_core.residual_restrict_2x2_f32_native(
            1, 1, 1, empty_f32, empty_f32, empty_f32, f32_one, f32_one, f32_one
        )
        is None
    )
    fine_f32 = np.zeros(1, dtype=np.float32)
    assert native_core.prolong_add_2x2_f32_native(1, 1, 1, f32_one, fine_f32) is False
    np.testing.assert_array_equal(fine_f32, np.zeros(1, dtype=np.float32))

    # v4 fused PCG/vector helpers also cleanly fall back on v1 libraries.
    f64_one = np.ones(1, dtype=np.float64)
    assert (
        native_core.apply_compact_dot_native(
            1, 1, 1, u8_one, None, u8_one, lut3, lut3, None, lut2, 0.1, 0.2, 0.3, 0.4, 1.0, f64_one
        )
        is None
    )
    assert (
        native_core.pcg_update_xr_norm2_native(
            f64_one.copy(), f64_one.copy(), f64_one, f64_one, 0.5
        )
        is None
    )
    assert native_core.pcg_update_p_native(f64_one.copy(), f64_one, 0.5) is False
    assert native_core.f64_to_f32_native(f64_one, f32_one.copy()) is False
    assert native_core.f32_to_f64_dot_native(f32_one, f64_one, f64_one.copy()) is None
    assert native_core.dot_f64_native(f64_one, f64_one) is None


def _make_compact_problem(layer_count, rows, cols, seed, with_hs=True, with_vias=True):
    rng = np.random.default_rng(seed)
    copper_mask = (rng.random((layer_count, rows, cols)) > 0.45).astype(np.uint8)
    if with_hs:
        hs_mask = (rng.random((rows, cols)) > 0.6).astype(np.uint8)
    else:
        hs_mask = np.zeros((rows, cols), dtype=np.uint8)

    if layer_count > 1:
        if with_vias:
            via_enh = rng.uniform(1.0, 50.0, size=(rows, cols)).astype(np.float32)
        else:
            via_enh = np.ones((rows, cols), dtype=np.float32)
        gz_scale = rng.uniform(0.02, 0.5, size=layer_count - 1).astype(np.float64)
    else:
        via_enh = None
        gz_scale = None

    gx_lut = np.empty(layer_count * 3, dtype=np.float64)
    gy_lut = np.empty(layer_count * 3, dtype=np.float64)
    cap_lut = np.empty(layer_count * 2, dtype=np.float64)
    for layer in range(layer_count):
        base_x = rng.uniform(0.05, 0.4)
        base_y = rng.uniform(0.05, 0.4)
        gx_lut[layer * 3 : layer * 3 + 3] = [base_x, base_x * 2.5, base_x * 12.0]
        gy_lut[layer * 3 : layer * 3 + 3] = [base_y, base_y * 2.5, base_y * 12.0]
        cap_fr4 = rng.uniform(0.01, 0.05)
        cap_lut[layer * 2 : layer * 2 + 2] = [cap_fr4, cap_fr4 + rng.uniform(0.02, 0.08)]

    pad_cap_cell = float(rng.uniform(0.01, 0.2))
    top_diag = float(rng.uniform(0.005, 0.05))
    bot_air_diag = float(rng.uniform(0.005, 0.05))
    bot_pad_diag = float(rng.uniform(0.05, 0.5))
    return (
        copper_mask,
        via_enh,
        hs_mask,
        gx_lut,
        gy_lut,
        gz_scale,
        cap_lut,
        pad_cap_cell,
        top_diag,
        bot_air_diag,
        bot_pad_diag,
    )


def _expand_compact_to_reference(
    layer_count,
    rows,
    cols,
    copper_mask,
    via_enh,
    hs_mask,
    gx_lut,
    gy_lut,
    gz_scale,
    cap_lut,
    pad_cap_cell,
    top_diag,
    bot_air_diag,
    bot_pad_diag,
):
    cm = copper_mask.astype(np.int64)
    hs = hs_mask.astype(bool)

    gx = np.empty((layer_count, rows, max(cols - 1, 0)), dtype=np.float64)
    gy = np.empty((layer_count, max(rows - 1, 0), cols), dtype=np.float64)
    for layer in range(layer_count):
        if cols > 1:
            gx[layer] = gx_lut[layer * 3 + cm[layer, :, :-1] + cm[layer, :, 1:]]
        if rows > 1:
            gy[layer] = gy_lut[layer * 3 + cm[layer, :-1, :] + cm[layer, 1:, :]]

    if layer_count > 1:
        gz = gz_scale[:, None, None] * via_enh.astype(np.float64)[None, :, :]
    else:
        gz = np.empty((0, rows, cols), dtype=np.float64)

    boundary_3d = np.zeros((layer_count, rows, cols), dtype=np.float64)
    boundary_3d[0, :, :] += top_diag
    boundary_3d[layer_count - 1, :, :] += np.where(hs, bot_pad_diag, bot_air_diag)

    cap_3d = np.empty((layer_count, rows, cols), dtype=np.float64)
    for layer in range(layer_count):
        cap_3d[layer] = cap_lut[layer * 2 + cm[layer]]
    cap_3d[layer_count - 1, hs] += pad_cap_cell

    return gx, gy, gz, boundary_3d.reshape(-1), cap_3d.reshape(-1)


_COMPACT_SHAPES = [
    (1, 1, 1),
    (1, 1, 5),
    (1, 5, 1),
    (1, 7, 9),
    (1, 8, 10),
    (4, 1, 1),
    (4, 7, 9),
    (4, 16, 18),
    (4, 33, 35),
]


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _COMPACT_SHAPES)
@pytest.mark.parametrize("scale", [0.0, 2.5])
@pytest.mark.parametrize("with_hs,with_vias", [(True, True), (False, False)])
def test_compiled_apply_compact_matches_reference(
    compiled_native_core, monkeypatch, layer_count, rows, cols, scale, with_hs, with_vias
):
    params = _make_compact_problem(
        layer_count,
        rows,
        cols,
        seed=500 + layer_count * 100 + rows * 7 + cols,
        with_hs=with_hs,
        with_vias=with_vias,
    )
    gx, gy, gz, boundary, cap = _expand_compact_to_reference(
        layer_count, rows, cols, *params
    )
    diag_mass = boundary + scale * cap
    rng = np.random.default_rng(77 + layer_count * 31 + rows * 3 + cols)
    vec = rng.normal(25.0, 8.0, size=layer_count * rows * cols)
    expected = _numpy_reference_dot(layer_count, rows, cols, gx, gy, gz, diag_mass, vec)

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)
    actual = native_core.apply_compact_native(
        layer_count, rows, cols, *params, scale, vec
    )
    assert actual is not None
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    out_buf = np.empty_like(vec)
    actual_out = native_core.apply_compact_native(
        layer_count, rows, cols, *params, scale, vec, out=out_buf
    )
    assert actual_out is out_buf
    np.testing.assert_allclose(out_buf, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _COMPACT_SHAPES + [(1, 3, 1050)])
@pytest.mark.parametrize("scale", [0.0, 1.75])
@pytest.mark.parametrize("zero_init,iterations", [(True, 1), (True, 2), (False, 2), (False, 0), (True, 0)])
def test_compiled_damped_jacobi_compact_f32_matches_reference(
    compiled_native_core,
    monkeypatch,
    layer_count,
    rows,
    cols,
    scale,
    zero_init,
    iterations,
):
    params = _make_compact_problem(
        layer_count, rows, cols, seed=600 + layer_count * 100 + rows * 7 + cols
    )
    gx, gy, gz, boundary, cap = _expand_compact_to_reference(
        layer_count, rows, cols, *params
    )
    diag_mass = boundary + scale * cap
    diag_3d = diag_mass.reshape(layer_count, rows, cols).copy()
    if cols > 1:
        diag_3d[:, :, :-1] += gx
        diag_3d[:, :, 1:] += gx
    if rows > 1:
        diag_3d[:, :-1, :] += gy
        diag_3d[:, 1:, :] += gy
    if layer_count > 1:
        diag_3d[:-1, :, :] += gz
        diag_3d[1:, :, :] += gz

    omega = 0.72
    omega_inv_diag = omega / diag_3d.reshape(-1)

    rng = np.random.default_rng(88 + layer_count * 31 + rows * 3 + cols)
    rhs_f32 = rng.normal(5.0, 2.0, size=layer_count * rows * cols).astype(np.float32)
    x0_f32 = rng.normal(15.0, 4.0, size=layer_count * rows * cols).astype(np.float32)

    expected = _numpy_reference_damped_jacobi(
        layer_count,
        rows,
        cols,
        gx,
        gy,
        gz,
        diag_mass,
        omega_inv_diag,
        rhs_f32.astype(np.float64),
        x0_f32.astype(np.float64),
        iterations,
        zero_init,
    )

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)
    scratch = np.empty(layer_count * rows * cols, dtype=np.float32)
    if zero_init:
        actual = native_core.damped_jacobi_compact_f32_native(
            layer_count,
            rows,
            cols,
            *params,
            scale,
            omega,
            rhs_f32,
            None,
            iterations,
            zero_init=True,
            scratch=scratch,
        )
    else:
        buf = x0_f32.copy()
        actual = native_core.damped_jacobi_compact_f32_native(
            layer_count,
            rows,
            cols,
            *params,
            scale,
            omega,
            rhs_f32,
            buf,
            iterations,
            zero_init=False,
            scratch=scratch,
        )
        assert actual is buf

    assert actual is not None
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual.astype(np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _COMPACT_SHAPES + [(1, 3, 1050)])
@pytest.mark.parametrize("scale", [0.0, 2.0])
def test_compiled_residual_restrict_2x2_compact_f32_matches_reference(
    compiled_native_core, monkeypatch, layer_count, rows, cols, scale
):
    params = _make_compact_problem(
        layer_count, rows, cols, seed=700 + layer_count * 100 + rows * 7 + cols
    )
    gx, gy, gz, boundary, cap = _expand_compact_to_reference(
        layer_count, rows, cols, *params
    )
    diag_mass = boundary + scale * cap
    rng = np.random.default_rng(99 + layer_count * 31 + rows * 3 + cols)
    rhs_f32 = rng.normal(5.0, 2.0, size=layer_count * rows * cols).astype(np.float32)
    x_f32 = rng.normal(15.0, 4.0, size=layer_count * rows * cols).astype(np.float32)

    ax = _numpy_reference_dot(
        layer_count, rows, cols, gx, gy, gz, diag_mass, x_f32.astype(np.float64)
    )
    expected = thermal_solver._restrict_2x2(
        (rhs_f32.astype(np.float64) - ax).reshape(layer_count, rows, cols)
    ).reshape(-1)

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)
    actual = native_core.residual_restrict_2x2_compact_f32_native(
        layer_count, rows, cols, *params, scale, rhs_f32, x_f32
    )
    assert actual is not None
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual.astype(np.float64), expected, rtol=1e-5, atol=1e-5)

    out_buf = np.empty_like(actual)
    actual_out = native_core.residual_restrict_2x2_compact_f32_native(
        layer_count, rows, cols, *params, scale, rhs_f32, x_f32, out=out_buf
    )
    assert actual_out is out_buf
    np.testing.assert_allclose(out_buf.astype(np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _COMPACT_SHAPES)
@pytest.mark.parametrize("with_hs,with_vias", [(True, True), (False, False)])
def test_compiled_coarsen_compact_to_f32_matches_reference(
    compiled_native_core, monkeypatch, layer_count, rows, cols, with_hs, with_vias
):
    params = _make_compact_problem(
        layer_count,
        rows,
        cols,
        seed=800 + layer_count * 100 + rows * 7 + cols,
        with_hs=with_hs,
        with_vias=with_vias,
    )
    gx, gy, gz, boundary, cap = _expand_compact_to_reference(
        layer_count, rows, cols, *params
    )
    fine_op = thermal_solver.StructuredThermalOperator(
        layer_count=layer_count,
        rows=rows,
        cols=cols,
        gx=gx,
        gy=gy,
        gz=gz,
        boundary_diag=boundary,
        stiffness_diag=np.zeros_like(boundary),
    )
    ref_coarse = thermal_solver._coarsen_structured_operator(fine_op)
    ref_cap_c = thermal_solver._restrict_2x2(
        cap.reshape(layer_count, rows, cols)
    ).reshape(-1)

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)
    res = native_core.coarsen_compact_to_f32_native(layer_count, rows, cols, *params)
    assert res is not None
    gx_c, gy_c, gz_c, boundary_c, stiffness_diag_c, cap_c = res

    for arr in (gx_c, gy_c, gz_c, boundary_c, stiffness_diag_c, cap_c):
        assert arr.dtype == np.float32
        assert arr.flags.c_contiguous

    assert gx_c.shape == ref_coarse.gx.shape
    assert gy_c.shape == ref_coarse.gy.shape
    assert gz_c.shape == ref_coarse.gz.shape
    assert boundary_c.shape == ref_coarse.boundary_diag.shape
    assert stiffness_diag_c.shape == ref_coarse.stiffness_diag.shape
    assert cap_c.shape == ref_cap_c.shape

    np.testing.assert_allclose(gx_c.astype(np.float64), ref_coarse.gx, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gy_c.astype(np.float64), ref_coarse.gy, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gz_c.astype(np.float64), ref_coarse.gz, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(boundary_c.astype(np.float64), ref_coarse.boundary_diag, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(stiffness_diag_c.astype(np.float64), ref_coarse.stiffness_diag, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cap_c.astype(np.float64), ref_cap_c, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _MG_SHAPES + [(1, 3, 1050)])
@pytest.mark.parametrize("zero_init,iterations", [(True, 1), (True, 2), (False, 2), (False, 0), (True, 0)])
def test_compiled_damped_jacobi_f32_matches_reference(
    compiled_native_core, monkeypatch, layer_count, rows, cols, zero_init, iterations
):
    gx, gy, gz, diag_mass, omega_inv_diag, rhs, x0 = _make_random_problem(
        layer_count, rows, cols, seed=900 + layer_count * 100 + rows * 7 + cols
    )
    gx_f = gx.astype(np.float32)
    gy_f = gy.astype(np.float32)
    gz_f = gz.astype(np.float32)
    dm_f = diag_mass.astype(np.float32)
    wi_f = omega_inv_diag.astype(np.float32)
    rhs_f = rhs.astype(np.float32)
    x0_f = x0.astype(np.float32)

    expected = _numpy_reference_damped_jacobi(
        layer_count,
        rows,
        cols,
        gx_f.astype(np.float64),
        gy_f.astype(np.float64),
        gz_f.astype(np.float64),
        dm_f.astype(np.float64),
        wi_f.astype(np.float64),
        rhs_f.astype(np.float64),
        x0_f.astype(np.float64),
        iterations,
        zero_init,
    )

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)
    if zero_init:
        actual = native_core.damped_jacobi_f32_native(
            layer_count, rows, cols, gx_f, gy_f, gz_f, dm_f, wi_f, rhs_f, None, iterations, zero_init=True
        )
    else:
        buf = x0_f.copy()
        actual = native_core.damped_jacobi_f32_native(
            layer_count, rows, cols, gx_f, gy_f, gz_f, dm_f, wi_f, rhs_f, buf, iterations, zero_init=False
        )
        assert actual is buf

    assert actual is not None
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual.astype(np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _MG_SHAPES + [(1, 3, 1050)])
def test_compiled_residual_restrict_2x2_f32_matches_reference(
    compiled_native_core, monkeypatch, layer_count, rows, cols
):
    gx, gy, gz, diag_mass, _, rhs, x0 = _make_random_problem(
        layer_count, rows, cols, seed=1000 + layer_count * 100 + rows * 7 + cols
    )
    gx_f = gx.astype(np.float32)
    gy_f = gy.astype(np.float32)
    gz_f = gz.astype(np.float32)
    dm_f = diag_mass.astype(np.float32)
    rhs_f = rhs.astype(np.float32)
    x0_f = x0.astype(np.float32)

    ax = _numpy_reference_dot(
        layer_count,
        rows,
        cols,
        gx_f.astype(np.float64),
        gy_f.astype(np.float64),
        gz_f.astype(np.float64),
        dm_f.astype(np.float64),
        x0_f.astype(np.float64),
    )
    expected = thermal_solver._restrict_2x2(
        (rhs_f.astype(np.float64) - ax).reshape(layer_count, rows, cols)
    ).reshape(-1)

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)
    actual = native_core.residual_restrict_2x2_f32_native(
        layer_count, rows, cols, gx_f, gy_f, gz_f, dm_f, rhs_f, x0_f
    )
    assert actual is not None
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual.astype(np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _MG_SHAPES)
def test_compiled_prolong_add_2x2_f32_matches_reference(
    compiled_native_core, monkeypatch, layer_count, rows, cols
):
    rng = np.random.default_rng(1100 + layer_count * 100 + rows * 7 + cols)
    coarse_rows = (rows + 1) // 2
    coarse_cols = (cols + 1) // 2
    coarse_f = rng.normal(1.5, 4.0, size=layer_count * coarse_rows * coarse_cols).astype(np.float32)
    fine_f = rng.normal(25.0, 5.0, size=layer_count * rows * cols).astype(np.float32)

    expected = _numpy_reference_prolong_add(
        layer_count, rows, cols, coarse_f.astype(np.float64), fine_f.astype(np.float64)
    )
    actual = fine_f.copy()
    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)
    ok = native_core.prolong_add_2x2_f32_native(
        layer_count, rows, cols, coarse_f, actual
    )
    assert ok is True
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual.astype(np.float64), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
def test_clear_native_scratch_releases_tls_buffers(compiled_native_core, monkeypatch):
    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)
    gx, gy, gz, diag_mass, omega_inv_diag, rhs, _ = _make_random_problem(2, 6, 8, seed=1200)
    native_core.damped_jacobi_native(
        2, 6, 8, gx, gy, gz, diag_mass, omega_inv_diag, rhs, None, 1, zero_init=True
    )
    native_core.damped_jacobi_f32_native(
        2,
        6,
        8,
        gx.astype(np.float32),
        gy.astype(np.float32),
        gz.astype(np.float32),
        diag_mass.astype(np.float32),
        omega_inv_diag.astype(np.float32),
        rhs.astype(np.float32),
        None,
        1,
        zero_init=True,
    )
    assert hasattr(compiled_native_core._tls, "jacobi_scratch")
    assert hasattr(compiled_native_core._tls, "jacobi_scratch_f32")

    native_core.clear_native_scratch()
    assert not hasattr(compiled_native_core._tls, "jacobi_scratch")
    assert not hasattr(compiled_native_core._tls, "jacobi_scratch_f32")


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("layer_count,rows,cols", _COMPACT_SHAPES + [(1, 80, 100), (4, 80, 100)])
@pytest.mark.parametrize("scale", [0.0, 2.5])
@pytest.mark.parametrize("with_hs,with_vias", [(True, True), (False, False)])
def test_compiled_apply_compact_dot_matches_reference(
    compiled_native_core, monkeypatch, layer_count, rows, cols, scale, with_hs, with_vias
):
    params = _make_compact_problem(
        layer_count,
        rows,
        cols,
        seed=1300 + layer_count * 100 + rows * 7 + cols,
        with_hs=with_hs,
        with_vias=with_vias,
    )
    gx, gy, gz, boundary, cap = _expand_compact_to_reference(
        layer_count, rows, cols, *params
    )
    diag_mass = boundary + scale * cap
    rng = np.random.default_rng(1400 + layer_count * 31 + rows * 3 + cols)
    vec = rng.normal(25.0, 8.0, size=layer_count * rows * cols)
    expected = _numpy_reference_dot(layer_count, rows, cols, gx, gy, gz, diag_mass, vec)
    expected_dot = float(np.dot(vec, expected))

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)

    def _forbid_copy(*args, **kwargs):
        raise AssertionError("Unexpected array copy on contiguous inputs")

    with monkeypatch.context() as mp:
        mp.setattr(native_core.np, "ascontiguousarray", _forbid_copy)
        out_buf = np.empty_like(vec)
        res = native_core.apply_compact_dot_native(
            layer_count, rows, cols, *params, scale, vec, out=out_buf
        )
    assert res is not None
    actual_out, actual_dot = res
    assert actual_out is out_buf
    np.testing.assert_allclose(actual_out, expected, rtol=1e-12, atol=1e-12)
    assert actual_dot == pytest.approx(expected_dot, rel=1e-12, abs=1e-12)

    res2 = native_core.apply_compact_dot_native(
        layer_count, rows, cols, *params, scale, vec
    )
    assert res2 is not None
    actual2, dot2 = res2
    np.testing.assert_allclose(actual2, expected, rtol=1e-12, atol=1e-12)
    assert dot2 == pytest.approx(expected_dot, rel=1e-12, abs=1e-12)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("n", [0, 1, 7, 100000])
def test_compiled_pcg_update_xr_norm2_matches_numpy(compiled_native_core, monkeypatch, n):
    rng = np.random.default_rng(1500 + n)
    x = rng.normal(25.0, 5.0, size=n)
    r = rng.normal(0.0, 2.0, size=n)
    p = rng.normal(0.0, 3.0, size=n)
    ap = rng.normal(0.0, 4.0, size=n)
    alpha = 0.37

    expected_x = x + alpha * p
    expected_r = r - alpha * ap
    expected_norm2 = float(np.dot(expected_r, expected_r))

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)

    def _forbid_copy(*args, **kwargs):
        raise AssertionError("Unexpected array copy on contiguous float64 inputs")

    with monkeypatch.context() as mp:
        mp.setattr(native_core.np, "ascontiguousarray", _forbid_copy)
        norm2 = native_core.pcg_update_xr_norm2_native(x, r, p, ap, alpha)

    assert norm2 is not None
    assert norm2 == pytest.approx(expected_norm2, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(x, expected_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(r, expected_r, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("n", [0, 1, 7, 100000])
def test_compiled_pcg_update_p_matches_numpy(compiled_native_core, monkeypatch, n):
    rng = np.random.default_rng(1600 + n)
    p = rng.normal(1.0, 2.0, size=n)
    z = rng.normal(-0.5, 3.0, size=n)
    beta = 0.618

    expected_p = z + beta * p

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)

    def _forbid_copy(*args, **kwargs):
        raise AssertionError("Unexpected array copy on contiguous float64 inputs")

    with monkeypatch.context() as mp:
        mp.setattr(native_core.np, "ascontiguousarray", _forbid_copy)
        ok = native_core.pcg_update_p_native(p, z, beta)

    assert ok is True
    np.testing.assert_allclose(p, expected_p, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("n", [0, 1, 7, 100000])
def test_compiled_f64_to_f32_matches_numpy(compiled_native_core, monkeypatch, n):
    rng = np.random.default_rng(1700 + n)
    src = rng.normal(0.0, 10.0, size=n)
    dst = np.empty(n, dtype=np.float32)
    expected = src.astype(np.float32)

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)

    def _forbid_copy(*args, **kwargs):
        raise AssertionError("Unexpected array copy on contiguous inputs")

    with monkeypatch.context() as mp:
        mp.setattr(native_core.np, "ascontiguousarray", _forbid_copy)
        ok = native_core.f64_to_f32_native(src, dst)

    assert ok is True
    np.testing.assert_array_equal(dst, expected)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("n", [0, 1, 7, 100000])
def test_compiled_f32_to_f64_dot_matches_numpy(compiled_native_core, monkeypatch, n):
    rng = np.random.default_rng(1800 + n)
    src = rng.normal(0.0, 5.0, size=n).astype(np.float32)
    r = rng.normal(1.0, 3.0, size=n)
    dst = np.empty(n, dtype=np.float64)

    expected_dst = src.astype(np.float64)
    expected_dot = float(np.dot(r, expected_dst))

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)

    def _forbid_copy(*args, **kwargs):
        raise AssertionError("Unexpected array copy on contiguous inputs")

    with monkeypatch.context() as mp:
        mp.setattr(native_core.np, "ascontiguousarray", _forbid_copy)
        dot = native_core.f32_to_f64_dot_native(src, r, dst)

    assert dot is not None
    np.testing.assert_array_equal(dst, expected_dst)
    assert dot == pytest.approx(expected_dot, rel=1e-12, abs=1e-12)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("n", [0, 1, 7, 100000])
def test_compiled_dot_f64_matches_numpy(compiled_native_core, monkeypatch, n):
    rng = np.random.default_rng(1900 + n)
    a = rng.normal(2.0, 4.0, size=n)
    b = rng.normal(-1.0, 3.0, size=n)

    expected_ab = float(np.dot(a, b))
    expected_aa = float(np.dot(a, a))

    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)

    def _forbid_copy(*args, **kwargs):
        raise AssertionError("Unexpected array copy on contiguous float64 inputs")

    with monkeypatch.context() as mp:
        mp.setattr(native_core.np, "ascontiguousarray", _forbid_copy)
        dot_ab = native_core.dot_f64_native(a, b)
        dot_aa = native_core.dot_f64_native(a, a)

    assert dot_ab is not None
    assert dot_aa is not None
    assert dot_ab == pytest.approx(expected_ab, rel=1e-12, abs=1e-12)
    assert dot_aa == pytest.approx(expected_aa, rel=1e-12, abs=1e-12)


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
def test_v4_vector_helpers_reject_invalid_buffers(compiled_native_core, monkeypatch):
    monkeypatch.setattr(native_core, "NATIVE_CORE", compiled_native_core)
    a = np.ones(4, dtype=np.float64)
    ro = np.ones(4, dtype=np.float64)
    ro.flags.writeable = False
    f32_ro = np.ones(4, dtype=np.float32)
    f32_ro.flags.writeable = False
    mismatch = np.ones(3, dtype=np.float64)

    assert native_core.pcg_update_xr_norm2_native(ro, a.copy(), a, a, 1.0) is None
    assert native_core.pcg_update_xr_norm2_native(a.copy(), a.copy(), mismatch, a, 1.0) is None
    assert native_core.pcg_update_p_native(ro, a, 1.0) is False
    assert native_core.pcg_update_p_native(a.copy(), mismatch, 1.0) is False
    assert native_core.f64_to_f32_native(a, f32_ro) is False
    assert native_core.f64_to_f32_native(mismatch, np.empty(4, dtype=np.float32)) is False
    assert native_core.f32_to_f64_dot_native(np.ones(4, dtype=np.float32), a, ro) is None
    assert native_core.dot_f64_native(a, mismatch) is None


@pytest.mark.skipif(not _HAS_LINUX_GPP, reason="Requires Linux with g++")
@pytest.mark.parametrize("env_val", ["2", "0", "invalid"])
def test_omp_threads_env_override_in_subprocess(compiled_native_core, env_val):
    import os

    pkg_dir = Path(native_core.__file__).resolve().parent
    code = (
        "import numpy as np\n"
        "import native_core\n"
        "a = np.arange(100000, dtype=np.float64) * 0.001\n"
        "b = np.cos(a)\n"
        "d = native_core.dot_f64_native(a, b)\n"
        "ref = float(np.dot(a, b))\n"
        "assert d is not None and abs(d - ref) <= 1e-12 * max(1.0, abs(ref)), (d, ref)\n"
    )
    env = dict(os.environ)
    env["THERMALSIM_NATIVE_CORE"] = compiled_native_core.path
    env["THERMALSIM_OMP_THREADS"] = env_val
    env["PYTHONPATH"] = str(pkg_dir)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(pkg_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
