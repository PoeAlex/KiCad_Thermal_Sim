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
