"""Tests for the optional native CPU-core loader."""

from pathlib import Path

import numpy as np

from ThermalSim import native_core


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
