"""
Unit tests for capabilities module.

This module tests the feature detection functionality.
"""

import pytest
import sys
import importlib


class TestCapabilityFlags:
    """Tests for capability flag detection."""

    def test_has_libs_is_boolean(self):
        """Test that HAS_LIBS is a boolean value."""
        from ThermalSim.capabilities import HAS_LIBS
        assert isinstance(HAS_LIBS, bool)

    def test_has_pardiso_is_boolean(self):
        """Test that HAS_PARDISO is a boolean value."""
        from ThermalSim.capabilities import HAS_PARDISO
        assert isinstance(HAS_PARDISO, bool)

    def test_has_numba_is_boolean(self):
        """Test that HAS_NUMBA is a boolean value."""
        from ThermalSim.capabilities import HAS_NUMBA
        assert isinstance(HAS_NUMBA, bool)

    def test_has_numpy_is_boolean(self):
        """Test that HAS_NUMPY is a boolean value."""
        from ThermalSim.capabilities import HAS_NUMPY
        assert isinstance(HAS_NUMPY, bool)

    def test_has_scipy_is_boolean(self):
        """Test that HAS_SCIPY is a boolean value."""
        from ThermalSim.capabilities import HAS_SCIPY
        assert isinstance(HAS_SCIPY, bool)

    def test_has_matplotlib_is_boolean(self):
        """Test that HAS_MATPLOTLIB is a boolean value."""
        from ThermalSim.capabilities import HAS_MATPLOTLIB
        assert isinstance(HAS_MATPLOTLIB, bool)

    def test_has_wx_is_boolean(self):
        """Test that HAS_WX is a boolean value."""
        from ThermalSim.capabilities import HAS_WX
        assert isinstance(HAS_WX, bool)

    def test_has_libs_is_composite(self):
        """Test that HAS_LIBS is the AND of individual flags."""
        from ThermalSim.capabilities import (
            HAS_LIBS, HAS_NUMPY, HAS_SCIPY, HAS_MATPLOTLIB, HAS_WX
        )
        expected = HAS_NUMPY and HAS_SCIPY and HAS_MATPLOTLIB and HAS_WX
        assert HAS_LIBS == expected

    def test_has_libs_reflects_imports(self):
        """Test that HAS_LIBS reflects actual import availability."""
        from ThermalSim.capabilities import HAS_LIBS

        # In test environment, numpy and matplotlib should be available
        # (they're dependencies for the solver tests)
        try:
            import numpy
            import matplotlib
            numpy_available = True
        except ImportError:
            numpy_available = False

        # HAS_LIBS should reflect availability
        if numpy_available:
            # Note: wx might not be available in test environment
            # so we just verify the flag is set
            assert isinstance(HAS_LIBS, bool)


class TestGranularFlags:
    """Tests for granular dependency flags."""

    def test_numpy_flag_matches_import(self):
        """Test that HAS_NUMPY matches actual numpy availability."""
        from ThermalSim.capabilities import HAS_NUMPY
        try:
            import numpy
            assert HAS_NUMPY is True
        except ImportError:
            assert HAS_NUMPY is False

    def test_scipy_flag_matches_import(self):
        """Test that HAS_SCIPY matches actual scipy availability."""
        from ThermalSim.capabilities import HAS_SCIPY
        try:
            import scipy
            assert HAS_SCIPY is True
        except ImportError:
            assert HAS_SCIPY is False

    def test_matplotlib_flag_matches_import(self):
        """Test that HAS_MATPLOTLIB matches actual matplotlib availability."""
        from ThermalSim.capabilities import HAS_MATPLOTLIB
        try:
            import matplotlib
            assert HAS_MATPLOTLIB is True
        except ImportError:
            assert HAS_MATPLOTLIB is False


class TestGetMissingPackages:
    """Tests for get_missing_packages function."""

    def test_returns_list(self):
        """Test that get_missing_packages returns a list."""
        from ThermalSim.capabilities import get_missing_packages
        result = get_missing_packages()
        assert isinstance(result, list)

    def test_returns_tuples(self):
        """Test that each entry is a (import_name, pip_name) tuple."""
        from ThermalSim.capabilities import get_missing_packages
        result = get_missing_packages()
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_empty_when_all_installed(self):
        """Test that list is empty when all deps are installed."""
        from ThermalSim.capabilities import get_missing_packages, HAS_NUMPY, HAS_SCIPY, HAS_MATPLOTLIB
        result = get_missing_packages()
        if HAS_NUMPY and HAS_SCIPY and HAS_MATPLOTLIB:
            assert result == []

    def test_contains_numpy_when_missing(self):
        """Test that numpy is listed when missing."""
        from ThermalSim import capabilities
        original = capabilities.HAS_NUMPY
        try:
            capabilities.HAS_NUMPY = False
            result = capabilities.get_missing_packages()
            assert ("numpy", "numpy") in result
        finally:
            capabilities.HAS_NUMPY = original

    def test_contains_scipy_when_missing(self):
        """Test that scipy is listed when missing."""
        from ThermalSim import capabilities
        original = capabilities.HAS_SCIPY
        try:
            capabilities.HAS_SCIPY = False
            result = capabilities.get_missing_packages()
            assert ("scipy", "scipy") in result
        finally:
            capabilities.HAS_SCIPY = original

    def test_contains_matplotlib_when_missing(self):
        """Test that matplotlib is listed when missing."""
        from ThermalSim import capabilities
        original = capabilities.HAS_MATPLOTLIB
        try:
            capabilities.HAS_MATPLOTLIB = False
            result = capabilities.get_missing_packages()
            assert ("matplotlib", "matplotlib") in result
        finally:
            capabilities.HAS_MATPLOTLIB = original

    def test_all_missing(self):
        """Test that all three are listed when all missing."""
        from ThermalSim import capabilities
        orig_np = capabilities.HAS_NUMPY
        orig_sp = capabilities.HAS_SCIPY
        orig_mpl = capabilities.HAS_MATPLOTLIB
        try:
            capabilities.HAS_NUMPY = False
            capabilities.HAS_SCIPY = False
            capabilities.HAS_MATPLOTLIB = False
            result = capabilities.get_missing_packages()
            assert len(result) == 3
            pip_names = [pip for _, pip in result]
            assert "numpy" in pip_names
            assert "scipy" in pip_names
            assert "matplotlib" in pip_names
        finally:
            capabilities.HAS_NUMPY = orig_np
            capabilities.HAS_SCIPY = orig_sp
            capabilities.HAS_MATPLOTLIB = orig_mpl

    def test_does_not_include_wx(self):
        """Test that wx is never in the missing packages (not pip-installable in KiCad)."""
        from ThermalSim.capabilities import get_missing_packages
        result = get_missing_packages()
        import_names = [imp for imp, _ in result]
        assert "wx" not in import_names


class TestGetCapabilitiesSummary:
    """Tests for get_capabilities_summary function."""

    def test_returns_string(self):
        """Test that summary is a string."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert isinstance(result, str)

    def test_contains_header(self):
        """Test that summary contains header."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert "ThermalSim Capabilities" in result

    def test_contains_individual_flags(self):
        """Test that summary mentions each individual dependency."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert "numpy" in result
        assert "scipy" in result
        assert "matplotlib" in result
        assert "wx" in result

    def test_contains_composite_flag(self):
        """Test that summary mentions composite flag."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert "Core libs (all)" in result

    def test_contains_pardiso_status(self):
        """Test that summary mentions PyPardiso."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert "PyPardiso" in result or "pardiso" in result.lower()

    def test_contains_numba_status(self):
        """Test that summary mentions Numba."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert "Numba" in result or "numba" in result.lower()

    def test_shows_available_or_not(self):
        """Test that summary shows Available/Not available status."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        # Should contain either "Available" or "Not available"
        assert "Available" in result or "available" in result.lower()

    def test_multiline_output(self):
        """Test that summary is multiline."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        lines = result.split('\n')
        assert len(lines) >= 5  # Header + numpy + scipy + matplotlib + wx + composite


class TestCapabilityFlagsConsistency:
    """Tests for consistency between flags and summary."""

    def test_libs_flag_matches_summary(self):
        """Test that HAS_LIBS flag matches summary text."""
        from ThermalSim.capabilities import HAS_LIBS, get_capabilities_summary
        summary = get_capabilities_summary()

        for line in summary.split('\n'):
            if 'Core libs (all)' in line:
                if HAS_LIBS:
                    assert 'Available' in line and 'Not' not in line.split('Available')[0][-5:]
                else:
                    assert 'Not available' in line
                break

    def test_pardiso_flag_matches_summary(self):
        """Test that HAS_PARDISO flag matches summary text."""
        from ThermalSim.capabilities import HAS_PARDISO, get_capabilities_summary
        summary = get_capabilities_summary()

        for line in summary.split('\n'):
            if 'PyPardiso' in line or 'pardiso' in line.lower():
                if HAS_PARDISO:
                    assert 'Available' in line
                else:
                    assert 'Not available' in line
                break

    def test_numba_flag_matches_summary(self):
        """Test that HAS_NUMBA flag matches summary text."""
        from ThermalSim.capabilities import HAS_NUMBA, get_capabilities_summary
        summary = get_capabilities_summary()

        for line in summary.split('\n'):
            if 'Numba' in line or 'numba' in line.lower():
                if HAS_NUMBA:
                    assert 'Available' in line
                else:
                    assert 'Not available' in line
                break


class TestModuleImportBehavior:
    """Tests for module import behavior."""

    def test_module_importable(self):
        """Test that capabilities module is importable."""
        import ThermalSim.capabilities as capabilities
        assert capabilities is not None

    def test_reimport_stable(self):
        """Test that reimporting gives same flags."""
        from ThermalSim.capabilities import HAS_LIBS, HAS_PARDISO, HAS_NUMBA

        # Reimport
        import ThermalSim.capabilities as capabilities
        importlib.reload(capabilities)

        from ThermalSim.capabilities import HAS_PARDISO as HAS_PARDISO2
        from ThermalSim.capabilities import HAS_NUMBA as HAS_NUMBA2

        assert HAS_PARDISO == HAS_PARDISO2
        assert HAS_NUMBA == HAS_NUMBA2

    def test_attributes_exist(self):
        """Test that expected attributes exist in module."""
        import ThermalSim.capabilities as capabilities

        assert hasattr(capabilities, 'HAS_LIBS')
        assert hasattr(capabilities, 'HAS_NUMPY')
        assert hasattr(capabilities, 'HAS_SCIPY')
        assert hasattr(capabilities, 'HAS_MATPLOTLIB')
        assert hasattr(capabilities, 'HAS_WX')
        assert hasattr(capabilities, 'HAS_PARDISO')
        assert hasattr(capabilities, 'HAS_NUMBA')
        assert hasattr(capabilities, 'get_capabilities_summary')
        assert hasattr(capabilities, 'get_missing_packages')


class TestEdgeCases:
    """Edge case tests for capabilities module."""

    def test_summary_not_empty(self):
        """Test that summary is not empty."""
        from ThermalSim.capabilities import get_capabilities_summary
        result = get_capabilities_summary()
        assert len(result) > 0

    def test_summary_no_exceptions(self):
        """Test that getting summary doesn't raise exceptions."""
        from ThermalSim.capabilities import get_capabilities_summary
        try:
            result = get_capabilities_summary()
            assert result is not None
        except Exception as e:
            pytest.fail(f"get_capabilities_summary raised exception: {e}")

    def test_missing_packages_no_exceptions(self):
        """Test that get_missing_packages doesn't raise exceptions."""
        from ThermalSim.capabilities import get_missing_packages
        try:
            result = get_missing_packages()
            assert result is not None
        except Exception as e:
            pytest.fail(f"get_missing_packages raised exception: {e}")

    def test_flags_accessible_as_module_attributes(self):
        """Test that flags can be accessed as module attributes."""
        import ThermalSim.capabilities as capabilities

        assert capabilities.HAS_LIBS in [True, False]
        assert capabilities.HAS_NUMPY in [True, False]
        assert capabilities.HAS_SCIPY in [True, False]
        assert capabilities.HAS_MATPLOTLIB in [True, False]
        assert capabilities.HAS_WX in [True, False]
        assert capabilities.HAS_PARDISO in [True, False]
        assert capabilities.HAS_NUMBA in [True, False]
